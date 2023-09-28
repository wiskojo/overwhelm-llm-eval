import argparse
import asyncio
import itertools as it
import json
import logging
import random
from typing import Any

from evaluator import Evaluator
from processor import process_api_requests_from_file
from utils import (
    get_process_api_requests_params,
    get_token_costs,
    num_tokens_from_messages,
)


def generate_baseline_requests(
    data: dict[str, Any],
    evaluator: Evaluator,
    num_reject_samples: int,
    num_accept_samples: int,
) -> list[dict]:
    all_accept_examples = data["accept_examples"]
    accept_examples = _sample_data(all_accept_examples, num_accept_samples)
    requests = []

    for rule_id, rule in enumerate(data["rules"]):
        rule_reject_examples = rule["reject_examples"]
        reject_examples = _sample_data(rule_reject_examples, num_reject_samples)

        # Combine reject and accept examples
        all_examples = reject_examples + accept_examples

        # Get batch requests
        batch_requests = evaluator.get_request_batch([rule["rule"]], all_examples)

        # Generate metadata for each request
        for idx, (example, request) in enumerate(zip(all_examples, batch_requests)):
            label = 1 if idx < len(reject_examples) else 0
            request_meta = {
                "trial_type": "baseline",
                "eval_model": evaluator.model,
                "rule_id": rule_id,
                "example": example,
                "label": label,
            }
            request["metadata"] = request_meta
            requests.append(request)

    return requests


def generate_experiment_requests(
    data: dict[str, Any],
    evaluator: Evaluator,
    num_reject_samples: int,
    num_accept_samples: int,
    num_rules_to_test: list[int],
    num_trials: int,
) -> list[dict]:
    all_accept_examples = data["accept_examples"]
    requests = []

    for num_rules in num_rules_to_test:
        for trial in range(num_trials):
            sampled_rules = random.sample(list(enumerate(data["rules"])), num_rules)
            sampled_rules_reject_examples = list(
                it.chain(*[rule[1]["reject_examples"] for rule in sampled_rules])
            )
            reject_examples = _sample_data(
                sampled_rules_reject_examples, num_reject_samples
            )
            accept_examples = _sample_data(all_accept_examples, num_accept_samples)

            # Combine reject and accept examples
            all_examples = reject_examples + accept_examples

            # Get batch requests
            batch_requests = evaluator.get_request_batch(
                [rule[1]["rule"] for rule in sampled_rules], all_examples
            )

            # Generate metadata for each request
            for idx, (example, request) in enumerate(zip(all_examples, batch_requests)):
                label = 1 if idx < len(reject_examples) else 0
                request_meta = {
                    "trial_type": "experiment",
                    "eval_model": evaluator.model,
                    "num_rules": num_rules,
                    "trial": trial + 1,
                    "rule_ids": [rule[0] for rule in sampled_rules],
                    "example": example,
                    "label": label,
                }
                request["metadata"] = request_meta
                requests.append(request)

    return requests


def _sample_data(examples: list[Any], num_samples: int) -> list[Any]:
    return (
        examples
        if len(examples) < num_samples
        else random.sample(examples, num_samples)
    )


async def main(args):
    random.seed(args.seed)

    with open(args.rules, "r") as file:
        data = json.load(file)

    evaluator = Evaluator(model=args.eval_model)
    baseline_requests = generate_baseline_requests(
        data,
        evaluator,
        num_reject_samples=args.num_reject_samples,
        num_accept_samples=args.num_accept_samples,
    )
    experiment_requests = generate_experiment_requests(
        data,
        evaluator,
        num_reject_samples=args.num_reject_samples,
        num_accept_samples=args.num_accept_samples,
        num_rules_to_test=args.num_rules_to_test,
        num_trials=args.num_trials,
    )
    requests = baseline_requests + experiment_requests
    if args.max_requests is not None:
        requests = requests[: args.max_requests]
    requests_jsonl = "\n".join(json.dumps(request) for request in requests)

    with open(args.requests, "w") as file:
        file.write(requests_jsonl)

    token_costs = get_token_costs(args.eval_model)
    prompt_tokens = sum([num_tokens_from_messages(req["messages"]) for req in requests])
    completion_tokens = sum([req["max_tokens"] for req in requests])
    total_cost = (
        prompt_tokens * token_costs["prompt"]
        + completion_tokens * token_costs["completion"]
    )

    if not args.skip_confirmation:
        print(f"Total number of requests: {len(requests):,}")
        print(f"Total prompt tokens: {prompt_tokens:,}")
        print(f"Total sampled tokens: {completion_tokens:,}")
        print(f"Estimated cost: ${total_cost:.4f}")
        should_process_request = input(
            "Do you want to proceed with processing the requests? (y/n)\n"
        )
        if should_process_request.lower() != "y":
            return

    process_params = get_process_api_requests_params(args.eval_model)

    await process_api_requests_from_file(
        requests_filepath=args.requests,
        save_filepath=args.results,
        request_url=process_params["request_url"],
        api_key=process_params["api_key"],
        max_requests_per_minute=process_params["max_requests_per_minute"],
        max_tokens_per_minute=process_params["max_tokens_per_minute"]
        * 0.9,  # Multiply by 0.9 to keep under the rate limit
        token_encoding_name=process_params["token_encoding_name"],
        max_attempts=5,
        logging_level=logging.INFO,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment.")
    parser.add_argument(
        "--rules", default="./data/rules.json", help="Path to the rules file"
    )
    parser.add_argument(
        "--requests", default="./data/requests.jsonl", help="Path to the requests file"
    )
    parser.add_argument(
        "--results", default="./data/results.jsonl", help="Path to the results file"
    )
    parser.add_argument("--num_trials", type=int, default=20, help="Number of trials")
    parser.add_argument(
        "--num_reject_samples", type=int, default=10, help="Number of reject samples"
    )
    parser.add_argument(
        "--num_accept_samples", type=int, default=40, help="Number of accept samples"
    )
    parser.add_argument(
        "--num_rules_to_test",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25, 50],
        help="List of number of rules to test",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4", "meta-llama/Llama-2-70b-chat-hf"],
        help="Evaluation model to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the random number generator",
    )
    parser.add_argument(
        "--skip_confirmation",
        action="store_true",
        help="Skip confirmation before processing the requests",
    )
    parser.add_argument(
        "--max_requests",
        type=int,
        default=None,
        help="Maximum number of requests to process. Useful for testing. Default is None, which means no limit.",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
