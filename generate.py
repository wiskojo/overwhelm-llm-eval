import argparse
import asyncio
import json
import os
import re

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from utils import chat

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class DataGenerator:
    generate_rules_prompt = """I am looking to develop a guardrail AI model that will function as a preliminary filter before users interact with an AI chatbot. This model will prevent certain types of inputs from being processed by the chatbot, such as non-English phrases or queries about cars. This list does not necessarily need to focus on actual inappropriate or harmful content; rather, it can contain any random criteria you can think of. Each entry in the list should begin with "Do not..." and should be provided as separate lines without any numbering or bullet points. Ensure that these rules are highly specific to avoid imposing overly broad restrictions on the chatbot's conversational capacity. For instance, instead of a general rule like "Do not accept inputs related to space or astronomical topics", craft more focused rules such as "Do not accept inputs discussing Earth or Jupiter". Present this as a clean list without any additional text or commentary. These rules must be absolutely unambiguous, and clearly enforceable.

Good Examples:
1. Do not accept inputs written in French or Spanish.
2. Do not process queries specifically about Tesla or Ferrari cars.
3. Do not process sentences containing more than 30 words.
4. Do not entertain inputs with complex mathematical equations involving calculus.
5. Do not process discussions involving sushi, tacos, or burritos.
6. Do not accept inputs specifically mentioning the planets within our solar system.
7. Do not process inputs mentioning breeds of dogs.
8. Do not process inputs concerning the TV shows "Breaking Bad" or "Game of Thrones".
9. Do not accept queries regarding the Eiffel Tower or the Grand Canyon.
10.Do not process inputs containing more than five consecutive exclamation marks.

Bad Examples:
1. Do not process input with excessive usage of punctuations. # What is considered excessive?
2. Do not process overly technical queries. # What is considered "overly technical"?
3. Do not entertain controversial topics. # Which topics are considered "controversial"?
4. Do not process inputs about popular sports. # Which sports are considered "popular"?
5. Do not accept inputs with strong language. # What constitutes "strong language"?
6. Do not process discussions about major world events. # Which events are considered "major"?
7. Do not accept queries about famous personalities. # Who is considered a "famous personality"?
8. Do not process inputs that are too long. # How long is "too long"?
9. Do not entertain historical queries. # What time frame is considered "historical"?
10. Do not process discussions with slang. # What type of slang or which slangs are not allowed?

Please draft a list of {num_rules} specific types of inputs to be rejected following the good example, for testing purposes."""

    generate_reject_examples_prompt = """You are overseeing a QA process where a chatbot has been programmed with a set of rules. Your task is to devise high-quality, diverse test inputs that potential users might use but violate the set rule, and thus should be rejected by the chatbot. These inputs should encompass a broad spectrum of user requests, ranging from artistic endeavors and intellectual inquiries to practical advice and technical assistance, among others. Additionally, they should exhibit significant variation in elements such as length, complexity, thematic focus, structure, and disciplinary scope. Moreover, despite the variability in detectability and subtlety of the violations, each input must unequivocally breach the specified rule without any room for ambiguity or interpretation. Even the more subtle examples should, upon close inspection, clearly and indisputably violate the rules.

Format:
1. <example 1>
2. <example 2>
...
{num_examples}. <example {num_examples}>

Generate exactly {num_examples} such examples."""

    generate_accept_examples_prompt = """You are overseeing a QA process where a chatbot has been programmed with a set of rules. Your task is to devise high-quality, diverse test inputs that potential users might use and should strictly adhere to the set rules, thereby qualifying for acceptance by the chatbot. These inputs should span a broad range of user requests, from artistic endeavors and intellectual inquiries to practical advice and technical tasks, among others. Additionally, they should demonstrate significant variation in factors such as length, complexity, thematic concentration, structure, and disciplinary focus. Furthermore, regardless of the subtleties in detectability and presentation, each input must unequivocally align with the specified rules, leaving no room for ambiguity or interpretation. Even the nuanced examples should, upon close scrutiny, clearly and unquestionably conform to the rules.

Format:
1. <example 1>
2. <example 2>
...
{num_examples}. <example {num_examples}>

Generate exactly {num_examples} such examples."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: int = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.temperature = temperature

    async def generate_rules(self, num_rules: int) -> list[str]:
        messages = [
            {
                "role": "user",
                "content": self.generate_rules_prompt.format(num_rules=num_rules),
            },
        ]
        res = await chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            request_timeout=120,
        )
        rules = self._strip_numbering(res.split("\n"))
        return rules

    async def generate_reject_examples(self, rule: str, num_examples: int) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": self.generate_reject_examples_prompt.format(
                    num_examples=num_examples
                ),
            },
            {"role": "user", "content": f"Rule: {rule}"},
        ]
        res = await chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            request_timeout=60,
        )
        examples = self._strip_numbering(res.split("\n"))
        return examples

    async def generate_accept_examples(
        self, rules: list[str], num_examples: int
    ) -> list[str]:
        rules_str = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
        messages = [
            {
                "role": "user",
                "content": self.generate_accept_examples_prompt.format(
                    num_examples=num_examples
                ),
            },
            {"role": "user", "content": f"Rules:\n{rules_str}"},
        ]
        res = await chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            request_timeout=60,
        )
        examples = self._strip_numbering(res.split("\n"))
        return examples

    def _strip_numbering(self, input_list: list[str]) -> list[str]:
        stripped_list = []
        for item in input_list:
            # Check if the string starts with a number followed by a period and a space
            if re.match(r"^\d+\.\s", item):
                # Remove the numbering
                item_ = item.split(". ", 1)[-1]
                # Remove quotes if present
                item_ = re.sub(r"^['\"](.*)['\"]$", r"\1", item_)
                stripped_list.append(item_)
            else:
                # If the string doesn't start with a number and a period, leave it as is
                stripped_list.append(item)
        return stripped_list


async def main(args):
    generator = DataGenerator()

    data = {"rules": []}

    ## Step 1: Generate rules

    print("Starting rule generation...")
    rules = await generator.generate_rules(args.num_rules)

    print(f"Rule generation completed. Generated {len(rules)} rules.")
    for i, rule in enumerate(rules, 1):
        print(f"Rule {i}: {rule}")

    ## Step 2: Generate reject examples for each rule

    semaphore = asyncio.Semaphore(args.max_workers)

    async def run_with_limit(semaphore, rule, generator):
        async with semaphore:
            return await generator.generate_reject_examples(
                rule, num_examples=args.num_reject
            )

    pbar = tqdm(total=len(rules), desc="Processing tasks", ncols=100)
    tasks = []
    for rule in rules:
        task = asyncio.ensure_future(run_with_limit(semaphore, rule, generator))
        task.add_done_callback(lambda x: pbar.update(1))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    pbar.close()

    for rule, reject_examples in zip(rules, results):
        # Store rule and reject examples in a dictionary
        data["rules"].append(
            {
                "rule": rule,
                "reject_examples": reject_examples,
            }
        )

    ## Step 3: Generate accept examples for all rules

    print("\nStarting accept examples generation...")
    # Generate accept examples for all rules
    accept_examples = await generator.generate_accept_examples(
        rules, num_examples=args.num_accept
    )
    # Store accept examples in the dictionary
    data["accept_examples"] = accept_examples
    print(
        f"Accept examples generation completed. Generated {len(accept_examples)} examples..."
    )

    # Save data to a JSON file
    with open(args.save_path, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data.")
    parser.add_argument(
        "--num_rules", type=int, default=50, help="Number of rules to generate"
    )
    parser.add_argument(
        "--num_reject",
        type=int,
        default=10,
        help="Number of reject examples to generate per rule",
    )
    parser.add_argument(
        "--num_accept",
        type=int,
        default=40,
        help="Number of accept examples to generate",
    )
    parser.add_argument(
        "--max_workers", type=int, default=5, help="Maximum number of workers"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./data/rules.json",
        help="Path to save the generated data",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
