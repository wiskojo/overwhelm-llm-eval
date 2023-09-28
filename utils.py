import os

import openai
import tiktoken
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")


@retry(stop=stop_after_attempt(10), wait=wait_fixed(5))
async def chat(messages, model, temperature, max_tokens=None, request_timeout=10):
    try:
        response = await openai.ChatCompletion.acreate(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages.

    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_token_costs(model: str) -> dict:
    """
    Returns the cost per token for prompt and sampled tokens based on the model.

    Args:
        model (str): The model name.

    Returns:
        dict: A dictionary with the cost per token for prompt and sampled tokens.

    Raises:
        ValueError: If the model is not recognized.
    """
    token_costs = {
        "gpt-3.5-turbo": {
            "prompt": 0.0000015,
            "completion": 0.000002,
        },
        "gpt-4": {"prompt": 0.00003, "completion": 0.00006},
        "meta-llama/Llama-2-70b-chat-hf": {
            "prompt": 0.000001,
            "completion": 0.000001,
        },
    }

    if model not in token_costs:
        raise ValueError(f"Token costs for model {model} are not implemented.")

    return token_costs[model]


def get_process_api_requests_params(model: str) -> dict:
    """
    Returns the parameters for processing requests based on the model.
    The constants used here are derived from the OpenAI rate limits page:
    https://platform.openai.com/account/rate-limits

    Args:
        model (str): The model name.

    Returns:
        dict: A dictionary with the parameters for processing requests.

    Raises:
        ValueError: If the model is not recognized.
    """
    process_api_requests_params = {
        "gpt-3.5-turbo": {
            "request_url": "https://api.openai.com/v1/chat/completions",
            "api_key": OPENAI_API_KEY,
            "max_requests_per_minute": 3500,
            "max_tokens_per_minute": 90000,
            "token_encoding_name": "cl100k_base",
        },
        "gpt-4": {
            "request_url": "https://api.openai.com/v1/chat/completions",
            "api_key": OPENAI_API_KEY,
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 40000,
            "token_encoding_name": "cl100k_base",
        },
        "meta-llama/Llama-2-70b-chat-hf": {
            "request_url": "https://api.deepinfra.com/v1/openai/chat/completions",
            "api_key": DEEPINFRA_API_KEY,
            "max_requests_per_minute": 200,
            "max_tokens_per_minute": 40000,
            "token_encoding_name": "cl100k_base",  # TODO: No Llama 2 tokenizer on tiktoken yet
        },
    }

    if model not in process_api_requests_params:
        raise ValueError(
            f"Processing request parameters for model {model} are not implemented."
        )

    return process_api_requests_params[model]
