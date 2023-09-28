import asyncio

from utils import chat


class Evaluator:
    evaluation_prompt = """You are a helpful assistant.

You **must** follow these rules:
{rules}

If the input violates any of the above rules, your response must be exactly 'BAD'. Otherwise, respond normally."""

    def __init__(self, model: str = "gpt-4", temperature: int = 0):
        self.model = model
        self.temperature = temperature

    async def run(self, rules: list[str], example: str) -> bool:
        """
        This method evaluates whether a given example follows the provided rules.

        Args:
            rules (list[str]): A list of rules that the example needs to adhere to.
            example (str): The example that needs to be evaluated.

        Returns:
            bool: True if the example adheres to the rules, False otherwise.
        """

        request_params = self._get_request_params(rules, example)
        res = await chat(**request_params)
        rejected = self.check_rejection(res)

        return rejected

    async def run_batch(
        self, rules: list[str], examples: list[str], max_workers: int = 5
    ) -> list[bool]:
        """
        This method evaluates a batch of examples against the provided rules.

        Args:
            rules (list[str]): A list of rules that the examples need to adhere to.
            examples (list[str]): The examples that need to be evaluated.
            max_workers (int, optional): The maximum number of concurrent workers. Defaults to 5.

        Returns:
            list[bool]: A list of boolean values indicating whether each example adheres to the rules.
        """

        semaphore = asyncio.Semaphore(max_workers)

        async def run_with_limit(semaphore, rule, example):
            async with semaphore:
                return await self.run(rule, example)

        tasks = [run_with_limit(semaphore, rules, example) for example in examples]
        results = await asyncio.gather(*tasks)
        return results

    def get_request(self, rules: list[str], example: str) -> dict:
        """
        This method generates the request parameters for a given example and rules.

        Args:
            rules (list[str]): A list of rules that the example needs to adhere to.
            example (str): The example that needs to be evaluated.

        Returns:
            dict: The request parameters as a dictionary.
        """
        request_params = self._get_request_params(rules, example)
        return request_params

    def check_rejection(self, res: str) -> bool:
        """
        This method checks if the response from the chat function is a rejection.

        Args:
            res (str): The response from the chat function.

        Returns:
            bool: True if the response is a rejection, False otherwise.
        """
        return res.strip() == "BAD"

    def get_request_batch(self, rules: list[str], examples: list[str]) -> list[dict]:
        """
        This method generates the request parameters for a batch of examples and rules.

        Args:
            rules (list[str]): A list of rules that the examples need to adhere to.
            examples (list[str]): The examples that need to be evaluated.

        Returns:
            list[dict]: The request parameters for each example as a list of dictionaries.
        """
        requests = [self.get_request(rules, example) for example in examples]
        return requests

    def _get_request_params(self, rules: list[str], example: str) -> dict:
        """
        Private helper function to get all the parameters for the chat function.

        Args:
            rules (list[str]): A list of rules that the example needs to adhere to.
            example (str): The example that needs to be evaluated.

        Returns:
            dict: A dictionary containing all the parameters for the chat function.
        """
        rules_str = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
        messages = [
            {
                "role": "system",
                "content": self.evaluation_prompt.format(rules=rules_str),
            },
            {"role": "user", "content": example},
        ]
        return {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 3,  # We use a max_tokens of 3 because we only need to verify whether the output is "BAD". The actual response content is irrelevant. While gpt-3.5-turbo and gpt-4 tokenizer (cl100k_base) needs exactly max_token=1 for this, other tokenizers like Llama 2 might need more.
        }
