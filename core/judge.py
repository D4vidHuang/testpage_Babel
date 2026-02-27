"""OpenRouter/Ollama API wrapper for LLM judge calls."""

import requests
import json
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv


class LLMJudge:
    """Simple wrapper around the OpenRouter chat completions API.
    
    Returns the raw JSON response from the API.
    """

    def __init__(
        self,
        models: list,
        temperature: float,
        system_message: str,
        assignment_description: str,
        provider: str = "openrouter",
        ollama_url: str = "http://127.0.0.1:11434/api/generate",
    ):
        """Initialize the LLMJudge.

        Args:
            models: List of model identifiers (e.g., ["anthropic/claude-3.5-sonnet"])
            temperature: Sampling temperature for the model output
            system_message: Base instruction for the LLM's role
            assignment_description: The specific task for the judge
            provider: 'openrouter' (default) or 'ollama'
            ollama_url: Custom URL for Ollama (only used when provider='ollama')
        """
        # Load environment variables from .env file if it exists
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        self.provider = provider.lower()

        if self.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY not set. Set it in the environment or create a .env file."
                )
            self.api_key = api_key
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == "ollama":
            self.api_key = None
            self.api_url = ollama_url
            self.headers = {"Content-Type": "application/json"}
        else:
            raise ValueError("Unsupported provider: must be 'openrouter' or 'ollama'")

        self.models = models
        self.temperature = temperature
        self.system_message = system_message
        self.assignment_description = assignment_description

    def _construct_user_prompt(self, text_to_evaluate: str) -> str:
        """Combine the assignment description and text into a single prompt."""
        return f"{self.assignment_description}\n\n--- TEXT TO EVALUATE ---\n{text_to_evaluate}"

    def judge(self, text_to_evaluate: str, model_index: int = 0):
        """Send the text to the model for judging and return the raw JSON response.

        Args:
            text_to_evaluate: The text/code to evaluate
            model_index: Index of the model to use from self.models list (default: 0)
        """
        user_prompt = self._construct_user_prompt(text_to_evaluate)
        response = None
        data = None

        def get_quant(model_name):
            if model_name.lower().split(":")[0] in ["qwen/qwen3-vl-235b-a22b-instruct", "deepseek/deepseek-v3.2-speciale", "qwen/qwen3-next-80b-a3b-instruct", "deepseek/deepseek-r1-0528", "deepseek/deepseek-v3.2"]:
                return "fp8"
            if model_name.lower().split(":")[0] in ["qwen/qwen3-coder-next", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
                return "bf16"
            return None

        try:
            if self.provider == "openrouter":
                if get_quant(self.models[model_index]):
                                    payload = {
                    "model": self.models[model_index],
                    "temperature": self.temperature,
                    "seed": 42,
                    "max_tokens": 10000,
                    "provider": {
                        "quantizations": [get_quant(self.models[model_index])],
                        "max_price": {"prompt": 1, "completion": 3.1}
                    },
                    "messages": [
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                else:
                    payload = {
                        "model": self.models[model_index],
                        "temperature": self.temperature,
                        "seed": 42,
                        "max_tokens": 10000,
                        "provider": {
                            "max_price": {"prompt": 1, "completion": 3.1}
                        },
                        "messages": [
                            {"role": "system", "content": self.system_message},
                            {"role": "user", "content": user_prompt},
                        ],
                        
                    }

                max_retries = 10
                base_backoff = 5.0
                max_backoff = 120.0
                retryable_statuses = {429, 503, 524, 500}

                for attempt in range(max_retries + 1):
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            data=json.dumps(payload),
                        )

                        if response.status_code in retryable_statuses:
                            retry_after = response.headers.get("Retry-After")
                            if retry_after and retry_after.isdigit():
                                sleep_seconds = int(retry_after)
                            else:
                                exp = min(max_backoff, base_backoff * (2 ** attempt))
                                sleep_seconds = random.uniform(0, exp)

                            if attempt < max_retries:
                                time.sleep(sleep_seconds)
                                continue

                        if 400 <= response.status_code < 500 and response.status_code not in retryable_statuses:
                            response.raise_for_status()

                        if 500 <= response.status_code < 600 and response.status_code not in retryable_statuses:
                            response.raise_for_status()

                        response.raise_for_status()
                        data = response.json()
                        break

                    except requests.exceptions.Timeout:
                        if attempt >= max_retries:
                            raise
                        exp = min(max_backoff, base_backoff * (2 ** attempt))
                        time.sleep(random.uniform(0, exp))

            elif self.provider == "ollama":
                payload = {
                    "model": self.models[model_index],
                    "system": self.system_message,
                    "prompt": user_prompt,
                    "stream": False
                }
                response = requests.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()

                # Normalize Ollama response to OpenRouter format
                if not isinstance(data, dict) or "choices" not in data:
                    if isinstance(data, dict) and "generated" in data:
                        first = data["generated"][0] if data["generated"] else {}
                        content = first.get("content") or first.get("text") if isinstance(first, dict) else first
                        data = {"choices": [{"message": {"content": content}}]}
                    elif isinstance(data, dict) and ("result" in data or "output" in data):
                        content = data.get("result") or data.get("output")
                        data = {"choices": [{"message": {"content": content}}]}
                    else:
                        text_body = response.text if response else json.dumps(data)
                        data = {"choices": [{"message": {"content": text_body}}]}

            return data

        except requests.exceptions.HTTPError as http_err:
            body = response.text if response is not None else "<no response body>"
            print(f"HTTP error occurred: {http_err} - {body}")
            raise
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
