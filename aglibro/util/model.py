from abc import ABC, abstractmethod
from typing import List
import os

from aglibro.util.api_requests import create_chatgpt_config, request_chatgpt_engine

MODELS = {
    "gpt-3.5-turbo-0125": {
        "max_context": 16_385,
        "cost_per_input_token": 5e-07,
        "cost_per_output_token": 1.5e-06,
    },
    "gpt-3.5-turbo-1106": {
        "max_context": 16_385,
        "cost_per_input_token": 1.5e-06,
        "cost_per_output_token": 2e-06,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_context": 16_385,
        "cost_per_input_token": 1.5e-06,
        "cost_per_output_token": 2e-06,
    },
    "gpt-4-32k-0613": {
        "max_context": 32_768,
        "cost_per_input_token": 6e-05,
        "cost_per_output_token": 0.00012,
    },
    "gpt-4-0613": {
        "max_context": 8_192,
        "cost_per_input_token": 3e-05,
        "cost_per_output_token": 6e-05,
    },
    "gpt-4-1106-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4-0125-preview": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_context": 128_000,
        "cost_per_input_token": 1e-05,
        "cost_per_output_token": 3e-05,
    },
    "gpt-4o-2024-05-13": {
        "max_context": 128_000,
        "cost_per_input_token": 5e-06,
        "cost_per_output_token": 15e-06,
    },
    "gpt-4o-2024-08-06": {
        "max_context": 128_000,
        "cost_per_input_token": 2.5e-06,
        "cost_per_output_token": 10e-06,
    },
    "gpt-4o-mini-2024-07-18": {
        "max_context": 128_000,
        "cost_per_input_token": 1.5e-07,
        "cost_per_output_token": 6e-07,
    },
    "claude-3-5-sonnet-20241022": {
        "max_context": 200_000,
        "cost_per_input_token": 3e-06,
        "cost_per_output_token": 15e-06,
    }
}

SHORTCUTS = {
    "gpt3": "gpt-3.5-turbo-1106",
    "gpt3-legacy": "gpt-3.5-turbo-16k-0613",
    "gpt4": "gpt-4-1106-preview",
    "gpt4-legacy": "gpt-4-0613",
    "gpt4-0125": "gpt-4-0125-preview",
    "gpt3-0125": "gpt-3.5-turbo-0125",
    "gpt4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt4o": "gpt-4o-2024-05-13",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
}

def get_model_price(model: str) -> tuple[float, float]:
    if model in SHORTCUTS:
        model = SHORTCUTS[model]
    if model in MODELS:
        return MODELS[model]["cost_per_input_token"], MODELS[model]["cost_per_output_token"]
    else:
        raise ValueError("Model not found: {}".format(model))
    
class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        logger.info("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            if ret.usage:
                completion_tokens = ret.usage.completion_tokens
                prompt_tokens = ret.usage.prompt_tokens
            else:
                completion_tokens = 0
                prompt_tokens = 0
                with open("errorlog.txt", "a") as f:
                    f.write(f"Error in OpenAIChatDecoder.codegen: usage is None:\n{ret}\n")
                print(f"Error in OpenAIChatDecoder.codegen: usage is None:\n{ret}\n")
                self.logger.error(f"Error in OpenAIChatDecoder.codegen: usage is None:\n{ret}\n")
                
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
            
        base_url = "https://api.deepseek.com"
        if 'OPENAI_BASE_URL' in os.environ:
            base_url = os.environ['OPENAI_BASE_URL']

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url=base_url
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class ClaudeChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(self, message: str, num_samples: int = 1) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)
            
        base_url = os.environ.get('OPENAI_BASE_URL')

        trajs = []
        for _ in range(batch_size):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url=base_url
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens if ret.usage else 0,
                            "prompt_tokens": ret.usage.prompt_tokens if ret.usage else 0,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "claude":
        return ClaudeChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError
