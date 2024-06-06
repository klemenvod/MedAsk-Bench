import openai
import replicate
from config import OPENAI_API_KEY, REPLICATE_API_KEY


class APIClient:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        replicate.api_key = REPLICATE_API_KEY

    def call_openai(self, model, messages, temperature):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]

    def call_replicate(self, model_url, prompt, system_prompt, max_new_tokens):
        output = replicate.run(
            model_url,
            input={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_new_tokens,
            },
        )
        return "".join(output)
