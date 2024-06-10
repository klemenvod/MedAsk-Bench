import openai
import replicate
import anthropic
from config import OPENAI_API_KEY, REPLICATE_API_KEY, ANTHROPIC_API_KEY


class APIClient:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        replicate.api_key = REPLICATE_API_KEY
        self.anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)

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

    def call_anthropic(self, model, system_prompt, prompt, max_tokens, temperature):
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
