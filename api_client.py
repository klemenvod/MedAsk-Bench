import openai
import replicate
import anthropic
import google.generativeai as genai
from config import OPENAI_API_KEY, REPLICATE_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY


class APIClient:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        replicate.api_key = REPLICATE_API_KEY
        self.anthropic_client = anthropic.Client(api_key=ANTHROPIC_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)

    def call_openai(self, model, prompt, temperature):
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]

    def call_replicate(self, model, prompt, system_prompt, max_tokens):
        output = replicate.run(
            model,
            input={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": max_tokens,
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

    def call_google(self, model, prompt, max_tokens, temperature):
        model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        response = model.generate_content(prompt)
        return response.text
