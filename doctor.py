from api_client import APIClient
from models import MODELS
from config import MAX_INFERENCES


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=MAX_INFERENCES):
        self.infs = 0
        self.MAX_INFS = max_infs
        self.presentation = ""
        self.backend = backend_str
        self.scenario = scenario
        self.api_client = APIClient()
        self.reset()

    def inference_doctor(self, doctor_dialogue):
        model_config = MODELS[self.backend]
        prompt = f"\nHere is the dialogue history:\n{doctor_dialogue}\nProvide the doctor's response."

        if self.backend in ["gpt4", "gpt3.5", "gpt4o"]:
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt},
            ]
            response = self.api_client.call_openai(
                model_config["url"],
                messages,
                model_config["temperature"],
            )
        elif self.backend in ["llama-3-70b-instruct", "mixtral-8x7b"]:
            response = self.api_client.call_replicate(
                model_config["url"],
                prompt,
                self.system_prompt(),
                model_config["max_new_tokens"],
            )
        else:
            raise ValueError(f"Unsupported model: {self.backend}")

        self.infs += 1
        return response

    def system_prompt(self):
        base = "You are a doctor named Dr. Babi, diagnosing a {} patient through an online chat platform. You will ask them concise questions (1-3 sentences each) in order to understand their disease. After gathering sufficient information, type the end tag and list the 5 most likely diagnoses in this format: DIAGNOSIS READY: [diagnosis1, diagnosis2, diagnosis3, diagnosis4, diagnosis5]"
        return base.format(self.presentation)

    def reset(self):
        self.presentation = self.scenario.doctor_info
