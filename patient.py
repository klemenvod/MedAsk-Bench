from api_client import APIClient
from models import MODELS
from scenario import ScenarioAgentclinic, ScenarioAvey


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4"):
        self.scenario = scenario
        self.backend = backend_str
        self.api_client = APIClient()
        self.reset()

    def inference_patient(self, doctor_dialogue):
        model_config = MODELS[self.backend]
        prompt = f"\nHere is the dialogue history:\n{doctor_dialogue}\nProvide the patient's response."

        if self.backend in ["gpt4", "gpt3.5", "gpt4o"]:
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt},
            ]
            return self.api_client.call_openai(
                model_config["url"],
                messages,
                model_config["temperature"],
            )
        elif self.backend in ["llama-3-70b-instruct", "mixtral-8x7b"]:
            return self.api_client.call_replicate(
                model_config["url"],
                prompt,
                self.system_prompt(),
                model_config["max_new_tokens"],
            )
        else:
            raise ValueError(f"Unsupported model: {self.backend}")

    def start_conversation(self):
        system_prompt = self.first_prompt()
        messages = [{"role": "system", "content": system_prompt}]

        if self.backend in ["gpt4", "gpt3.5", "gpt4o"]:
            return self.api_client.call_openai(
                MODELS[self.backend]["url"],
                messages,
                MODELS[self.backend]["temperature"],
            )
        elif self.backend in ["llama-3-70b-instruct", "mixtral-8x7b"]:
            prompt = system_prompt
            return self.api_client.call_replicate(
                MODELS[self.backend]["url"],
                prompt,
                max_new_tokens=MODELS[self.backend]["max_new_tokens"],
            )
        else:
            raise ValueError(f"Unsupported model: {self.backend}")

    def system_prompt(self):
        base = "You are a patient with the following background:\n{}\n\nYou have the following additional details:\n{}\n\nA doctor will ask you questions to diagnose your condition. Provide concise answers of 1-3 sentences, sharing only the relevant information from the additional details if asked. If the doctor asks about something not mentioned in the additional details, simply say 'I don't know.'"

        if isinstance(self.scenario, ScenarioAgentclinic):
            scenario = (
                "The patient is a "
                + self.symptoms["Demographics"]
                + ". "
                + self.symptoms["History"]
            )
            background = (
                "Secondary symptoms: "
                + ", ".join(self.symptoms["Secondary_Symptoms"])
                + "\nPast medical history: "
                + self.symptoms["Past_Medical_History"]
                + "\nSocial history: "
                + self.symptoms["Social_History"]
                + "\nReview of systems: "
                + self.symptoms["Review_of_Systems"]
                + "\nTemperature: "
                + self.symptoms["Temperature"]
            )
        elif isinstance(self.scenario, ScenarioAvey):
            scenario = (
                "The patient is a "
                + self.symptoms["Demographics"]
                + ". "
                + self.symptoms["Presentation"]
            )
            background = (
                "Absent findings: "
                + self.symptoms["Absent_Findings"]
                + "\nPhysical history: "
                + self.symptoms["Physical_History"]
                + "\nFamily history: "
                + self.symptoms["Family_History"]
                + "\nSocial history: "
                + self.symptoms["Social_History"]
            )
        else:
            raise ValueError(f"Unsupported scenario type: {type(self.scenario)}")

        return base.format(scenario, background)

    def first_prompt(self):
        base = "You are a patient with the following background:\n{}\n\nYou have gone to the doctor to get a diagnosis for your condition. Start the conversation by presenting ONLY your primary symptom as your initial complaint:\n{}Provide a concise description of your primary symptom to the doctor."

        if isinstance(self.scenario, ScenarioAgentclinic):
            scenario = (
                "The patient is a "
                + self.symptoms["Demographics"]
                + ". "
                + self.symptoms["History"]
            )
            initial_complaint = self.symptoms["Primary_Symptom"]
        elif isinstance(self.scenario, ScenarioAvey):
            scenario = (
                "The patient is a "
                + self.symptoms["Demographics"]
                + ". "
                + self.symptoms["Presentation"]
            )
            initial_complaint = self.symptoms["Chief_Complaints"]
        else:
            raise ValueError(f"Unsupported scenario type: {type(self.scenario)}")

        return base.format(scenario, initial_complaint)

    def reset(self):
        self.symptoms = self.scenario.patient_info
