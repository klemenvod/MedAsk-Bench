import argparse
from transformers import pipeline
import openai, re, random, time, json, replicate, os

llama_url = "meta/llama-2-70b-chat"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"


def load_huggingface_model(model_name):
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe


def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response


class Scenario:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.correct_diagnosis = scenario_dict["correct_diagnosis"]
        self.examiner_info = scenario_dict["demographics"]
        self.patient_info = {
            "Demographics": scenario_dict["demographics"],
            "History": scenario_dict["history"],
            "Primary_Symptom": scenario_dict["primary_symptom"],
            "Secondary_Symptoms": scenario_dict["secondary_symptoms"],
            "Temperature": scenario_dict["temperature"],
            "Past_Medical_History": scenario_dict["past_medical_history"],
            "Social_History": scenario_dict["social_history"],
            "Review_of_Systems": scenario_dict["review_of_systems"],
        }

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> str:
        return self.examiner_info

    def diagnosis_information(self) -> dict:
        return self.correct_diagnosis


class ScenarioLoader:
    def __init__(self) -> None:
        with open("agentclinic_vignettes.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [
            Scenario(scenario_dict) for scenario_dict in self.scenario_strs
        ]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self) -> Scenario:
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id) -> Scenario:
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

    def inference_patient(self, question) -> str:
        answer = str()
        if self.backend == "gpt4":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the doctor response: "
                    + question
                    + "Now please continue with your next response\nPatient: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]
        elif self.backend == "gpt3.5":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the doctor response: "
                    + question
                    + "Now please continue with your next response\nPatient: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]
        elif self.backend == "gpt4o":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the doctor response: "
                    + question
                    + "Now please continue with your next response\nPatient: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]
        elif self.backend == "mixtral-8x7b":
            prompt = (
                "\nHere is a history of your dialogue: "
                + self.agent_hist
                + "\n Here was the doctor response: "
                + question
                + "Now please continue with your next response\nPatient: "
            )
            output = replicate.run(
                mixtral_url,
                input={
                    "prompt": prompt,
                    "system_prompt": self.system_prompt(),
                    "max_new_tokens": 75,
                },
            )
            answer = "".join(output)
        elif "HF_" in self.backend:
            input_text = (
                self.system_prompt()
                + "\nHere is a history of your dialogue: "
                + self.agent_hist
                + "\n Here was the doctor response: "
                + question
                + "Now please continue with your next response\nPatient: "
            )
            if self.pipe is None:
                self.pipe = load_huggingface_model(self.backend.replace("HF_", ""))
            answer = inference_huggingface(input_text, self.pipe)
        else:
            raise Exception("No model by the name {}".format(self.backend))

        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are a patient with the following background:\n{}\n\nYou have the following additional details:\n{}\n\nA doctor will ask you questions to diagnose your condition. Provide concise answers of 1-3 sentences, sharing only the relevant information from the additional details if asked. If the doctor asks about something not mentioned in the additional details, simply say 'I don't know.'"

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

        return base.format(scenario, background)

    def first_prompt(self) -> str:
        base = "You are a patient with the following background:\n{}\n\nYou have gone to the doctor to get a diagnosis for your condition. Start the conversation by presenting your primary symptom as your initial complaint:\n{}"

        scenario = (
            "The patient is a "
            + self.symptoms["Demographics"]
            + ". "
            + self.symptoms["History"]
        )
        initial_complaint = self.symptoms["Primary_Symptom"]

        return base.format(scenario, initial_complaint)

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_info

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20) -> None:
        # number of inference calls to the patient
        self.infs = 0
        # maximum number of inference calls to the patient
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for patient
        self.presentation = ""
        # language model backend for patient agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None

    def inference_doctor(self, question) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
        if self.backend == "gpt4":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the patient response: "
                    + question
                    + "Now please continue with your next response\nDoctor: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]

        elif self.backend == "gpt3.5":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the patient response: "
                    + question
                    + "Now please continue with your next response\nDoctor: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]

        elif self.backend == "gpt4o":
            messages = [
                {"role": "system", "content": self.system_prompt()},
                {
                    "role": "user",
                    "content": "\nHere is a history of your dialogue: "
                    + self.agent_hist
                    + "\n Here was the patient response: "
                    + question
                    + "Now please continue with your next response\nDoctor: ",
                },
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.05,
            )
            answer = response["choices"][0]["message"]["content"]

        elif self.backend == "llama-2-70b-chat":
            prompt = (
                "\nHere is a history of your dialogue: "
                + self.agent_hist
                + "\n Here was the patient response: "
                + question
                + "Now please continue with your next response\nDoctor: "
            )
            prompt = prompt[:]  # token limit
            output = replicate.run(
                llama_url,
                input={
                    "prompt": prompt,
                    "system_prompt": self.system_prompt(),
                    "max_new_tokens": 150,
                },
            )
            answer = "".join(output)

        elif self.backend == "mixtral-8x7b":
            prompt = (
                "\nHere is a history of your dialogue: "
                + self.agent_hist
                + "\n Here was the patient response: "
                + question
                + "Now please continue with your next response\nDoctor: "
            )
            output = replicate.run(
                mixtral_url,
                input={
                    "prompt": prompt,
                    "system_prompt": self.system_prompt(),
                    "max_new_tokens": 75,
                },
            )
            answer = "".join(output)
        elif "HF_" in self.backend:
            input_text = (
                self.system_prompt()
                + "\nHere is a history of your dialogue: "
                + self.agent_hist
                + "\n Here was the patient response: "
                + question
                + "Now please continue with your next response\nDoctor: "
            )
            if self.pipe is None:
                self.pipe = load_huggingface_model(self.backend.replace("HF_", ""))
            answer = inference_huggingface(input_text, self.pipe)
        else:
            raise Exception("No model by the name {}".format(self.backend))

        self.infs += 1
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are a doctor named Dr. Babi, diagnosing a {} patient. You will ask them concise questions (1-3 sentences each) in order to understand their disease. After gathering sufficient information, type the and tag and list 5 most likely diagnoses in this format: DIAGNOSIS READY: [diagnosis1, diagnosis2, diasnosis3, diagnosis4, diagnosis5]"
        return base.format(self.presentation)

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_info


def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    if moderator_llm == "gpt4":
        messages = [
            {
                "role": "system",
                "content": "You are responsible for determining if the correct diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.",
            },
            {
                "role": "user",
                "content": "\nHere is the correct diagnosis: "
                + correct_diagnosis
                + "\n Here was the doctor diagnosis: "
                + diagnosis
                + "\nAre these the same?",
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.0,
        )
        answer = response["choices"][0]["message"]["content"]
    elif "HF_" in moderator_llm:
        input_text = (
            "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."
            + "\nHere is the correct diagnosis: "
            + correct_diagnosis
            + "\n Here was the doctor diagnosis: "
            + diagnosis
            + "\nAre these the same?"
        )
        answer = inference_huggingface(input_text, mod_pipe)
    else:
        raise Exception("No model by the name {}".format(moderator_llm))

    return answer.lower()


def main(
    api_key,
    replicate_api_key,
    inf_type,
    doctor_llm,
    patient_llm,
    moderator_llm,
    num_scenarios,
):
    openai.api_key = api_key
    if patient_llm == "mixtral-8x7b" or doctor_llm in [
        "llama-2-70b-chat",
        "mixtral-8x7b",
    ]:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key

    scenario_loader = ScenarioLoader()
    total_correct = 0
    total_presents = 0

    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None

    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        pi_dialogue = str()
        scenario = scenario_loader.get_scenario(id=_scenario_id)
        patient_agent = PatientAgent(scenario=scenario, backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario, backend_str=doctor_llm, max_infs=20
        )

        for _inf_id in range(20):
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue)
            print(
                "Doctor [{}%]:".format(int(((_inf_id + 1) / 20) * 100)), doctor_dialogue
            )

            if "DIAGNOSIS READY" in doctor_dialogue:
                correctness = (
                    compare_results(
                        doctor_dialogue,
                        scenario.diagnosis_information(),
                        moderator_llm,
                        pipe,
                    )
                    == "yes"
                )
                if correctness:
                    total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print(
                    "Scene {}, The diagnosis was ".format(_scenario_id),
                    "CORRECT" if correctness else "INCORRECT",
                    int((total_correct / total_presents) * 100),
                )
                break

            if inf_type == "human_patient":
                pi_dialogue = input("\nResponse to doctor: ")
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            print("Patient [{}%]:".format(int(((_inf_id + 1) / 20) * 100)), pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Diagnosis Simulation CLI")
    parser.add_argument(
        "--openai_api_key", type=str, required=False, help="OpenAI API Key"
    )
    parser.add_argument(
        "--replicate_api_key", type=str, required=False, help="Replicate API Key"
    )
    parser.add_argument(
        "--inf_type",
        type=str,
        choices=["llm", "human_doctor", "human_patient"],
        default="llm",
    )
    parser.add_argument("--doctor_llm", type=str, default="gpt4")
    parser.add_argument("--patient_llm", type=str, default="gpt4")
    parser.add_argument("--moderator_llm", type=str, default="gpt4")
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1,
        required=False,
        help="Number of scenarios to simulate",
    )
    args = parser.parse_args()

    main(
        args.openai_api_key,
        args.replicate_api_key,
        args.inf_type,
        args.doctor_llm,
        args.patient_llm,
        args.moderator_llm,
        args.num_scenarios,
    )
