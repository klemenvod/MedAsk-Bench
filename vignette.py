import json
import random
from config import AGENTCLINIC_VIGNETTES


class Vignette:
    def __init__(self, vignette_dict):
        self.vignette_dict = vignette_dict
        self.correct_diagnosis = vignette_dict["correct_diagnosis"]
        self.doctor_info = vignette_dict["demographics"]
        self.patient_info = {
            "Demographics": vignette_dict["demographics"],
            "History": vignette_dict["history"],
            "Primary_Symptom": vignette_dict["primary_symptom"],
            "Secondary_Symptoms": vignette_dict["secondary_symptoms"],
            "Temperature": vignette_dict["temperature"],
            "Past_Medical_History": vignette_dict["past_medical_history"],
            "Social_History": vignette_dict["social_history"],
            "Review_of_Systems": vignette_dict["review_of_systems"],
        }

    def patient_information(self):
        return self.patient_info

    def doctor_information(self):
        return self.doctor_info

    def diagnosis_information(self):
        return self.correct_diagnosis


class VignetteLoader:
    def __init__(self):
        with open(AGENTCLINIC_VIGNETTES, "r") as f:
            self.vignette_strs = [json.loads(line) for line in f]
        self.vignettes = [
            Vignette(vignette_dict) for vignette_dict in self.vignette_strs
        ]
        self.num_vignettes = len(self.vignettes)

    def sample_vignette(self):
        return self.vignettes[random.randint(0, len(self.vignettes) - 1)]

    def get_vignette(self, id):
        if id is None:
            return self.sample_vignette()
        return self.vignettes[id]
