import json


class Scenario:
    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict
        self.correct_diagnosis = scenario_dict["correct_diagnosis"]
        self.doctor_info = scenario_dict["demographics"]
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

    def patient_information(self):
        return self.patient_info

    def doctor_information(self):
        return self.doctor_info

    def diagnosis_information(self):
        return self.correct_diagnosis


class ScenarioLoader:
    def __init__(self, file_path):
        with open(file_path, "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [
            Scenario(scenario_dict) for scenario_dict in self.scenario_strs
        ]
        self.num_scenarios = len(self.scenarios)
