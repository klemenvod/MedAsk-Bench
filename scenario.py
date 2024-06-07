import json
import os


class ScenarioAgentclinic:
    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict
        self.correct_diagnosis = scenario_dict["correct_diagnosis"]
        self.doctor_info = scenario_dict["demographics"]
        self.patient_info = {
            "Demographics": scenario_dict["demographics"],
            "History": scenario_dict.get("history", ""),
            "Primary_Symptom": scenario_dict.get("primary_symptom", ""),
            "Secondary_Symptoms": scenario_dict.get("secondary_symptoms", []),
            "Temperature": scenario_dict.get("temperature", ""),
            "Past_Medical_History": scenario_dict.get("past_medical_history", ""),
            "Social_History": scenario_dict.get("social_history", ""),
            "Review_of_Systems": scenario_dict.get("review_of_systems", ""),
        }

    def patient_information(self):
        return self.patient_info

    def doctor_information(self):
        return self.doctor_info

    def diagnosis_information(self):
        return self.correct_diagnosis


class ScenarioAvey:
    def __init__(self, scenario_dict):
        self.scenario_dict = scenario_dict
        self.correct_diagnosis = scenario_dict["correct_diagnosis"]
        self.doctor_info = scenario_dict["demographics"]
        self.patient_info = {
            "Demographics": scenario_dict["demographics"],
            "Chief_Complaints": scenario_dict["chief_complaints"],
            "Presentation": scenario_dict["presentation"],
            "Absent_Findings": scenario_dict["absent_findings"],
            "Physical_History": scenario_dict["physical_history"],
            "Family_History": scenario_dict["family_history"],
            "Social_History": scenario_dict["social_history"],
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

        file_name = os.path.basename(file_path)
        if file_name == "agentclinic_vignettes.jsonl":
            self.scenarios = [
                ScenarioAgentclinic(scenario_dict)
                for scenario_dict in self.scenario_strs
            ]
        elif file_name == "avey_vignettes.jsonl":
            self.scenarios = [
                ScenarioAvey(scenario_dict) for scenario_dict in self.scenario_strs
            ]
        else:
            raise ValueError(f"Unsupported file name: {file_name}")

        self.num_scenarios = len(self.scenarios)
