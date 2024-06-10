import argparse
from config import MAX_INFERENCES, AGENTCLINIC_VIGNETTES, AVEY_VIGNETTES
from scenario import ScenarioLoader
from patient import PatientAgent
from doctor import DoctorAgent
from evaluator import compare_results


def main(doctor_llm, patient_llm, evaluator_llm, num_scenarios, file_path):
    scenario_loader = ScenarioLoader(file_path)
    total_correct = 0
    total_presents = 0

    for scenario in scenario_loader.scenarios[:num_scenarios]:
        total_presents += 1
        patient_agent = PatientAgent(scenario, patient_llm)
        doctor_agent = DoctorAgent(scenario, doctor_llm, MAX_INFERENCES)

        doctor_dialogue = ""

        patient_reply = patient_agent.start_conversation()
        print("Patient:", patient_reply)
        doctor_dialogue = f"Patient: {patient_reply}\n"

        while True:
            doctor_reply = doctor_agent.inference_doctor(doctor_dialogue)
            print("Doctor:", doctor_reply)
            doctor_dialogue += f"Doctor: {doctor_reply}\n"

            if "DIAGNOSIS READY" in doctor_reply:
                correct_diagnosis_present, position = (
                    compare_results(
                        doctor_reply,
                        scenario.diagnosis_information(),
                        evaluator_llm,
                    )
                )

                if correct_diagnosis_present == "yes":
                    total_correct += 1

                print("\nCorrect answer:", scenario.diagnosis_information())
                print(
                    f"Scene {total_presents}, The diagnosis was",
                    "CORRECT" if correct_diagnosis_present == "yes" else "INCORRECT",
                    f"in position {position}.",
                    int((total_correct / total_presents) * 100),
                )
                break

            if doctor_agent.infs >= doctor_agent.MAX_INFS:
                print("Maximum inferences reached. Diagnosis not completed.")
                break

            patient_reply = patient_agent.inference_patient(doctor_dialogue)
            print("Patient:", patient_reply)
            doctor_dialogue += f"Patient: {patient_reply}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symptom Assessment Simulation CLI")
    parser.add_argument("--doctor_llm", type=str, default="mixtral-8x7b")
    parser.add_argument("--patient_llm", type=str, default="gpt4")
    parser.add_argument("--evaluator_llm", type=str, default="gpt4")
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=1,
        required=False,
        help="Number of scenarios to simulate",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        choices=["avey", "agentclinic"],
        required=True,
        help="Specify 'avey' for AVEY_VIGNETTES or 'agentclinic' for AGENTCLINIC_VIGNETTES",
    )
    args = parser.parse_args()

    # Map the user input to the corresponding file path
    if args.file_path == "avey":
        file_path = AVEY_VIGNETTES
    elif args.file_path == "agentclinic":
        file_path = AGENTCLINIC_VIGNETTES
    else:
        raise ValueError("Invalid file path specified")

    main(
        args.doctor_llm,
        args.patient_llm,
        args.evaluator_llm,
        args.num_scenarios,
        file_path,
    )
