import argparse
from config import MAX_INFERENCES
from vignette import VignetteLoader
from patient import PatientAgent
from doctor import DoctorAgent
from evaluator import compare_results


def main(doctor_llm, patient_llm, evaluator_llm, num_vignettes):
    vignette_loader = VignetteLoader()
    total_correct = 0
    total_presents = 0

    for vignette in vignette_loader.vignettes[:num_vignettes]:
        total_presents += 1
        patient_agent = PatientAgent(vignette, patient_llm)
        doctor_agent = DoctorAgent(vignette, doctor_llm, MAX_INFERENCES)

        doctor_dialogue = ""

        patient_reply = patient_agent.start_conversation()
        print("Patient:", patient_reply)
        doctor_dialogue = f"Patient: {patient_reply}\n"

        while True:
            doctor_reply = doctor_agent.inference_doctor(doctor_dialogue)
            print("Doctor:", doctor_reply)
            doctor_dialogue += f"Doctor: {doctor_reply}\n"

            if "DIAGNOSIS READY" in doctor_reply:
                correctness = (
                    compare_results(
                        doctor_reply,
                        vignette.diagnosis_information(),
                        evaluator_llm,
                    )
                    == "yes"
                )

                if correctness:
                    total_correct += 1

                print("\nCorrect answer:", vignette.diagnosis_information())
                print(
                    f"Scene {total_presents}, The diagnosis was",
                    "CORRECT" if correctness else "INCORRECT",
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
    parser = argparse.ArgumentParser(description="Medical Diagnosis Simulation CLI")
    parser.add_argument("--doctor_llm", type=str, default="gpt4")
    parser.add_argument("--patient_llm", type=str, default="gpt4")
    parser.add_argument("--evaluator_llm", type=str, default="gpt4")
    parser.add_argument(
        "--num_vignettes",
        type=int,
        default=1,
        required=False,
        help="Number of vignettes to simulate",
    )
    args = parser.parse_args()

    main(args.doctor_llm, args.patient_llm, args.evaluator_llm, args.num_vignettes)
