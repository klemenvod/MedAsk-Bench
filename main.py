import argparse
import json
import os
from tqdm import tqdm
from config import MAX_INFERENCES, AGENTCLINIC_VIGNETTES, AVEY_VIGNETTES
from scenario import ScenarioLoader
from patient import PatientAgent
from doctor import DoctorAgent
from evaluator import compare_results
from util import calculate_word_counts


def main(
    doctor_llm, patient_llm, evaluator_llm, file_path, run_experiment, num_experiments
):
    scenario_loader = ScenarioLoader(file_path)
    scenarios = (
        scenario_loader.scenarios if run_experiment else [scenario_loader.scenarios[0]]
    )

    if run_experiment:
        os.makedirs("experiments", exist_ok=True)

    for experiment_idx in range(1, num_experiments + 1):
        if run_experiment:
            jsonl_file = open(
                f"experiments/{args.doctor_llm}_experiment_{experiment_idx}.jsonl",
                "w",
            )

        for idx, scenario in tqdm(
            enumerate(scenarios, start=1),
            total=len(scenarios),
            desc=f"Experiment {experiment_idx}",
        ):
            patient_agent = PatientAgent(scenario, patient_llm)
            doctor_agent = DoctorAgent(scenario, doctor_llm, MAX_INFERENCES)

            doctor_dialogue = ""

            patient_reply = patient_agent.start_conversation()
            if not run_experiment:
                print("Patient:", patient_reply)

            doctor_dialogue = f"Patient: {patient_reply}\n"

            while True:
                doctor_reply = doctor_agent.inference_doctor(doctor_dialogue)
                if not run_experiment:
                    print("Doctor:", doctor_reply)
                doctor_dialogue += f"Doctor: {doctor_reply}\n"

                if "DIAGNOSIS READY" in doctor_reply:
                    correct_diagnosis_present, position = compare_results(
                        doctor_reply,
                        scenario.diagnosis_information(),
                        evaluator_llm,
                    )

                    if not run_experiment:
                        print("\nCorrect answer:", scenario.diagnosis_information())
                        print(
                            (
                                f"The diagnosis was CORRECT"
                                if correct_diagnosis_present == "yes"
                                else "INCORRECT"
                            ),
                            f"in position {position}.",
                        )
                    else:
                        words_total, words_per_turn_doc, words_per_turn_patient = (
                            calculate_word_counts(doctor_dialogue)
                        )
                        data = {
                            "correct_dx": scenario.diagnosis_information(),
                            "vignette_source": args.file_path,
                            "diff_dx": doctor_reply,
                            "is_correct_dx_present": correct_diagnosis_present,
                            "dx_position": position,
                            "num_turns": doctor_agent.infs,
                            "words_total": words_total,
                            "words_per_turn_doc": words_per_turn_doc,
                            "words_per_turn_patient": words_per_turn_patient,
                            "whole_dialogue": doctor_dialogue,
                        }
                        jsonl_file.write(json.dumps(data) + "\n")
                    break

                if doctor_agent.infs >= doctor_agent.MAX_INFS:
                    if not run_experiment:
                        print("Maximum inferences reached. Diagnosis not completed.")

                    else:
                        words_total, words_per_turn_doc, words_per_turn_patient = (
                            calculate_word_counts(doctor_dialogue)
                        )
                        data = {
                            "correct_dx": scenario.diagnosis_information(),
                            "vignette_source": args.file_path,
                            "diff_dx": doctor_reply,
                            "is_correct_dx_present": "Maximum inferences reached",
                            "dx_position": "",
                            "num_turns": doctor_agent.infs,
                            "words_total": words_total,
                            "words_per_turn_doc": words_per_turn_doc,
                            "words_per_turn_patient": words_per_turn_patient,
                            "whole_dialogue": doctor_dialogue,
                        }
                        jsonl_file.write(json.dumps(data) + "\n")
                    break

                patient_reply = patient_agent.inference_patient(doctor_dialogue)
                if not run_experiment:
                    print("Patient:", patient_reply)
                doctor_dialogue += f"Patient: {patient_reply}\n"
                tqdm.write(
                    f"Experiment {experiment_idx} - Scenario {idx}: Dialogue turns: {doctor_agent.infs}/{doctor_agent.MAX_INFS}"
                )

        if run_experiment:
            jsonl_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Symptom Assessment Simulation")
    parser.add_argument("--doctor_llm", type=str, default="gemini")
    parser.add_argument("--patient_llm", type=str, default="gpt4")
    parser.add_argument("--evaluator_llm", type=str, default="gpt4")
    parser.add_argument(
        "--file_path",
        type=str,
        choices=["avey", "agentclinic"],
        required=True,
        help="Specify 'avey' for AVEY_VIGNETTES or 'agentclinic' for AGENTCLINIC_VIGNETTES",
    )
    parser.add_argument(
        "--run_experiment",
        action="store_true",
        help="Run the benchmark on all scenarios and save results to a CSV file",
    )
    parser.add_argument(
        "--num_experiments",
        type=int,
        default=1,
        help="Number of iterations through the vignettes (default: 1)",
    )
    args = parser.parse_args()

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
        file_path,
        args.run_experiment,
        args.num_experiments,
    )
