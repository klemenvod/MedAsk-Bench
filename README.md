# MedAsk-Bench, a Symptom Assessment Simulation

This Python script simulates a symptom assessment conversation between a patient and a doctor using different language models (LLMs). The script allows you to run experiments on a set of predefined scenarios and store the results in a JSONL file for further analysis.

## Features

- Simulates a symptom assessment conversation between a patient and a doctor
- Supports different LLMs for the patient, doctor, and evaluator roles
- Loads scenarios from a specified file path (AVEY_VIGNETTES or AGENTCLINIC_VIGNETTES)
- Stores the experiment results in a JSONL file, including:
  - Correct diagnosis
  - Vignette source
  - Doctor's differential diagnosis
  - Correctness of the diagnosis
  - Position of the correct diagnosis
  - Number of conversation turns
  - Total word count in the dialogue
  - Average words per turn for the doctor and patient
  - Entire dialogue history

## Setup

1. Make sure you have Python 3.x installed on your system.

2. Install pipenv by running the following command: `pip install pipenv`

3. Clone the repository and navigate to the project directory.

4. Create a new virtual environment and install the required dependencies using pipenv: `pipenv install`

## Usage

To run the script using pipenv, use the following command: `python main.py --doctor_llm <doctor_llm> --file_path <avey/agentclinic> [--run_experiment] [--num_experiments <num_experiments>]`

- `--doctor_llm`: The LLM to use for the doctor role (default: "gpt3.5")
- `--patient_llm`: The LLM to use for the patient role (default: "gpt4")
- `--evaluator_llm`: The LLM to use for the evaluator role (default: "gpt4")
- `--file_path`: The file path to load scenarios from ("avey" for AVEY_VIGNETTES or "agentclinic" for AGENTCLINIC_VIGNETTES)
- `--run_experiment`: Flag to run the benchmark on all scenarios and save results to a JSONL file
- `--num_experiments`: Number of iterations through the vignettes (default: 1)

## Examples

1. This command will run the symptom assessment simulation using GPT-3.5 for the doctor, load scenarios from the AGENTCLINIC_VIGNETTES file, and run the simulation on only one scenario, printing the simulated dialogue between doctor and patient.
`run python main.py --doctor_llm gemini --file_path agentclinic`
 
2. This command will run the symptom assessment simulation using GPT-3.5 for the doctor, load scenarios from the AVEY_VIGNETTES file, run the experiment on all scenarios, and perform 3 iterations through the vignettes. The results will be saved in a JSONL file.
`run python main.py --doctor_llm gpt3.5 --file_path avey --run_experiment --num_experiments 3`

## Dependencies

The required dependencies are listed in the `Pipfile`. They will be automatically installed when you run `pipenv install`.

## License

This project is licensed under the [MIT License](LICENSE).
