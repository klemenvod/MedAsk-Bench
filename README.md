# MedAsk-Bench, A Symptom Assessment Simulation

This repository contains a simulation framework for assessing medical symptoms using large language models (LLMs). The simulation involves interactions between virtual patients and doctors, with patient scenarios drawn from predefined vignettes. The objective is to evaluate the performance of different LLMs in diagnosing medical conditions based on these interactions.

## Features

- Simulates a symptom assessment conversation between a patient and a doctor and evaluates the diagnostic accuracy of doctor LLM
- Supports different LLMs for the doctor role (GPT3.5, GPT4, GPT4o, gemini-1.5-pro, claude-opus, llama-3-70b-instruct and mixtral-8x7b)
- For patient and evaluator roles, it currently supports GPT4
- Loads scenarios from AVEY_VIGNETTES (400 patient scenarios) or AGENTCLINIC_VIGNETTES (107 patient scenarios)
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
 
## Directory Structure

- `experiments/`: Contains experimental results.
- `vignettes/`: Stores vignette files used in the simulations.
  - `agentclinic_vignettes.jsonl`: Vignettes for the agent clinic scenario.
  - `avey_vignettes.jsonl`: Vignettes for the AVEY scenario.
- `api_client.py`: Contains code for LLM API interactions.
- `config.py`: Holds configuration constants.
- `doctor.py`: Defines the `DoctorAgent` class and its behavior.
- `evaluator.py`: Contains evaluation logic to compare diagnostic results.
- `main.py`: The main entry point of the simulation.
- `models.py`: Defines LLM models used and tested.
- `patient.py`: Defines the `PatientAgent` class and its behavior.
- `scenario.py`: Handles loading and managing scenarios from vignettes.
- `util.py`: Contains utility functions used across the project.

## Setup

1. Make sure you have Python 3.11 installed on your system.

2. Install pipenv by running the following command: `pip install pipenv`

3. Clone the repository and navigate to the project directory.

4. Create a new virtual environment and install the required dependencies using pipenv: `pipenv install`

5. Create a `config.py` file in your root. Copy the variables from `config.template`

6. Add your API tokens to `config.py`. If you want to run models on replicate (mixtral or llama), use the command `export REPLICATE_API_TOKEN=your_api_token`

## Usage

To run the script using pipenv, use the following command: `python main.py --doctor_llm <doctor_llm> --file_path <avey/agentclinic> [--run_experiment] [--num_experiments <num_experiments>]`

- `--doctor_llm`: The LLM to use for the doctor role (default: "gpt3.5")
- `--patient_llm`: The LLM to use for the patient role (default: "gpt4")
- `--evaluator_llm`: The LLM to use for the evaluator role (default: "gpt4")
- `--file_path`: The file path to load scenarios from ("avey" for AVEY_VIGNETTES or "agentclinic" for AGENTCLINIC_VIGNETTES) *REQUIRED
- `--run_experiment`: Flag to run the benchmark on all scenarios and save results to a JSONL file
- `--num_experiments`: Number of iterations through the vignettes (default: 1)

## Examples

1. This command will run the symptom assessment simulation using gemini for the doctor, load scenarios from the AGENTCLINIC_VIGNETTES file, and run a single simulation, printing the simulated dialogue between doctor and patient.
`run python main.py --doctor_llm gemini --file_path agentclinic`
 
2. This command will run the symptom assessment simulation using GPT-3.5 for the doctor, load scenarios from the AVEY_VIGNETTES file, run the experiment on all scenarios, and perform 3 iterations through the vignettes. The results will be saved in a JSONL file.
`run python main.py --doctor_llm gpt3.5 --file_path avey --run_experiment --num_experiments 3`

## Code Overview

### `main.py`

The entry point of the application. It parses command-line arguments and initiates the simulation process. The core simulation loop is defined in the `main` function, which interacts with virtual patients and doctors and logs the results.

### `util.py`

Contains utility functions that handle the simulation logic.

### `scenario.py`

Defines the `ScenarioLoader` class, responsible for loading and managing scenarios from the vignette files.

### `patient.py`

Defines the `PatientAgent` class, which simulates patient interactions based on scenarios from clinical vignettes and a specified LLM.

### `doctor.py`

Defines the `DoctorAgent` class, which simulates doctor interactions, generates inferences, and attempts diagnoses.

### `evaluator.py`

Contains the `compare_results` function, which evaluates the accuracy of diagnoses made by the doctor agent.

### `config.py`

Holds configuration constants.

### `models.py`

Defines the configuration for various LLMs used in the simulation. This includes model URLs, token limits, and temperature settings.

### `api_client.py`

Implements the `APIClient` class, which provides methods to interact with different LLM APIs, including OpenAI, Replicate, Anthropic, and Google. The class manages API keys and handles the details of making API calls to these services.


## Dependencies

The required dependencies are listed in the `Pipfile`. They will be automatically installed when you run `pipenv install`.

## License

This project is licensed under the [MIT License](LICENSE).
