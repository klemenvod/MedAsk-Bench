from api_client import APIClient


def compare_results(doctor_reply, correct_diagnosis, evaluator_llm):
    if evaluator_llm == "gpt4":
        api_client = APIClient()
        messages = [
            {
                "role": "system",
                "content": "Given a list of differential diagnoses and the correct diagnosis, determine if the correct diagnosis is present in the list. If it is, specify its position. Respond in the following format: \n\nCorrect diagnosis present: YES/NO \nPosition: [number]",
            },
            {
                "role": "user",
                "content": (
                    "\nHere are the differential diagnoses: "
                    + doctor_reply
                    + "\nHere is the correct diagnosis: "
                    + correct_diagnosis
                    + "\nResponse:"
                ),
            },
        ]
        response = api_client.call_openai(
            "gpt-4-turbo-preview",
            messages,
            temperature=0.0,
        )
        lines = response.lower().split("\n")

        # Extract the "YES/NO" value and position number
        correct_diagnosis_present = lines[0].split(":")[1].strip()
        position = lines[1].split(":")[1].strip()

        # Return a tuple with the extracted values
        return (correct_diagnosis_present, position)
    else:
        raise ValueError(f"Unsupported evaluator model: {evaluator_llm}")
