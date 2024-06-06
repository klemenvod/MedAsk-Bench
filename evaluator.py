from api_client import APIClient


def compare_results(doctor_reply, correct_diagnosis, evaluator_llm):
    if evaluator_llm == "gpt4":
        api_client = APIClient()
        messages = [
            {
                "role": "system",
                "content": "Is the correct diagnosis found in the list of differential diagnoses? Respond only with Yes or No. Nothing else.",
            },
            {
                "role": "user",
                "content": (
                    "\nHere is the correct diagnosis: "
                    + correct_diagnosis
                    + "\nHere was the doctor diagnosis: "
                    + doctor_reply
                    + "\nAre these the same?"
                ),
            },
        ]
        response = api_client.call_openai(
            "gpt-4-turbo-preview",
            messages,
            temperature=0.0,
        )
        return response.lower()
    else:
        raise ValueError(f"Unsupported evaluator model: {evaluator_llm}")
