def calculate_word_counts(doctor_dialogue):
    if doctor_dialogue.strip():
        total_words = len(doctor_dialogue.split())
        doctor_turns = doctor_dialogue.split("Doctor: ")[1:]
        patient_turns = doctor_dialogue.split("Patient: ")[1:]
        avg_words_per_turn_doc = (
            round(
                sum(len(turn.split()) for turn in doctor_turns) / len(doctor_turns), 1
            )
            if doctor_turns
            else 0
        )
        avg_words_per_turn_patient = (
            round(
                sum(len(turn.split()) for turn in patient_turns) / len(patient_turns), 1
            )
            if patient_turns
            else 0
        )
    else:
        total_words = 0
        avg_words_per_turn_doc = 0
        avg_words_per_turn_patient = 0

    return total_words, avg_words_per_turn_doc, avg_words_per_turn_patient
