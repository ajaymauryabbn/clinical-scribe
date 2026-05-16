import os
import json
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.graph import scribe_agent

load_dotenv()

def test_agent_with_synthetic_data():
    # Load a synthetic transcript
    data_path = "data/synthetic/hypertension_follow-up.json"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, "r") as f:
        synthetic_data = json.load(f)

    # Prepare input transcript format
    # The synthetic data has 'raw_transcript_text', but the agent expects a list of turns.
    # We'll do a simple split for testing.
    raw_text = synthetic_data["raw_transcript_text"]
    transcript = []
    for line in raw_text.split("\n"):
        if line.startswith("DOCTOR:"):
            transcript.append({"speaker": "DOCTOR", "text": line.replace("DOCTOR:", "").strip()})
        elif line.startswith("PATIENT:"):
            transcript.append({"speaker": "PATIENT", "text": line.replace("PATIENT:", "").strip()})

    print(f"Running agent for: {synthetic_data['scenario']}")
    
    # Run the agent
    initial_state = {
        "transcript": transcript,
        "doctor_turns": "",
        "patient_turns": "",
        "entities": {},
        "icd_codes": [],
        "drug_corrections": [],
        "soap_note": {},
        "flags": []
    }
    
    result = scribe_agent.invoke(initial_state)
    
    print("\n--- AGENT RESULT ---")
    print(json.dumps(result["soap_note"], indent=2))
    print("\n--- ICD CODES ---")
    print(json.dumps(result["icd_codes"], indent=2))
    print("\n--- DRUG CORRECTIONS ---")
    print(json.dumps(result["drug_corrections"], indent=2))

if __name__ == "__main__":
    test_agent_with_synthetic_data()
