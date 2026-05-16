import os
import json
import time
import concurrent.futures
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import random

# Load environment variables
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# Expanded Scenarios
SCENARIOS = {
    "MAJOR": [
        "Unstable Angina / Suspected MI (Immediate Referral)",
        "Severe Abdominal Pain - Suspected Appendicitis",
        "Chronic Kidney Disease Stage 4 - Dialysis Discussion",
        "Neurological Deficit - Suspected Stroke",
        "Post-COVID Lung Fibrosis with severe hypoxia",
        "Uncontrolled Type 1 Diabetes with Ketoacidosis signs",
        "Suspected Malignancy - Breast Lump follow-up",
        "Congestive Heart Failure - Grade 3",
        "Severe Osteoarthritis - Surgery Referral",
        "Psychiatric Emergency - Severe Depression with Suicidal Ideation"
    ],
    "MINOR": [
        "Mild Acidity & Lifestyle counseling",
        "Grade 1 Fatty Liver - Diet & Exercise focus",
        "Vitamin D deficiency & sun exposure advice",
        "Pre-hypertension - stress management & yoga",
        "Mild dandruff / Seborrheic dermatitis",
        "Occasional Tension Headache - posture correction",
        "Slight Obesity - Calorie deficit counseling",
        "Dry Eye Syndrome from screen time",
        "Mild Varicose Veins - Compression stockings",
        "Common wart removal follow-up"
    ],
    "SEASONAL": [
        "Viral Fever / Flu during Monsoon",
        "Dengue Fever follow-up (Platelet monitoring)",
        "Typhoid Fever - Antibiotic course",
        "Allergic Rhinitis - Dust/Pollen allergy",
        "Heat Stroke / Exhaustion in Summer",
        "Acute Diarrhea / Food Poisoning",
        "Asthma exacerbation in Winter smog",
        "Conjunctivitis outbreak",
        "Skin Fungal Infection (Tinea) in humidity",
        "Malaria - Chills and Rigors"
    ]
}

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai", "Lucknow", "Ahmedabad"]
LANGUAGE_MIXES = [
    "Full English (Academic style)",
    "Mostly English with very few Hindi fillers",
    "Moderate Hinglish (50-50 mix)",
    "Heavy Hindi (80% Hindi, English only for medical terms)",
    "Heavy Hindi (90% Hindi, deep colloquial dialect)"
]

SYSTEM_PROMPT = "You are generating realistic synthetic medical consultation transcripts for a clinical documentation AI. Fictional data only."

TRANSCRIPT_PROMPT = """
Generate a realistic doctor-patient consultation transcript.

Scenario: {scenario}
Complexity: {complexity}
City Context: {city}
Language Mix: {lang_mix}
Duration: 4-6 minutes (30-45 exchanges)

Rules:
- STRICTLY follow the Language Mix: {lang_mix}.
- If 'Heavy Hindi', the doctor speaks mostly in Hindi but uses English for 'BP', 'Prescription', 'Surgery', 'Infection', and drug names.
- If 'Major', the doctor must express urgency and potentially refer the patient to a specialist or hospital.
- If 'Minor', the focus is on lifestyle, diet, and minimal meds.
- Use real Indian drug names (e.g., Telma, Glycomet, Pan-D, Dolo, Augmentin).
- DO NOT use placeholders like [Doctor]. Use DOCTOR: and PATIENT:.
- Format:
DOCTOR: <text>
PATIENT: <text>

End with a prescription/advice block.
"""

SOAP_PROMPT = """
Generate a structured English SOAP note from the transcript.
Output ONLY valid JSON with keys: subjective, objective, assessment, plan.
Transcript:
{transcript}
"""

def generate_single_point(i: int):
    # Randomly select parameters
    complexity = random.choice(["MAJOR", "MINOR", "SEASONAL"])
    scenario = random.choice(SCENARIOS[complexity])
    city = random.choice(CITIES)
    lang_mix = random.choice(LANGUAGE_MIXES)
    
    try:
        # 1. Transcript
        t_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": TRANSCRIPT_PROMPT.format(
                    scenario=scenario, complexity=complexity, city=city, lang_mix=lang_mix
                )}
            ]
        )
        transcript = t_resp.choices[0].message.content

        # 2. SOAP
        s_resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a clinical documentation specialist. Output ONLY JSON."},
                {"role": "user", "content": SOAP_PROMPT.format(transcript=transcript)}
            ],
            response_format={"type": "json_object"}
        )
        soap = json.loads(s_resp.choices[0].message.content)

        data = {
            "id": f"gen_{i}_{int(time.time())}",
            "metadata": {
                "complexity": complexity,
                "scenario": scenario,
                "city": city,
                "lang_mix": lang_mix
            },
            "transcript": transcript,
            "soap": soap
        }
        
        # Save immediately
        folder = f"data/synthetic/{complexity.lower()}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/case_{i}.json", "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Generated {i}/500: {scenario} ({complexity})")
        return True
    except Exception as e:
        print(f"Error on {i}: {e}")
        return False

def main():
    total_points = 500
    max_workers = 8 # As requested
    
    print(f"Starting generation of {total_points} points with {max_workers} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_point, i) for i in range(1, total_points + 1)]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
