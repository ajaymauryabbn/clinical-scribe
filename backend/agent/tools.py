import os
import pandas as pd
from typing import List, Dict
import re
from rapidfuzz import process, fuzz

# Load dictionaries
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HINDI_MEDICAL_PATH = os.path.join(DATA_DIR, "hindi_medical.csv")
DRUG_NAMES_PATH = os.path.join(DATA_DIR, "drug_names.csv")

try:
    hindi_medical_df = pd.read_csv(HINDI_MEDICAL_PATH)
    drug_names_df = pd.read_csv(DRUG_NAMES_PATH)
except Exception as e:
    print(f"Warning: Could not load dictionaries: {e}")
    hindi_medical_df = pd.DataFrame()
    drug_names_df = pd.DataFrame()

def correct_drug_names(text: str) -> List[Dict]:
    """
    Corrects misspellings of Indian drug brands using fuzzy matching.
    """
    corrections = []
    if drug_names_df.empty:
        return corrections
        
    words = text.split()
    brand_names = drug_names_df["brand_name"].tolist()
    
    for word in words:
        # Simple heuristic: only check words starting with capital or length > 4
        if len(word) > 4:
            match = process.extractOne(word, brand_names, scorer=fuzz.WRatio)
            if match and match[1] > 85 and match[0] != word:
                corrections.append({
                    "original": word,
                    "corrected": match[0],
                    "confidence": match[1]
                })
    return corrections

def extract_vitals(text: str) -> Dict:
    """
    Extracts vitals (BP, Pulse, Temp, SpO2) using regex.
    """
    vitals = {}
    
    # BP: 120/80
    bp_match = re.search(r"(\d{2,3}/\d{2,3})", text)
    if bp_match:
        vitals["BP"] = bp_match.group(1)
        
    # Pulse/Heart Rate
    pulse_match = re.search(r"(?:pulse|heart rate|HR)\D*(\d{2,3})", text, re.I)
    if pulse_match:
        vitals["Pulse"] = pulse_match.group(1)
        
    # Temperature
    temp_match = re.search(r"(\d{2,3}(?:\.\d)?)\D*(?:F|C|temp)", text, re.I)
    if temp_match:
        vitals["Temperature"] = temp_match.group(1)
        
    return vitals

def map_hindi_phrases(text: str) -> List[Dict]:
    """
    Maps common Hindi medical phrases to English terms.
    """
    matches = []
    if hindi_medical_df.empty:
        return matches
        
    text_lower = text.lower()
    for _, row in hindi_medical_df.iterrows():
        if row["hindi_phrase"].lower() in text_lower:
            matches.append({
                "hindi": row["hindi_phrase"],
                "english": row["english_medical_term"],
                "section": row["soap_section"]
            })
    return matches
