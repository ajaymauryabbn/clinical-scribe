import os
import pandas as pd
from typing import List, Dict
import re
from rapidfuzz import process, fuzz
import simple_icd_10 as icd
import spacy

# Load dictionaries
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
HINDI_MEDICAL_PATH = os.path.join(DATA_DIR, "hindi_medical.csv")
DRUG_NAMES_PATH = os.path.join(DATA_DIR, "drug_names.csv")

# Load scispaCy model
try:
    nlp = spacy.load("en_core_sci_sm")
except Exception as e:
    print(f"Warning: Could not load scispaCy model: {e}")
    nlp = None

try:
    hindi_medical_df = pd.read_csv(HINDI_MEDICAL_PATH)
    drug_names_df = pd.read_csv(DRUG_NAMES_PATH)
except Exception as e:
    print(f"Warning: Could not load dictionaries: {e}")
    hindi_medical_df = pd.DataFrame()
    drug_names_df = pd.DataFrame()

def medical_ner(text: str) -> List[str]:
    """
    Extracts medical entities using scispaCy.
    """
    if not nlp:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def redact_pii(text: str) -> str:
    """
    Simple regex-based PII redaction for names and numbers.
    Redacts:
    - Phone numbers (10 digits)
    - Patterns like "My name is [Name]" or "I am [Name]"
    - Email addresses
    """
    # Redact phone numbers
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)
    
    # Redact email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    
    # Redact names in common patterns (Simple heuristic)
    name_patterns = [
        r"(?:my name is|i am|patient name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"(?:naam hai|mera naam)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, text, re.I)
        for match in matches:
            name = match.group(1)
            # Only redact if it looks like a name (not a common word)
            # For now, just redact any match to be safe
            text = text.replace(name, '[NAME]')
            
    return text

def search_icd10(query: str, max_results: int = 3) -> List[Dict]:
    """
    Searches for ICD-10 codes by keyword in description.
    """
    if not query:
        return []
        
    all_codes = icd.get_all_codes()
    # Simple search: keyword in description
    results = []
    query_lower = query.lower()
    
    for code in all_codes:
        desc = icd.get_description(code)
        if query_lower in desc.lower():
            results.append({
                "code": code,
                "description": desc
            })
            if len(results) >= max_results:
                break
                
    return results

def correct_drug_names(text: str) -> List[Dict]:
    """
    Corrects misspellings of Indian drug brands using fuzzy matching.
    """
    corrections = []
    if drug_names_df.empty:
        return corrections
        
    # Remove punctuation and split into words
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()
    brand_names = drug_names_df["brand_name"].tolist()
    
    for word in words:
        # Check words length >= 3 to catch 'Pan', 'Avil', 'Dolo'
        if len(word) >= 3:
            match = process.extractOne(word, brand_names, scorer=fuzz.WRatio)
            
            # Stricter threshold for short words to avoid false positives with common Hindi/English words
            threshold = 95 if len(word) < 5 else 85
            
            if match and match[1] >= threshold and match[0].lower() != word.lower():
                corrections.append({
                    "original": word,
                    "corrected": match[0],
                    "confidence": match[1]
                })
    return corrections

def extract_vitals(text: str) -> Dict:
    """
    Extracts vitals (BP, Pulse, Temp, SpO2) using regex.
    Adds [VERIFY] tag as per spec requirement for safety.
    """
    vitals = {}
    
    # BP: 120/80
    bp_match = re.search(r"(\d{2,3}/\d{2,3})", text)
    if bp_match:
        vitals["BP"] = f"{bp_match.group(1)} [VERIFY]"
        
    # Pulse/Heart Rate
    pulse_match = re.search(r"(?:pulse|heart rate|HR)\D*(\d{2,3})", text, re.I)
    if pulse_match:
        vitals["Pulse"] = f"{pulse_match.group(1)} bpm [VERIFY]"
        
    # Temperature
    temp_match = re.search(r"(\d{2,3}(?:\.\d)?)\D*(?:F|C|temp|degree)", text, re.I)
    if temp_match:
        vitals["Temperature"] = f"{temp_match.group(1)} [VERIFY]"
        
    # SpO2
    spo2_match = re.search(r"(?:spo2|oxygen|saturation)\D*(\d{2,3})", text, re.I)
    if spo2_match:
        vitals["SpO2"] = f"{spo2_match.group(1)}% [VERIFY]"
        
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
