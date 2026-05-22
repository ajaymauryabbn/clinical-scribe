import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from backend.agent.tools import extract_vitals, map_hindi_phrases, correct_drug_names, search_icd10, redact_pii
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

class ScribeState(TypedDict):
    transcript: list # List of dicts with speaker, text, start, end
    redacted_transcript: list
    doctor_turns: str
    patient_turns: str
    entities: dict
    icd_codes: list
    drug_corrections: list
    soap_note: dict
    prescriptions: list
    flags: list

def redact_pii_node(state: ScribeState):
    """
    Node to redact PII from the transcript before further processing.
    """
    print("---REDACTING PII---")
    redacted_transcript = []
    for turn in state["transcript"]:
        redacted_turn = turn.copy()
        redacted_turn["text"] = redact_pii(turn["text"])
        redacted_transcript.append(redacted_turn)
        
    return {"redacted_transcript": redacted_transcript}

def extract_entities_node(state: ScribeState):
    """
    Node to extract medical entities using local tools.
    """
    print("---EXTRACTING ENTITIES---")
    
    # Use redacted transcript for privacy
    transcript = state.get("redacted_transcript", state["transcript"])
    
    # Concatenate doctor turns
    doctor_turns = "\n".join([t["text"] for t in transcript if t["speaker"] == "DOCTOR"])
    patient_turns = "\n".join([t["text"] for t in transcript if t["speaker"] == "PATIENT"])
    full_text = f"{doctor_turns}\n{patient_turns}"
    
    vitals = extract_vitals(full_text)
    hindi_matches = map_hindi_phrases(full_text)
    drug_corrections = correct_drug_names(doctor_turns)
    
    return {
        "doctor_turns": doctor_turns,
        "patient_turns": patient_turns,
        "entities": {
            "vitals": vitals,
            "hindi_phrases": hindi_matches
        },
        "drug_corrections": drug_corrections
    }

def lookup_icd_node(state: ScribeState):
    """
    Node to map diagnosis terms to ICD-10 codes using local tool + LLM.
    """
    print("---LOOKING UP ICD CODES---")
    
    # 1. Ask LLM to extract specific medical conditions from findings
    prompt_extract = f"""
    Based on the following findings, list the specific medical conditions or diagnoses mentioned or implied.
    Findings: {state['entities']}
    Transcript: {state['doctor_turns']}
    
    Output ONLY a JSON object with a 'conditions' key containing a list of strings (e.g., {{"conditions": ["Hypertension", "Type 2 Diabetes"]}}).
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt_extract}],
        response_format={"type": "json_object"},
    )
    
    try:
        content = json.loads(response.choices[0].message.content)
        conditions = content.get("conditions", [])
    except Exception as e:
        conditions = []
        
    # 2. Use local tool to find candidate codes for each condition
    icd_candidates = []
    for condition in conditions:
        matches = search_icd10(condition)
        if matches:
            icd_candidates.extend(matches)
            
    # 3. Use LLM to refine and select the most accurate codes
    if not icd_candidates:
        # Fallback to LLM if no local matches found
        prompt_refine = f"""
        Suggest the most relevant ICD-10 codes for these conditions: {conditions}.
        Output ONLY a JSON object with a 'codes' key containing a list of objects with 'code' and 'description'.
        """
    else:
        prompt_refine = f"""
        From the following candidate ICD-10 codes, select the most accurate ones for the conditions identified: {conditions}.
        Candidates: {icd_candidates}
        
        Output ONLY a JSON object with a 'codes' key containing a list of objects with 'code' and 'description'.
        """
        
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt_refine}],
        response_format={"type": "json_object"},
    )
    
    try:
        content = json.loads(response.choices[0].message.content)
        icd_codes = content.get("codes", []) if isinstance(content, dict) else content
    except Exception as e:
        icd_codes = icd_candidates[:3] # Fallback to first 3 tool results
        
    return {"icd_codes": icd_codes}

def generate_soap_node(state: ScribeState):
    """
    Node to generate the final SOAP note using the LLM.
    """
    print("---GENERATING SOAP NOTE---")
    
    transcript = state.get("redacted_transcript", state["transcript"])
    
    prompt = f"""
    You are a clinical documentation specialist. Generate a structured English SOAP note from this Hinglish transcript.
    
    Transcript:
    {transcript}
    
    Extracted Vitals: {state['entities']['vitals']}
    Hindi phrase translations: {state['entities']['hindi_phrases']}
    Suggested ICD-10: {state['icd_codes']}
    
    Rules:
    - Translate clinical Hindi to medical English.
    - Output ONLY JSON with keys: subjective, objective, assessment, plan.
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    try:
        soap_note = json.loads(response.choices[0].message.content)
    except:
        soap_note = {"error": "Failed to parse SOAP"}
        
    return {"soap_note": soap_note}

def verify_soap_node(state: ScribeState):
    """
    Node to verify the SOAP note for clinical consistency and accuracy.
    """
    print("---VERIFYING SOAP NOTE---")
    
    transcript = state.get("redacted_transcript", state["transcript"])
    
    prompt = f"""
    You are a clinical safety auditor. Cross-check the generated SOAP note against the original transcript.
    
    Transcript: {transcript}
    Generated SOAP: {state['soap_note']}
    
    Check for:
    1. Factual contradictions (e.g., wrong BP, wrong drug dose).
    2. Missing clinical findings mentioned in the transcript.
    3. Hallucinations (findings in SOAP not in transcript).
    
    Output ONLY JSON with keys:
    - 'consistent': boolean
    - 'corrections': list of strings describing discrepancies
    - 'flags': list of items the doctor MUST verify (e.g., "Verify Amlodipine dose", "High BP 160/100 not addressed")
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    try:
        verification = json.loads(response.choices[0].message.content)
        flags = verification.get("flags", [])
    except:
        flags = ["Verification step failed - please review carefully"]
        
    return {"flags": flags}

def parse_prescription_node(state: ScribeState):
    """
    Node to extract structured prescription data from the Plan section.
    """
    print("---PARSING PRESCRIPTIONS---")
    
    plan = state["soap_note"].get("plan", "")
    if not plan:
        return {"prescriptions": []}
        
    prompt = f"""
    Extract structured prescription data from this clinical plan:
    {plan}
    
    Output ONLY a JSON list of objects with:
    - 'drug': medication name
    - 'dose': e.g., '5mg'
    - 'frequency': e.g., 'once daily' or '1-0-1'
    - 'duration': e.g., '30 days'
    - 'route': e.g., 'oral'
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    try:
        content = json.loads(response.choices[0].message.content)
        prescriptions = content.get("prescriptions", []) if isinstance(content, dict) else content
    except:
        prescriptions = []
        
    return {"prescriptions": prescriptions}

def create_scribe_graph():
    workflow = StateGraph(ScribeState)

    workflow.add_node("redact_pii", redact_pii_node)
    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("lookup_icd", lookup_icd_node)
    workflow.add_node("generate_soap", generate_soap_node)
    workflow.add_node("verify_soap", verify_soap_node)
    workflow.add_node("parse_prescription", parse_prescription_node)

    workflow.set_entry_point("redact_pii")
    workflow.add_edge("redact_pii", "extract_entities")
    workflow.add_edge("extract_entities", "lookup_icd")
    workflow.add_edge("lookup_icd", "generate_soap")
    workflow.add_edge("generate_soap", "verify_soap")
    workflow.add_edge("verify_soap", "parse_prescription")
    workflow.add_edge("parse_prescription", END)

    return workflow.compile()

scribe_agent = create_scribe_graph()
