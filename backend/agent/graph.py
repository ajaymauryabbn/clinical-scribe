import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from backend.agent.tools import extract_vitals, map_hindi_phrases, correct_drug_names, search_icd10, redact_pii, medical_ner
from backend.agent.prompts import (
    CONDITION_EXTRACTION_PROMPT,
    ICD_REFINE_PROMPT_NO_CANDIDATES,
    ICD_REFINE_PROMPT_WITH_CANDIDATES,
    SOAP_GENERATION_PROMPT,
    SOAP_VERIFICATION_PROMPT,
    PRESCRIPTION_PARSING_PROMPT
)
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
    scispacy_entities: list
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
    Node to extract medical entities using local tools and apply drug corrections.
    """
    print("---EXTRACTING ENTITIES---")
    
    # Use redacted transcript for privacy
    transcript = state.get("redacted_transcript", state["transcript"])
    
    # Concatenate turns for tool processing
    doctor_turns = "\n".join([t["text"] for t in transcript if t["speaker"] == "DOCTOR"])
    patient_turns = "\n".join([t["text"] for t in transcript if t["speaker"] == "PATIENT"])
    full_text = f"{doctor_turns}\n{patient_turns}"
    
    vitals = extract_vitals(full_text)
    hindi_matches = map_hindi_phrases(full_text)
    drug_corrections = correct_drug_names(doctor_turns)
    scispacy_entities = medical_ner(full_text)
    
    # Apply drug corrections to the transcript for better SOAP generation
    corrected_transcript = []
    for turn in transcript:
        new_turn = turn.copy()
        if turn["speaker"] == "DOCTOR":
            for corr in drug_corrections:
                # Simple replacement (case-insensitive where possible)
                pattern = re.compile(re.escape(corr["original"]), re.IGNORECASE)
                new_turn["text"] = pattern.sub(corr["corrected"], new_turn["text"])
        corrected_transcript.append(new_turn)
    
    return {
        "doctor_turns": doctor_turns,
        "patient_turns": patient_turns,
        "entities": {
            "vitals": vitals,
            "hindi_phrases": hindi_matches
        },
        "scispacy_entities": scispacy_entities,
        "drug_corrections": drug_corrections,
        "redacted_transcript": corrected_transcript
    }

def lookup_icd_node(state: ScribeState):
    """
    Node to map diagnosis terms to ICD-10 codes using local tool + LLM.
    """
    print("---LOOKING UP ICD CODES---")
    
    # 1. Ask LLM to extract specific medical conditions from findings
    prompt_extract = CONDITION_EXTRACTION_PROMPT.format(
        entities={**state['entities'], "scispacy": state.get('scispacy_entities', [])},
        doctor_turns=state['doctor_turns']
    )
    
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
        prompt_refine = ICD_REFINE_PROMPT_NO_CANDIDATES.format(conditions=conditions)
    else:
        prompt_refine = ICD_REFINE_PROMPT_WITH_CANDIDATES.format(
            conditions=conditions,
            icd_candidates=icd_candidates
        )
        
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
    
    prompt = SOAP_GENERATION_PROMPT.format(
        transcript=transcript,
        vitals=state['entities']['vitals'],
        hindi_phrases=state['entities']['hindi_phrases'],
        icd_codes=state['icd_codes']
    )
    
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
    
    prompt = SOAP_VERIFICATION_PROMPT.format(
        transcript=transcript,
        soap_note=state['soap_note']
    )
    
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
        
    prompt = PRESCRIPTION_PARSING_PROMPT.format(plan=plan)
    
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
