import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from backend.agent.tools import extract_vitals, map_hindi_phrases, correct_drug_names
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

class ScribeState(TypedDict):
    transcript: list # List of dicts with speaker, text, start, end
    doctor_turns: str
    patient_turns: str
    entities: dict
    icd_codes: list
    drug_corrections: list
    soap_note: dict
    flags: list

def extract_entities_node(state: ScribeState):
    """
    Node to extract medical entities using local tools.
    """
    print("---EXTRACTING ENTITIES---")
    
    # Concatenate doctor turns
    doctor_turns = "\n".join([t["text"] for t in state["transcript"] if t["speaker"] == "DOCTOR"])
    patient_turns = "\n".join([t["text"] for t in state["transcript"] if t["speaker"] == "PATIENT"])
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
    Node to map diagnosis terms to ICD-10 codes using LLM.
    """
    print("---LOOKING UP ICD CODES---")
    
    prompt = f"""
    Given the following clinical findings and symptoms, suggest the most relevant ICD-10 codes.
    Findings: {state['entities']}
    
    Output ONLY a JSON list of objects with 'code' and 'description'.
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    
    try:
        content = json.loads(response.choices[0].message.content)
        icd_codes = content.get("codes", []) if isinstance(content, dict) else content
    except:
        icd_codes = []
        
    return {"icd_codes": icd_codes}

def generate_soap_node(state: ScribeState):
    """
    Node to generate the final SOAP note using the LLM.
    """
    print("---GENERATING SOAP NOTE---")
    
    prompt = f"""
    You are a clinical documentation specialist. Generate a structured English SOAP note from this Hinglish transcript.
    
    Transcript:
    {state['transcript']}
    
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

def create_scribe_graph():
    workflow = StateGraph(ScribeState)

    workflow.add_node("extract_entities", extract_entities_node)
    workflow.add_node("lookup_icd", lookup_icd_node)
    workflow.add_node("generate_soap", generate_soap_node)

    workflow.set_entry_point("extract_entities")
    workflow.add_edge("extract_entities", "lookup_icd")
    workflow.add_edge("lookup_icd", "generate_soap")
    workflow.add_edge("generate_soap", END)

    return workflow.compile()

scribe_agent = create_scribe_graph()
