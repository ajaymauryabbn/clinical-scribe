# Prompt templates for the Clinical Scribe agent nodes

PII_REDACTION_SYSTEM = """
You are a privacy-first clinical assistant. Your task is to redact all Personally Identifiable Information (PII) from the given transcript.
Redact:
- Patient names
- Phone numbers
- Email addresses
- Aadhaar numbers or other ID numbers
- Detailed addresses

Replace PII with placeholders like [NAME], [PHONE], [EMAIL], [ADDRESS].
"""

CONDITION_EXTRACTION_PROMPT = """
Based on the following clinical findings and transcript, list the specific medical conditions or diagnoses mentioned or implied.

Findings: {entities}
Transcript: {doctor_turns}

Output ONLY a JSON object with a 'conditions' key containing a list of strings (e.g., {{"conditions": ["Hypertension", "Type 2 Diabetes"]}}).
"""

ICD_REFINE_PROMPT_NO_CANDIDATES = """
Suggest the most relevant ICD-10 codes for these conditions: {conditions}.
Output ONLY a JSON object with a 'codes' key containing a list of objects with 'code' and 'description'.
"""

ICD_REFINE_PROMPT_WITH_CANDIDATES = """
From the following candidate ICD-10 codes, select the most accurate ones for the conditions identified: {conditions}.
Candidates: {icd_candidates}

Output ONLY a JSON object with a 'codes' key containing a list of objects with 'code' and 'description'.
"""

SOAP_GENERATION_PROMPT = """
You are a clinical documentation specialist. Generate a structured English SOAP note from this Hinglish transcript.

Transcript:
{transcript}

Extracted Vitals: {vitals}
Hindi phrase translations: {hindi_phrases}
Suggested ICD-10: {icd_codes}

Rules:
- Translate clinical Hindi to medical English.
- Output ONLY JSON with keys: subjective, objective, assessment, plan.
"""

SOAP_VERIFICATION_PROMPT = """
You are a clinical safety auditor. Cross-check the generated SOAP note against the original transcript.

Transcript: {transcript}
Generated SOAP: {soap_note}

Check for:
1. Factual contradictions (e.g., wrong BP, wrong drug dose).
2. Missing clinical findings mentioned in the transcript.
3. Hallucinations (findings in SOAP not in transcript).

Output ONLY JSON with keys:
- 'consistent': boolean
- 'corrections': list of strings describing discrepancies
- 'flags': list of items the doctor MUST verify (e.g., "Verify Amlodipine dose", "High BP 160/100 not addressed")
"""

PRESCRIPTION_PARSING_PROMPT = """
Extract structured prescription data from this clinical plan:
{plan}

Output ONLY a JSON object with a 'prescriptions' key containing a list of objects with:
- 'drug': medication name
- 'dose': e.g., '5mg'
- 'frequency': e.g., 'once daily' or '1-0-1'
- 'duration': e.g., '30 days'
- 'route': e.g., 'oral'
"""
