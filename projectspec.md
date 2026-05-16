# Hinglish Clinical Scribe — Project Spec

> **Status:** Pre-build  
> **Owner:** Ajay Maurya  
> **Target:** Portfolio project → potential product  
> **Stack:** Python · Whisper · LangGraph · GPT-4o · Streamlit · HuggingFace Spaces · FastAPI · Render

---

## Table of contents

1. [Why this exists](#1-why-this-exists)
2. [What we are building](#2-what-we-are-building)
3. [Who it is for](#3-who-it-is-for)
4. [The Hinglish problem](#4-the-hinglish-problem)
5. [Data sources](#5-data-sources)
6. [Synthetic data generation](#6-synthetic-data-generation)
7. [High-level technical architecture](#7-high-level-technical-architecture)
8. [Component-level implementation guide](#8-component-level-implementation-guide)
9. [Product design](#9-product-design)
10. [Deployment and debugging](#10-deployment-and-debugging)
11. [Safety, privacy and disclaimers](#11-safety-privacy-and-disclaimers)
12. [Future versions](#12-future-versions)
13. [Open questions and decisions log](#13-open-questions-and-decisions-log)

---

## 1. Why this exists

### The market problem

Indian doctors spend 30–40% of their clinic time on documentation. A consultation lasting 8 minutes generates a clinical note that takes another 6–10 minutes to write. At a private clinic seeing 30–50 patients a day, this is 3–5 hours of lost time per doctor per day.

Existing solutions (Abridge, Nuance DAX, DeepScribe) solve this well — but exclusively for English-speaking doctors in US and UK health systems. They are:

- Priced for US enterprise ($300–$1,200/month per seat)
- Trained on American English clinical speech
- Integrated with US EHR systems (Epic, Cerner) that Indian clinics don't use
- Blind to the linguistic reality of Indian medical practice

### The linguistic reality

Indian doctors — particularly in urban private practice — do not speak in clean English or clean Hindi. They speak **Hinglish**: a fluid, mid-sentence code-mix of Hindi and English that is natural, fast, and entirely unaddressed by any existing clinical AI product.

A real consultation in Delhi sounds like this:

> "BP kaisa hai? 140/90? Theek hai, let's start Amlodipine 5mg. Pet mein koi dard toh nahi? Fasting sugar bhi karwa lo."

No existing transcription or SOAP generation product handles this well. Whisper will transcribe it imperfectly. GPT-4 will misread clinical intent. No product wraps them together into a usable clinical note for Indian doctors.

This is the gap.

### Why build this now

- India has 1.3 million registered allopathic doctors, ~200k in metro private practice
- India's healthtech market is growing fast (Practo, HealthPlix, Innovaccer all scaling)
- The DPDP Act (India's data privacy law, 2023) is creating demand for on-premise / no-data-retention solutions
- The tech stack to build this (Whisper, LangGraph, pyannote) became production-ready in 2024
- No direct competitor exists in this specific niche

### Why this project for the portfolio

This project demonstrates: voice AI, LLMs, agentic pipelines, code-mix NLP, medical domain, and deployment — all in a single working demo. It is immediately understandable to any recruiter or hiring manager because clinical documentation is a universally-known pain. It is differentiated because the Hinglish angle is specific, real, and uncrowded.

---

## 2. What we are building

A web application that:

1. Accepts audio input (microphone recording or file upload) of a doctor-patient consultation in Hinglish
2. Transcribes and diarizes it (separates doctor vs patient voice)
3. Runs a LangGraph agent over the transcript to extract medical entities, map ICD-10 codes, and identify drug names
4. Generates a structured SOAP note in English, suitable for clinical records
5. Displays the transcript and SOAP note side by side, with copy and download options

**What it is not:**
- Not a real-time live transcription product (V1 is upload/record and process)
- Not an EHR integration (V1 outputs text only)
- Not a diagnostic tool — it documents what the doctor said, it does not suggest diagnoses
- Not storing any patient data server-side

---

## 3. Who it is for

### Primary user (V1)
**Urban Indian private practice doctor**
- MBBS / MD, working in metro cities (Delhi, Mumbai, Bengaluru, Hyderabad)
- Sees 30–60 patients/day in a private clinic or hospital OPD
- Speaks Hinglish naturally during consultations
- Uses a smartphone or laptop
- Currently writes notes manually or dictates to a receptionist

### Secondary user (demo audience)
**Healthtech founders, CPOs, recruiters**
- Need to understand the product in under 60 seconds
- Will not interact via voice themselves
- Need a scenario selector / pre-built demo mode

### Not the target user (yet)
- Doctors in rural areas (connectivity constraints)
- Government hospital doctors (different documentation workflows)
- Doctors who exclusively speak regional languages other than Hindi

---

## 4. The Hinglish problem

This section exists to explain the core technical challenge to any implementer.

### What code-mixing means in practice

Code-mixing in Indian medical speech has three patterns:

**Pattern 1 — Inter-sentential** (switch between sentences)
> "Chest mein pain kab se hai? How many days?"

**Pattern 2 — Intra-sentential** (switch within a sentence — most common)
> "Aapka BP thoda high hai, let's start medication."

**Pattern 3 — Medical English embedded in Hindi syntax**
> "Fasting sugar 180 aa raha hai, HbA1c bhi karao."

Drug names, test names, and vital measurements are almost always spoken in English even when the surrounding sentence is Hindi.

### Why standard STT fails

Whisper large-v3 handles Hinglish better than any other open model, but it has two failure modes:

**Failure mode 1 — Language flip mid-word**
Whisper must commit to a language at the segment level. If a segment starts in Hindi and the doctor switches to English mid-sentence, Whisper sometimes transcribes the English portion in Hindi transliteration or drops it.

**Mitigation:** Transcribe with `language=None` (auto-detect) per speaker segment, not per file. Keep segments short (5–15 seconds each after diarization).

**Failure mode 2 — Medical term misrecognition**
"Amlodipine" becomes "Amla doping". "HbA1c" becomes "Habc". "Metformin" becomes "Metphomen".

**Mitigation:** Post-processing with a medical term correction lookup. Build a dictionary of 200 common Indian drug brand names and generic names. Run fuzzy match after transcription to correct obvious errors. Flag corrected terms with `[auto-corrected]` so the doctor can verify.

### SOAP generation in a mixed-language context

The LLM sees a transcript like:
> `DOCTOR: BP 140/90 hai. Let's start Amlodipine 5mg. Kal fasting sugar test karao.`

It must understand that:
- "BP 140/90 hai" → Blood pressure 140/90 mmHg (Objective)
- "Let's start Amlodipine 5mg" → Plan: initiate Amlodipine 5mg OD
- "Kal fasting sugar test karao" → Plan: fasting blood glucose test tomorrow

The system prompt must explicitly instruct the model to translate clinical meaning from Hindi phrases into standard English clinical documentation, not to transliterate.

---

## 5. Data sources

### 5.1 For transcription / STT

| Source | What it provides | Access | Use |
|--------|-----------------|--------|-----|
| `openai/whisper-large-v3` | Base STT model | HuggingFace, free | Core transcription |
| `pyannote/speaker-diarization-3.1` | Speaker separation | HuggingFace (accept terms) | Doctor vs patient labelling |
| `google/MedASR` | Medical-domain fine-tuned ASR | HuggingFace, free | Fallback / comparison |
| AI4Bharat IndicWhisper | Hindi-focused fine-tune of Whisper | HuggingFace, free | Test against for Hindi accuracy |

### 5.2 For SOAP note generation training / prompting

| Source | What it provides | Access | Use |
|--------|-----------------|--------|-----|
| `omi-health/medical-dialogue-to-soap-summary` | 10,000 dialogue-SOAP pairs (synthetic, English) | HuggingFace, open | Few-shot examples in prompt |
| MTSamples | 4,000+ real medical transcription reports | mtsamples.com, free | SOAP structure reference, medical vocabulary |
| NoteChat (ACL 2024) | Synthetic patient-physician dialogues conditioned on PMC notes | HuggingFace, open | Dialogue structure reference |
| PubMed Central (PMC) | Clinical case reports in English | Open access | Source for realistic medical scenarios |

### 5.3 For ICD-10 coding

| Source | What it provides | Access | Use |
|--------|-----------------|--------|-----|
| `simple_icd_10` Python library | Full ICD-10 hierarchy, lookup by code or description | PyPI, free, offline | ICD-10 code lookup in agent |
| WHO ICD-10 API | Live ICD-10 lookup | Free, requires registration | Optional fallback |

### 5.4 For drug name validation

| Source | What it provides | Access | Use |
|--------|-----------------|--------|-----|
| CDSCO drug database | Indian-approved drugs, brand names | cdsco.gov.in, public | Indian brand name validation |
| RxNorm API (US NLM) | Drug name normalization | Free, no key required | Generic name standardization |
| Manual curated list | Top 200 Indian brand names (Glycomet, Ecosprin, Telma, Cipla generics, etc.) | Build yourself | Post-STT correction dictionary |

### 5.5 For ICD-10 to plain language mapping (India-specific)

Build a small mapping CSV for the 50 most common Indian OPD conditions:
- Hypertension → ICD-10 I10
- Type 2 Diabetes → ICD-10 E11
- Acute URTI → ICD-10 J06.9
- Dengue fever → ICD-10 A90
- Typhoid → ICD-10 A01.0

(Full list to be built during development — derive from MTSamples + AIIMS OPD statistics)

---

## 6. Synthetic data generation

Since no real Hinglish clinical conversation dataset exists publicly, all training data and test data for V1 must be generated synthetically.

### 6.1 Generation strategy

Use GPT-4o to generate 200 realistic Hinglish consultation transcripts across 10 clinical scenarios. Each transcript should:
- Be 20–35 exchanges long (approx 3–5 minutes of speech)
- Have natural code-mixing (not forced — some doctors mix more, some less)
- Use real Indian drug brand names
- Cover one primary complaint and one or two secondary items
- Include realistic vitals, test results, and prescription language

### 6.2 Scenarios to cover (20 transcripts each)

1. Hypertension follow-up (most common Indian OPD complaint)
2. Type 2 diabetes management
3. Acute fever / viral URTI
4. Respiratory — asthma or COPD review
5. Gastroenterology — acidity, IBS, constipation
6. Musculoskeletal — back pain, joint pain
7. Dermatology — skin rash, allergy
8. Antenatal follow-up (ANC visit)
9. Paediatric fever (parent speaking, doctor in Hinglish)
10. Post-surgical follow-up

### 6.3 Generation prompt template

```python
SYSTEM = """
You are generating realistic synthetic medical consultation transcripts
for training and testing a clinical documentation AI.
These are fictional — no real patient data.
"""

USER = """
Generate a realistic doctor-patient consultation transcript for the following scenario.

Scenario: {scenario}
City: {city}  # Delhi / Mumbai / Bengaluru / Hyderabad — affects dialect slightly
Doctor style: {style}  # 'heavy Hinglish' / 'moderate Hinglish' / 'mostly English with Hindi phrases'
Duration: approximately 4 minutes (25-35 exchanges)

Rules:
- Doctor naturally code-mixes Hindi and English mid-sentence
- Patient responds mostly in Hindi, occasionally English
- Use real Indian drug brand names: Glycomet, Ecosprin, Telma-40, Shelcal, Pan-D, Dolo-650, Augmentin, Azithral, Metolar, etc.
- Include: chief complaint, history, vitals (BP, pulse, temperature, SpO2 where relevant), examination findings, diagnosis, prescription with dose and duration
- Do NOT use placeholder names like [Doctor] — use DOCTOR: and PATIENT:
- Write numbers as spoken: "one forty over ninety" or "140/90" (vary naturally)
- Include filler phrases: "haan", "theek hai", "dekhte hain", "koi baat nahi"

Format:
DOCTOR: <text>
PATIENT: <text>
...

End with a prescription block in the doctor's final lines.
"""
```

### 6.4 Generating corresponding ground-truth SOAP notes

For each generated transcript, make a second GPT-4o call to generate the gold-standard SOAP note. This becomes the evaluation target.

```python
SOAP_PROMPT = """
You are a clinical documentation specialist.
Given the consultation transcript below, generate a structured SOAP note in English.

Rules:
- Translate all Hindi clinical phrases into standard English medical language
- "pet mein dard" → "abdominal pain"
- "bukhaar" → "fever"
- "BP" stays as "BP" (already English)
- Preserve all drug names, dosages, and frequencies exactly as stated
- Flag any ambiguous or unclear items with [VERIFY]
- Use standard SOAP format: Subjective, Objective, Assessment, Plan

Transcript:
{transcript}

Output SOAP note:
"""
```

### 6.5 Storage format

Save each pair as a JSON file:

```json
{
  "id": "consult_001",
  "scenario": "hypertension_followup",
  "city": "Delhi",
  "doctor_style": "heavy_hinglish",
  "transcript": [
    {"speaker": "DOCTOR", "text": "BP kaisa hai aaj?", "turn": 1},
    {"speaker": "PATIENT", "text": "Thoda high aa raha hai, 150/95", "turn": 2}
  ],
  "raw_transcript_text": "DOCTOR: BP kaisa hai aaj?\nPATIENT: ...",
  "soap_note": {
    "subjective": "...",
    "objective": "...",
    "assessment": "...",
    "plan": "..."
  },
  "metadata": {
    "generated_at": "2026-05-01",
    "model": "gpt-4o",
    "approx_duration_minutes": 4
  }
}
```

### 6.6 Estimated cost

- 200 transcripts × ~1,500 tokens each = 300k tokens generation
- 200 SOAP notes × ~500 tokens each = 100k tokens
- Total: ~400k tokens at GPT-4o pricing (~$0.40–$1.20 depending on input/output split)
- Budget: under $5 total

### 6.7 Quality check

After generation, manually review 20 random transcripts (10% sample) for:
- Natural-sounding Hinglish (not robotic switching)
- Medically plausible content
- Correct drug names and dosages
- No hallucinated test values that are clinically impossible

Flag and regenerate any that fail the spot check.

---

## 7. High-level technical architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (HuggingFace Spaces)           │
│   [Record / Upload]  ←→  [Transcript view]  ←→  [SOAP output]  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP POST (audio bytes)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Render)                      │
│  /transcribe  →  Whisper + pyannote                             │
│  /generate    →  LangGraph agent                                │
└──────┬──────────────┬───────────────────────────────────────────┘
       │              │
       ▼              ▼
  Whisper v3      LangGraph Agent
  pyannote          ├── NER tool (scispaCy / regex)
  diarization       ├── ICD-10 tool (simple_icd_10)
                    ├── Drug correction tool (curated dict)
                    └── SOAP generator (GPT-4o)
```

### Technology choices and rationale

| Component | Choice | Why | Alternative if broken |
|-----------|--------|-----|-----------------------|
| STT | Whisper large-v3 | Best Hindi+English code-mix accuracy, free | OpenAI Whisper API ($0.006/min) |
| Diarization | pyannote 3.1 | Best open speaker separation | Simple silence-based segmentation |
| Agent framework | LangGraph | Your existing skill, good for sequential tool chaining | LangChain LCEL |
| NER | scispaCy `en_core_sci_sm` | Lightweight, clinical entity extraction | Regex-based extraction fallback |
| ICD-10 | `simple_icd_10` PyPI | Offline, fast, no API key | WHO ICD-10 API |
| LLM | GPT-4o (via API) | Best SOAP quality, handles Hinglish context well | Claude Sonnet (Anthropic API) |
| UI | Streamlit | Fast to build, HuggingFace native | Gradio |
| Backend | FastAPI | Async, your existing skill | Flask |
| UI hosting | HuggingFace Spaces | Free, 16GB RAM, community visibility | Streamlit Cloud |
| Backend hosting | Render free tier | Free, FastAPI-friendly | Railway |

---

## 8. Component-level implementation guide

This section gives implementation direction without being prescriptive about exact API calls, which may change. Adapt based on what is available and compatible at runtime.

### 8.1 Audio input module

**What it does:** Accept audio from microphone (browser) or file upload (.wav, .mp3, .m4a, .webm).

**Implementation notes:**
- Streamlit has `st.audio_input()` for microphone recording (added in Streamlit 1.31). Check if available; fall back to `audio_recorder_streamlit` package if not.
- For file upload, accept `.wav`, `.mp3`, `.m4a` via `st.file_uploader()`.
- Convert all formats to 16kHz mono WAV before sending to backend. Use `pydub` or `ffmpeg` subprocess.
- Limit file size to 25MB in V1 (roughly 25 minutes of audio). Add a clear error message if exceeded.
- Send audio as multipart form data to the FastAPI backend `/transcribe` endpoint.

**Common issues:**
- Browser microphone permissions vary. Test in Chrome and Firefox.
- Streamlit's audio recorder may not work on all HuggingFace Spaces hardware. Have a file upload fallback prominently visible.
- `.webm` from browsers needs `ffmpeg` conversion. Ensure `ffmpeg` is in the Dockerfile or `packages.txt` for HuggingFace Spaces.

### 8.2 Transcription and diarization module

**What it does:** Separate speaker turns, then transcribe each turn.

**Implementation approach:**

```
Audio file
  → pyannote diarization → list of (start, end, speaker_label) segments
  → for each segment: extract audio slice → Whisper transcribe
  → merge into labelled transcript
```

**Implementation notes:**
- pyannote requires accepting terms on HuggingFace and using an auth token. Store the token as an environment secret, never in code.
- pyannote 3.1 needs a HuggingFace PRO account or the model access request approved. Check current access requirements at runtime — this may have changed.
- If pyannote is unavailable or slow, implement a fallback: simple voice activity detection (VAD) using `silero-vad` to split on silences, then transcribe each chunk without speaker labels. Label all chunks as "SPEAKER_0" and "SPEAKER_1" based on audio characteristics (pitch-based heuristic for gender detection as a rough proxy).
- Whisper `language=None` enables auto language detection. This is slower than specifying `hi` or `en` but necessary for code-mix. Accept the latency.
- For segments under 2 seconds, skip transcription (likely filler sounds or silence).
- Merge short adjacent segments from the same speaker before transcribing (reduces API calls and improves context).

**Output format:**
```python
[
    {"speaker": "DOCTOR", "start": 0.0, "end": 5.2, "text": "BP kaisa hai aaj?"},
    {"speaker": "PATIENT", "start": 5.8, "end": 9.1, "text": "Thoda high aa raha hai doctor"},
    ...
]
```

**Drug name post-correction:**
After transcription, run each DOCTOR turn through a fuzzy matcher against the curated Indian drug names dictionary. Use `rapidfuzz` (faster than `fuzzywuzzy`). If a word matches a drug name with score > 85, replace it and tag as `[auto-corrected]`. Log all corrections for debugging.

### 8.3 LangGraph agent

**What it does:** Takes the labelled transcript and produces a structured SOAP note via a three-pass agentic pipeline.

**State schema:**
```python
class ScribeState(TypedDict):
    transcript: list[dict]          # raw labelled transcript
    doctor_turns: str               # concatenated doctor speech only
    patient_turns: str              # concatenated patient speech only
    entities: dict                  # {symptoms, vitals, drugs, tests, duration}
    icd_codes: list[dict]           # [{code, description, confidence}]
    drug_corrections: list[dict]    # [{original, corrected, flagged}]
    soap_note: dict                 # {subjective, objective, assessment, plan}
    flags: list[str]                # items needing doctor verification
```

**Tool 1 — Medical NER:**
- Use scispaCy `en_core_sci_sm` for English medical entity extraction
- For Hindi phrases, use a keyword lookup against a curated Hindi medical phrase dictionary
  - "pet mein dard" → abdominal pain
  - "sir dard" → headache  
  - "bukhaar" → fever
  - "khasi" → cough
  - Maintain this as a simple CSV: `hindi_phrase, english_medical_term, soap_section`
- Extract: symptoms, duration, vitals (use regex for BP/pulse/temp/SpO2/weight patterns), drug names, test names, allergies

**Tool 2 — ICD-10 lookup:**
- Map extracted diagnosis terms to ICD-10 codes using `simple_icd_10`
- Search by description, return top 3 candidates with confidence
- If no match found, flag for manual entry

**Tool 3 — SOAP generator (LLM call):**
- Pass full transcript + extracted entities + ICD candidates to GPT-4o
- Use the system prompt from Section 4
- Request structured JSON output with keys: `subjective`, `objective`, `assessment`, `plan`, `flags`
- Parse JSON response. If parsing fails, fall back to asking the model for plain text and parse manually.

**Agent flow:**
```
START → extract_entities → lookup_icd → generate_soap → END
```

This is a linear graph in V1, not a loop. Add loops/conditionals in V2 when adding self-correction.

### 8.4 API layer (FastAPI)

**Endpoints:**

```
POST /transcribe
  Input:  multipart audio file
  Output: {transcript: [...], processing_time_seconds: float}

POST /generate
  Input:  {transcript: [...]}
  Output: {soap_note: {...}, flags: [...], icd_codes: [...]}

GET  /health
  Output: {status: "ok"}
```

**Notes:**
- Both endpoints are potentially slow (Whisper on CPU can take 2-5x audio duration). Use async FastAPI and stream progress updates to the frontend using Server-Sent Events (SSE) or WebSocket if possible. Otherwise, use polling from the Streamlit frontend.
- Add a timeout of 300 seconds on both endpoints.
- Never log or persist audio bytes or transcript content server-side.
- Use `python-multipart` for file upload handling.

### 8.5 Streamlit frontend

**Layout:** Two-column layout after processing. Single column before.

**Before processing (landing state):**
```
[App title + one-line description]
[⚠️ Disclaimer: for demonstration only, no data stored]

[Tab 1: Record]     [Tab 2: Upload]     [Tab 3: Demo mode]
[🎙️ Start recording]    [Choose file]   [Select scenario ▼]

[Process consultation →]  (disabled until audio ready)
```

**After processing:**
```
[← Back]                              [📋 Copy SOAP]  [💾 Download]

Left panel (40%):                Right panel (60%):
─── Transcript ───               ─── SOAP Note ───
DOCTOR: BP kaisa hai...          Subjective:
PATIENT: Thoda high...           Patient presents with...

                                 Objective:
                                 BP: 140/90 mmHg...

                                 Assessment:
                                 1. Hypertension (I10)

                                 Plan:
                                 1. Tab Amlodipine 5mg OD × 30 days
                                 
                                 ⚠️ Please verify: [auto-corrected items]
```

**Demo mode (for recruiters):**
- Dropdown with 5 pre-built scenarios (Hypertension, Diabetes, Fever, Respiratory, Gastro)
- Selecting a scenario loads a pre-generated audio clip (or pre-generated transcript + SOAP, skipping audio processing entirely for speed)
- Show "Demo mode — using pre-generated data" banner clearly

**UI framework notes:**
- Use `st.session_state` for all stateful data between Streamlit reruns
- Use `st.spinner()` with descriptive messages during processing: "Separating speakers...", "Transcribing consultation...", "Generating SOAP note..."
- On mobile (HuggingFace Spaces is accessed on mobile by some users), single-column layout stacks correctly by default in Streamlit

---

## 9. Product design

### Core design principles

**Accuracy over completeness.** The SOAP note should flag uncertain items rather than confidently fill them in wrong. A doctor checking three flagged items is better than a doctor re-reading the entire note for errors.

**Doctor stays in control.** The app generates a draft. The doctor edits, approves, copies. Never auto-save, never auto-submit.

**Zero friction for the demo audience.** A recruiter or hiring manager must be able to understand what the product does within 30 seconds without speaking Hindi.

**Privacy by design.** No audio stored. No transcript stored. No patient names ever. All processing ephemeral.

### Information hierarchy

1. SOAP note — primary output, gets the most screen space
2. Transcript — secondary, for doctor to verify the transcription
3. Flags — tertiary, surface only when there are items to verify
4. ICD codes — shown as metadata on the Assessment section, collapsible

### Copy and download options

- **Copy SOAP** — copies formatted plain text to clipboard
- **Download .txt** — plain text SOAP note
- **Download .json** — structured JSON (for future EHR integration demos)

### Error states

| Error | What to show |
|-------|-------------|
| Audio too short (< 10 seconds) | "Recording seems too short. Please record at least 30 seconds of consultation." |
| No speech detected | "No speech detected in the audio. Please check your microphone." |
| Transcription confidence low | "Transcription confidence is low. Please review carefully before use." |
| LLM API error | "Unable to generate SOAP note. Please try again." (show raw transcript so work is not lost) |
| Drug name correction | "Some drug names were auto-corrected. Please verify: [list]" |

### Accessibility

- All buttons have descriptive labels (not just icons)
- Disclaimer is visible before any processing
- SOAP note output is selectable plain text (not an image or canvas)
- Keyboard navigable in Streamlit's default behaviour

---

## 10. Deployment and debugging

### Local development setup

```bash
# Clone repo
git clone https://github.com/ajaymauryabbn/hinglish-clinical-scribe
cd hinglish-clinical-scribe

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Environment variables (create .env file)
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...           # HuggingFace token for pyannote
ENVIRONMENT=development

# Run backend
uvicorn backend.main:app --reload --port 8000

# Run frontend (separate terminal)
streamlit run frontend/app.py
```

### Repository structure

```
hinglish-clinical-scribe/
├── frontend/
│   ├── app.py                  # Streamlit main app
│   ├── components/
│   │   ├── audio_input.py      # Record + upload UI
│   │   ├── transcript_view.py  # Transcript display
│   │   └── soap_view.py        # SOAP note display
│   └── demo_data/              # Pre-generated demo scenarios
│       ├── hypertension.json
│       ├── diabetes.json
│       └── ...
├── backend/
│   ├── main.py                 # FastAPI app + routes
│   ├── transcribe.py           # Whisper + pyannote pipeline
│   ├── agent/
│   │   ├── graph.py            # LangGraph definition
│   │   ├── tools.py            # NER, ICD-10, SOAP tools
│   │   ├── prompts.py          # All LLM prompt templates
│   │   └── state.py            # ScribeState TypedDict
│   └── utils/
│       ├── drug_dict.py        # Indian drug names + correction logic
│       ├── hindi_phrases.py    # Hindi → English medical phrase map
│       └── audio_utils.py      # Format conversion, VAD
├── data/
│   ├── synthetic/              # Generated transcripts + SOAP pairs
│   ├── drug_names.csv          # Indian drug brand + generic names
│   └── hindi_medical.csv       # Hindi phrase to English mapping
├── scripts/
│   └── generate_synthetic.py   # GPT-4o data generation script
├── tests/
│   ├── test_transcription.py
│   ├── test_agent.py
│   └── test_api.py
├── requirements.txt
├── packages.txt                # System packages for HuggingFace Spaces (e.g. ffmpeg)
├── README.md
└── projectspec.md              # This file
```

### HuggingFace Spaces deployment

1. Create a new Space: SDK = Streamlit, Hardware = CPU Basic (free)
2. Add secrets in Space settings:
   - `OPENAI_API_KEY`
   - `HF_TOKEN`
   - `BACKEND_URL` (your Render backend URL)
3. Add `packages.txt` with system dependencies:
   ```
   ffmpeg
   ```
4. The `requirements.txt` must include all Python dependencies
5. HuggingFace Spaces auto-deploys on push to the Space's git repo

**ZeroGPU for Whisper (optional upgrade):**
- Apply for ZeroGPU access in Space settings
- Decorate Whisper inference function with `@spaces.GPU`
- This gives access to shared A100 time — significantly faster than CPU inference
- ZeroGPU has a daily quota; CPU fallback should always be available

### Render backend deployment

1. Connect GitHub repo to Render
2. Set environment: Python, build command `pip install -r requirements.txt`, start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Add environment variables in Render dashboard
4. Use UptimeRobot (free) to ping `/health` every 5 minutes to prevent cold starts
5. Render free tier cold start is 30–60 seconds — acceptable for a portfolio demo, not for production

**When to upgrade Render:**
- Upgrade to $7/month Starter plan when you want to show the app to real users or interviewers without cold start embarrassment
- Upgrade to $25/month plan only if you get consistent traffic

### Debugging guide

**Problem: Whisper transcription is all in one language (not switching)**
- Check that `language=None` is set in Whisper call, not `language="hi"` or `language="en"`
- Check that diarization is producing short segments (< 15 seconds). Long segments reduce accuracy.

**Problem: pyannote not loading**
- Check HF_TOKEN is set and the model access has been approved on HuggingFace
- pyannote 3.1 requires accepting the model's terms — do this manually at `huggingface.co/pyannote/speaker-diarization-3.1`
- If pyannote fails, the app should fall back to the VAD-based segmentation silently

**Problem: SOAP note is in Hindi (not English)**
- The system prompt is not being passed correctly to the LLM
- Check that `prompts.py` is imported and the SYSTEM prompt is set in the LLM call, not just the USER message

**Problem: Drug names are wrong in SOAP note**
- The drug correction step runs before SOAP generation. Check that corrected transcript is being passed to the agent, not the raw transcript.
- Increase fuzzy match threshold if false positives (wrong corrections) occur — try 90 instead of 85

**Problem: HuggingFace Space runs out of memory**
- Whisper large-v3 uses ~3GB RAM. On the free tier (16GB), this should be fine, but if other models are loaded simultaneously it can fail.
- Load models lazily (on first request, not at startup) and cache with `@st.cache_resource`
- Consider using Whisper medium (1.5GB) as default and offering large-v3 as "high accuracy" mode

**Problem: Audio format not recognized**
- Ensure `ffmpeg` is in `packages.txt` for HuggingFace Spaces
- Use `pydub` with explicit format parameter: `AudioSegment.from_file(path, format="mp4")` rather than relying on auto-detection

---

## 11. Safety, privacy and disclaimers

### What must be in the UI (non-negotiable)

Before any processing, show this disclaimer prominently:

> "This tool is a demonstration prototype for AI-assisted clinical documentation. It is not a medical device. All generated notes must be reviewed and verified by a qualified medical professional before use in any clinical context. No audio or transcript data is stored or transmitted beyond this session."

### Data handling rules

- Never write audio files or transcripts to disk on the server
- Never log transcript content in application logs (log request metadata only: timestamp, duration, endpoint)
- Never store any data that could identify a patient
- Process audio in memory only, delete after processing
- All API calls to OpenAI should not include patient-identifiable information (use synthetic data in demos; in a real deployment, add a pre-processing step to redact names before sending to the LLM)

### DPDP Act compliance notes (India)

The Digital Personal Data Protection Act 2023 requires explicit consent for processing personal data. For V1 (demo):
- The disclaimer above constitutes awareness, not consent for real patient data
- Never process real patient data in V1
- Add a "I confirm this is synthetic/demonstration data" checkbox before processing in V1

### Clinical safety

- All drug names extracted must be shown with a [VERIFY] tag in V1
- The SOAP note must never auto-submit to any system
- Add a watermark "DRAFT — PENDING PHYSICIAN REVIEW" to all generated notes

---

## 12. Future versions

### V2 — Accuracy and reliability (1–3 months after V1)

- **Real-time streaming transcription** using Whisper + WebSocket (instead of upload and process)
- **Self-correcting agent** — add a verification pass where the LLM checks its own SOAP note against the transcript for factual consistency
- **Regional language support** — add Tamil and Telugu input (doctors in Bengaluru, Chennai, Hyderabad have different linguistic patterns). Use IndicWhisper for South Indian language ASR.
- **Structured prescription output** — parse the Plan section into a machine-readable prescription format: `{drug, dose, frequency, duration, route}`
- **Better drug NER** — fine-tune a small NER model on the synthetic data to replace regex/fuzzy matching

### V3 — Product features (3–6 months)

- **HealthPlix / Practo MCP server** — build an MCP server that pushes the generated SOAP note directly into these EMRs. This is the integration that makes it sellable.
- **Doctor style learning** — after 20+ consultations, the system learns the individual doctor's prescription style and terminology preferences
- **Template library** — pre-built SOAP templates by specialty (GP, Cardiology, Diabetes, Paediatrics) that the agent uses as structure reference
- **ICD-10 + billing codes** — add IPC (Indian Pharmacy Coding) and suggest billing codes alongside ICD-10 for insurance claim preparation
- **Offline mode** — use Whisper.cpp and a quantized LLM (Llama-3 8B via Ollama) for clinics with poor internet connectivity

### V4 — Go-to-market (6–12 months, if there is traction)

- **Clinic dashboard** — aggregate stats across all consultations: most common diagnoses, avg consultation time, prescription patterns
- **Multi-doctor clinic support** — admin account + multiple doctor profiles
- **ABDM integration** — connect to Ayushman Bharat Digital Mission for federated patient records (this is India's national health stack)
- **Mobile app** — React Native wrapper around the web app for recording on a smartphone in the clinic room
- **Pricing model** — ₹1,500–₹3,000/month per doctor seat for clinic version; ₹10,000–₹25,000/month for a small hospital OPD

---

## 13. Open questions and decisions log

These are things that need a decision during build. Document the decision made here.

| Question | Options | Decision | Date |
|----------|---------|----------|------|
| Whisper large-v3 vs medium — accuracy vs speed | large-v3 (more accurate, slower), medium (faster, less accurate for Hinglish) | Default to medium; offer large-v3 in "high accuracy" mode | TBD |
| LLM for SOAP generation — GPT-4o vs Claude Sonnet | GPT-4o (better at following complex JSON instructions), Claude (cheaper) | Start with GPT-4o; add Claude as fallback if API quota runs out | TBD |
| Audio storage — process in memory vs temp file | Memory (more private), Temp file (easier for long audio) | Process in memory for <= 10MB; temp file for larger | TBD |
| Demo mode — pre-recorded audio vs pre-generated output | Pre-recorded audio (more authentic demo), Pre-generated output (faster, no audio processing) | Pre-generated output for demo mode to avoid cold-start delays | TBD |
| pyannote token requirement — public token vs per-user | Single shared token (simpler), Per-user (more privacy) | Single shared token in V1 (demo only, no real patient data) | TBD |
| Frontend — Streamlit vs Gradio | Streamlit (more flexible UI), Gradio (simpler, HuggingFace native) | Streamlit — more control over two-panel layout | Decided |
| Backend — Render vs Railway | Render (simpler), Railway (better free tier limits) | Render for V1; revisit if cold starts are a demo problem | TBD |

---

## Appendix — Hindi medical phrase dictionary (starter set)

Build this as `data/hindi_medical.csv`:

```csv
hindi_phrase,english_medical_term,soap_section
sir dard,headache,subjective
pet mein dard,abdominal pain,subjective
chest mein dard,chest pain,subjective
khasi,cough,subjective
bukhaar,fever,subjective
sans lene mein takleef,difficulty breathing / dyspnoea,subjective
ulti,vomiting,subjective
dast,diarrhoea,subjective
kamzori,weakness / fatigue,subjective
neend nahi aana,insomnia,subjective
bhookh nahi lagti,loss of appetite,subjective
thakaan,fatigue,subjective
ghutne mein dard,knee pain,subjective
kamar mein dard,lower back pain,subjective
pair mein sujan,pedal oedema,objective
pet mein gas,flatulence / bloating,subjective
aankhon mein jalan,eye irritation,subjective
sar ghoomna,dizziness / vertigo,subjective
muh sookha rehta hai,dry mouth / xerostomia,subjective
peshab mein jalan,dysuria,subjective
```

Expand this to 150+ phrases during development using MTSamples and the synthetic data generation output.

---

*Last updated: May 2026*  
*Spec version: 1.0*  
*Next review: after V1 is deployed*
