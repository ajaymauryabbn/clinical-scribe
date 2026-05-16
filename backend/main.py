from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Hinglish Clinical Scribe API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint to handle audio upload, diarization and transcription.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Placeholder for transcription logic
    return {
        "filename": file.filename,
        "transcript": [
            {"speaker": "DOCTOR", "text": "Sample doctor text", "start": 0.0, "end": 2.0},
            {"speaker": "PATIENT", "text": "Sample patient text", "start": 2.1, "end": 4.0}
        ],
        "processing_time": 0.5
    }

from backend.agent.graph import scribe_agent

@app.post("/generate")
async def generate_soap(data: dict):
    """
    Endpoint to generate SOAP note from transcript.
    """
    transcript = data.get("transcript")
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided")
    
    # Run the LangGraph agent
    try:
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
        
        return {
            "soap_note": result["soap_note"],
            "flags": result["flags"],
            "icd_codes": result["icd_codes"],
            "drug_corrections": result["drug_corrections"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
