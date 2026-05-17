from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import time
from dotenv import load_dotenv
from backend.transcribe import get_pipeline
from backend.utils.audio_utils import save_upload_file, convert_to_wav

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
    
    start_time = time.time()
    temp_input = None
    temp_wav = None
    
    try:
        # 1. Save uploaded file
        temp_input = save_upload_file(file)
        
        # 2. Convert to standard format
        temp_wav = convert_to_wav(temp_input)
        
        # 3. Process with pipeline
        pipeline = get_pipeline()
        transcript = pipeline.process_audio(temp_wav)
        
        processing_time = time.time() - start_time
        
        return {
            "filename": file.filename,
            "transcript": transcript,
            "processing_time": processing_time
        }
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup
        if temp_input and os.path.exists(temp_input):
            os.remove(temp_input)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

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
            "drug_corrections": result["drug_corrections"],
            "prescriptions": result.get("prescriptions", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
