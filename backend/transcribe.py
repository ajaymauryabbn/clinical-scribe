import os
import whisper
from pyannote.audio import Pipeline
import torch
import time
from typing import List, Dict

class TranscriptionPipeline:
    def __init__(self, model_name="medium", hf_token=None):
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper model
        print(f"Loading Whisper model: {model_name} on {self.device}...")
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        
        # Load pyannote pipeline for diarization
        if hf_token:
            print("Loading pyannote diarization pipeline...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if self.diarization_pipeline:
                self.diarization_pipeline.to(torch.device(self.device))
        else:
            print("HF_TOKEN not provided. Diarization will be skipped.")
            self.diarization_pipeline = None

    def process_audio(self, audio_path: str) -> List[Dict]:
        """
        Diarize and transcribe audio.
        """
        # 1. Diarization
        segments = []
        if self.diarization_pipeline:
            print("Diarizing audio...")
            diarization = self.diarization_pipeline(audio_path)
            
            # Group segments by speaker to avoid too many small fragments
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
        else:
            # Fallback if no diarization: just one segment
            # Use whisper's internal VAD or just one chunk
            segments = [{"start": 0, "end": None, "speaker": "SPEAKER_00"}]
        
        # 2. Transcription
        print(f"Transcribing {len(segments)} segments...")
        final_transcript = []
        
        for seg in segments:
            # Extract segment audio (Whisper handles timing if passed, 
            # but for accuracy we can crop or just transcribe whole and use segment times)
            # Simplest for V1: Transcribe the whole thing and match segments
            pass
            
        # Refined V1 approach: Transcribe the whole file with timestamps
        result = self.whisper_model.transcribe(audio_path, language=None)
        
        # Merge whisper segments with diarization speakers
        # This is a simple heuristic: match whisper segment middle time to diarization segment
        for w_seg in result["segments"]:
            mid_time = (w_seg["start"] + w_seg["end"]) / 2
            speaker = "UNKNOWN"
            for d_seg in segments:
                if d_seg["end"] is None or (d_seg["start"] <= mid_time <= d_seg["end"]):
                    speaker = d_seg["speaker"]
                    break
            
            final_transcript.append({
                "speaker": speaker,
                "text": w_seg["text"].strip(),
                "start": w_seg["start"],
                "end": w_seg["end"]
            })
            
        return final_transcript

# Singleton instance
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        hf_token = os.getenv("HF_TOKEN")
        _pipeline = TranscriptionPipeline(hf_token=hf_token)
    return _pipeline
