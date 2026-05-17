import os
import whisper
from pyannote.audio import Pipeline
import torch
import time
import tempfile
from pydub import AudioSegment
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
            # Fallback if no diarization: just use Whisper's internal VAD segments
            print("Skipping diarization, using Whisper segments only.")
            result = self.whisper_model.transcribe(audio_path, language=None)
            for w_seg in result["segments"]:
                segments.append({
                    "start": w_seg["start"],
                    "end": w_seg["end"],
                    "speaker": "SPEAKER_00", # Default speaker
                    "text": w_seg["text"].strip()
                })
            return segments
        
        # 2. Transcription per segment
        print(f"Transcribing {len(segments)} segments...")
        final_transcript = []
        audio = AudioSegment.from_file(audio_path)
        
        for seg in segments:
            # Skip very short segments
            if seg["end"] - seg["start"] < 0.5:
                continue
                
            # Extract segment
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            seg_audio = audio[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_seg:
                seg_audio.export(tmp_seg.name, format="wav")
                
                # Transcribe segment with auto-language detection
                # This is slower but better for Hinglish mid-sentence switches
                res = self.whisper_model.transcribe(tmp_seg.name, language=None)
                text = res["text"].strip()
                
                if text:
                    final_transcript.append({
                        "speaker": seg["speaker"],
                        "text": text,
                        "start": seg["start"],
                        "end": seg["end"]
                    })
                
                os.remove(tmp_seg.name)
            
        return final_transcript

# Singleton instance
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        hf_token = os.getenv("HF_TOKEN")
        _pipeline = TranscriptionPipeline(hf_token=hf_token)
    return _pipeline
