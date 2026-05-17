import os
import tempfile
from openai import OpenAI
from typing import List, Dict
from pydub import AudioSegment

# Separate client for Whisper API (OpenAI) vs LLM (DeepSeek)
_whisper_client = None

def _get_whisper_client() -> OpenAI:
    global _whisper_client
    if _whisper_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Required for audio transcription.")
        _whisper_client = OpenAI(api_key=api_key)
    return _whisper_client


class TranscriptionPipeline:
    """
    API-based transcription using OpenAI Whisper API.
    Replaces local whisper + pyannote.audio + torch to stay within free-tier
    RAM (512 MB). Speaker diarization can be layered on later via pyannote
    cloud API once a paid Render plan is used.
    """

    MAX_CHUNK_BYTES = 24 * 1024 * 1024  # Whisper API hard limit: 25 MB per file

    def process_audio(self, audio_path: str) -> List[Dict]:
        client = _get_whisper_client()
        chunks = self._split_if_needed(audio_path)
        transcript: List[Dict] = []
        time_offset = 0.0

        for chunk_path in chunks:
            try:
                with open(chunk_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
                for seg in response.segments:
                    transcript.append({
                        "speaker": "SPEAKER_00",
                        "text": seg.text.strip(),
                        "start": round(seg.start + time_offset, 3),
                        "end": round(seg.end + time_offset, 3),
                    })
                if response.segments:
                    time_offset += response.segments[-1].end
            finally:
                if chunk_path != audio_path:
                    os.remove(chunk_path)

        return transcript

    def _split_if_needed(self, audio_path: str) -> List[str]:
        if os.path.getsize(audio_path) <= self.MAX_CHUNK_BYTES:
            return [audio_path]

        audio = AudioSegment.from_file(audio_path)
        chunk_ms = 10 * 60 * 1000  # 10-minute chunks
        chunks = []
        for start in range(0, len(audio), chunk_ms):
            chunk = audio[start:start + chunk_ms]
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            chunk.export(tmp.name, format="wav")
            chunks.append(tmp.name)
        return chunks


_pipeline = None

def get_pipeline() -> TranscriptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TranscriptionPipeline()
    return _pipeline
