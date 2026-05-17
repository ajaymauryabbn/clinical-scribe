import os
from pydub import AudioSegment
import tempfile

def convert_to_wav(input_path: str, target_sr=16000) -> str:
    """
    Converts audio file to 16kHz mono WAV format.
    Returns the path to the temporary wav file.
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")
    return temp_wav.name

def save_upload_file(upload_file) -> str:
    """
    Saves an UploadFile to a temporary location and returns the path.
    """
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload_file.file.read())
            return tmp.name
    finally:
        upload_file.file.close()
