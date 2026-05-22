import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.transcribe import TranscriptionPipeline

class TestTranscriptionPipeline(unittest.TestCase):
    @patch("backend.transcribe.whisper.load_model")
    @patch("backend.transcribe.Pipeline.from_pretrained")
    @patch("backend.transcribe.AudioSegment.from_file")
    def test_process_audio_parallel(self, mock_audio_from_file, mock_pyannote, mock_whisper_load):
        # Mock Whisper model
        mock_whisper = MagicMock()
        mock_whisper.transcribe.side_effect = lambda path, language: {"text": f"Transcribed {path}"}
        mock_whisper_load.return_value = mock_whisper
        
        # Mock Pyannote pipeline
        mock_pipeline_obj = MagicMock()
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.0), None, "SPEAKER_01"),
            (MagicMock(start=1.0, end=2.0), None, "SPEAKER_02"),
            (MagicMock(start=2.0, end=3.0), None, "SPEAKER_01"),
        ]
        mock_pipeline_obj.return_value = mock_diarization
        mock_pipeline_obj.to.return_value = mock_pipeline_obj
        mock_pyannote.return_value = mock_pipeline_obj
        
        # Mock AudioSegment
        mock_audio = MagicMock()
        mock_audio.__getitem__.return_value = mock_audio
        mock_audio_from_file.return_value = mock_audio
        
        # Initialize pipeline
        pipeline = TranscriptionPipeline(hf_token="test_token")
        
        # Run process_audio
        with patch("backend.transcribe.tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "test_seg.wav"
            transcript = pipeline.process_audio("dummy.wav")
            
        # Verify results
        self.assertEqual(len(transcript), 3)
        self.assertEqual(transcript[0]["speaker"], "SPEAKER_01")
        self.assertEqual(transcript[1]["speaker"], "SPEAKER_02")
        self.assertEqual(transcript[2]["speaker"], "SPEAKER_01")
        
        # Verify whisper was called 3 times
        self.assertEqual(mock_whisper.transcribe.call_count, 3)

if __name__ == "__main__":
    unittest.main()
