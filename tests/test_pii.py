import unittest
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agent.tools import redact_pii

class TestPIIRedaction(unittest.TestCase):
    def test_redact_phone(self):
        text = "My phone number is 9876543210."
        expected = "My phone number is [PHONE]."
        self.assertEqual(redact_pii(text), expected)

    def test_redact_email(self):
        text = "Contact me at patient@example.com for details."
        expected = "Contact me at [EMAIL] for details."
        self.assertEqual(redact_pii(text), expected)

    def test_redact_name_pattern_en(self):
        text = "Hello, my name is John Doe and I have a fever."
        expected = "Hello, my name is [NAME] and I have a fever."
        self.assertEqual(redact_pii(text), expected)

    def test_redact_name_pattern_hi(self):
        text = "Mera naam Rajesh Kumar hai."
        expected = "Mera naam [NAME] hai."
        self.assertEqual(redact_pii(text), expected)

    def test_redact_mixed(self):
        text = "I am Amit Sharma, reach me at 9123456789 or amit@test.com."
        # Note: "I am Amit Sharma" -> "I am [NAME]"
        # "9123456789" -> "[PHONE]"
        # "amit@test.com" -> "[EMAIL]"
        result = redact_pii(text)
        self.assertIn("[NAME]", result)
        self.assertIn("[PHONE]", result)
        self.assertIn("[EMAIL]", result)

if __name__ == "__main__":
    unittest.main()
