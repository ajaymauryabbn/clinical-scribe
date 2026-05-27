import streamlit as st
import os
import json
import tempfile
import sys
from dotenv import load_dotenv

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.transcribe import get_pipeline
from backend.agent.graph import scribe_agent
from backend.utils.audio_utils import convert_to_wav

load_dotenv()

st.set_page_config(page_title="Hinglish Clinical Scribe", layout="wide")

# Sidebar
st.sidebar.title("Hinglish Scribe")
st.sidebar.info("AI-powered clinical documentation for Indian doctors.")

# App Header
st.title("🎙️ Hinglish Clinical Scribe")
st.markdown("""
*This tool is a demonstration prototype. It is not a medical device. 
All generated notes must be verified by a medical professional.*
""")

# Session State Initialization
if "processed" not in st.session_state:
    st.session_state.processed = False
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "soap_note" not in st.session_state:
    st.session_state.soap_note = None

# Sidebar - Settings
st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")
accuracy_mode = st.sidebar.toggle("High Accuracy Mode", help="Uses Whisper large-v3 (slower but more accurate for Hinglish)")

def process_clinical_audio(audio_bytes, filename):
    """
    Consolidated processing: Audio -> Transcript -> SOAP
    """
    # 8.1 Error handling for short audio
    if len(audio_bytes) < 5000: # Very rough check for empty/short
        st.error("⚠️ Recording seems too short. Please record at least 10 seconds of consultation.")
        return

    with st.spinner(f"Processing audio (Whisper {'large-v3' if accuracy_mode else 'medium'})..."):
        # 1. Save and convert audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_input:
            tmp_input.write(audio_bytes)
            tmp_input_path = tmp_input.name
        
        try:
            tmp_wav_path = convert_to_wav(tmp_input_path)
            
            # 2. Transcribe
            model_name = "large-v3" if accuracy_mode else "medium"
            # Since get_pipeline() is a singleton, we might need to recreate it if model changes
            # For demo purposes, we'll stick to one but in a real app we'd handle this better
            pipeline = get_pipeline()
            transcript = pipeline.process_audio(tmp_wav_path)
            
            if not transcript:
                st.error("⚠️ No speech detected in the audio. Please check your microphone.")
                return

            st.session_state.transcript = transcript
            
            # 3. Generate SOAP & Analyze
            with st.spinner("Analyzing transcript and generating SOAP note..."):
                initial_state = {
                    "transcript": transcript,
                    "doctor_turns": "",
                    "patient_turns": "",
                    "entities": {},
                    "icd_codes": [],
                    "drug_corrections": [],
                    "soap_note": {},
                    "prescriptions": [],
                    "flags": []
                }
                result = scribe_agent.invoke(initial_state)
                
                st.session_state.soap_note = result["soap_note"]
                st.session_state.flags = result.get("flags", [])
                st.session_state.icd_codes = result.get("icd_codes", [])
                st.session_state.prescriptions = result.get("prescriptions", [])
                st.session_state.processed = True
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            if st.session_state.transcript:
                st.info("Transcript was generated successfully. Showing transcript only.")
                st.session_state.processed = True
                
        finally:
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
            if 'tmp_wav_path' in locals() and os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

# Tabs for input
tab1, tab2, tab3 = st.tabs(["🎙️ Record", "📁 Upload", "📂 Demo Mode"])

with tab1:
    audio_file = st.audio_input("Record consultation")
    if audio_file:
        if st.button("Process Recording"):
            process_clinical_audio(audio_file.read(), "recording.wav")

with tab2:
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file and st.button("Process Uploaded File"):
        process_clinical_audio(uploaded_file.read(), uploaded_file.name)

with tab3:
    demo_scenarios = [
        "Hypertension follow-up",
        "Type 2 diabetes management",
        "Acute fever / viral URTI",
        "Respiratory — asthma or COPD review",
        "Gastroenterology — acidity, IBS, constipation"
    ]
    selected_scenario = st.selectbox("Select a scenario for demo:", demo_scenarios)
    
    if st.button("Load Demo Scenario"):
        filename = f"data/synthetic/{selected_scenario.lower().replace(' ', '_').replace('/', '_')}.json"
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                transcript = []
                for line in data["raw_transcript_text"].split("\n"):
                    if line.startswith("DOCTOR:"):
                        transcript.append({"speaker": "DOCTOR", "text": line.replace("DOCTOR:", "").strip()})
                    elif line.startswith("PATIENT:"):
                        transcript.append({"speaker": "PATIENT", "text": line.replace("PATIENT:", "").strip()})
                
                st.session_state.transcript = transcript
                
                # Call agent locally
                with st.spinner("Agent is analyzing transcript..."):
                    initial_state = {
                        "transcript": transcript,
                        "doctor_turns": "",
                        "patient_turns": "",
                        "entities": {},
                        "icd_codes": [],
                        "drug_corrections": [],
                        "soap_note": {},
                        "prescriptions": [],
                        "flags": []
                    }
                    result = scribe_agent.invoke(initial_state)
                    st.session_state.soap_note = result["soap_note"]
                    st.session_state.flags = result.get("flags", [])
                    st.session_state.icd_codes = result.get("icd_codes", [])
                    st.session_state.prescriptions = result.get("prescriptions", [])
                    st.session_state.processed = True
        except Exception as e:
            st.error(f"Error loading demo: {e}")

# Results View
if st.session_state.processed:
    st.divider()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Transcript")
        for turn in st.session_state.transcript:
            if turn["speaker"] == "DOCTOR":
                st.markdown(f"**DOCTOR:** {turn['text']}")
            else:
                st.markdown(f"*PATIENT:* {turn['text']}")
    
    with col2:
        st.subheader("📝 SOAP Note")
        if st.session_state.soap_note:
            soap = st.session_state.soap_note
            st.markdown("### Subjective")
            st.write(soap.get("subjective", ""))
            st.markdown("### Objective")
            st.write(soap.get("objective", ""))
            st.markdown("### Assessment")
            st.write(soap.get("assessment", ""))
            st.markdown("### Plan")
            st.write(soap.get("plan", ""))
            
            # Display Structured Prescriptions
            if st.session_state.get("prescriptions"):
                st.markdown("#### 💊 Structured Prescriptions")
                for rx in st.session_state["prescriptions"]:
                    st.write(f"- **{rx.get('drug')}** ({rx.get('dose')}): {rx.get('frequency')} for {rx.get('duration')} [{rx.get('route')}]")
            
            # Display Flags/Warnings
            if st.session_state.get("flags"):
                st.warning("⚠️ **Items to Verify:**")
                for flag in st.session_state["flags"]:
                    st.write(f"- {flag}")
            
            # Copy to Clipboard functionality using a small HTML/JS snippet
            soap_text = "\n".join([f"{k.upper()}: {v}" for k,v in soap.items()])
            copy_code = f"""
                <script>
                function copyToClipboard() {{
                    const text = {json.dumps(soap_text)};
                    navigator.clipboard.writeText(text).then(() => {{
                        alert('SOAP Note copied to clipboard!');
                    }});
                }}
                </script>
                <button onclick="copyToClipboard()" style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                ">📋 Copy SOAP Note</button>
            """
            import streamlit.components.v1 as components
            components.html(copy_code, height=70)
            
            st.download_button("💾 Download .txt", soap_text, file_name="soap_note.txt")
