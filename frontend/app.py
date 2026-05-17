import streamlit as st
import os
import json
import requests
from dotenv import load_dotenv

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

# Tabs for input
tab1, tab2, tab3 = st.tabs(["🎙️ Record", "📁 Upload", "📂 Demo Mode"])

with tab1:
    audio_file = st.audio_input("Record consultation")
    if audio_file:
        if st.button("Process Recording"):
            with st.spinner("Transcribing and generating SOAP note..."):
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                files = {"file": ("recording.wav", audio_file, "audio/wav")}
                try:
                    # 1. Transcribe
                    t_res = requests.post(f"{backend_url}/transcribe", files=files, timeout=300)
                    if t_res.status_code == 200:
                        transcript = t_res.json()["transcript"]
                        st.session_state.transcript = transcript
                        
                        # 2. Generate SOAP
                        g_res = requests.post(f"{backend_url}/generate", json={"transcript": transcript}, timeout=60)
                        if g_res.status_code == 200:
                            result = g_res.json()
                            st.session_state.soap_note = result["soap_note"]
                            st.session_state.flags = result.get("flags", [])
                            st.session_state.icd_codes = result.get("icd_codes", [])
                            st.session_state.prescriptions = result.get("prescriptions", [])
                            st.session_state.processed = True
                        else:
                            st.error("Failed to generate SOAP note.")
                    else:
                        st.error(f"Transcription failed: {t_res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

with tab2:
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file and st.button("Process Uploaded File"):
        with st.spinner("Transcribing and generating SOAP note..."):
            backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                # 1. Transcribe
                t_res = requests.post(f"{backend_url}/transcribe", files=files, timeout=300)
                if t_res.status_code == 200:
                    transcript = t_res.json()["transcript"]
                    st.session_state.transcript = transcript
                    
                    # 2. Generate SOAP
                    g_res = requests.post(f"{backend_url}/generate", json={"transcript": transcript}, timeout=60)
                    if g_res.status_code == 200:
                        result = g_res.json()
                        st.session_state.soap_note = result["soap_note"]
                        st.session_state.flags = result.get("flags", [])
                        st.session_state.icd_codes = result.get("icd_codes", [])
                        st.session_state.prescriptions = result.get("prescriptions", [])
                        st.session_state.processed = True

                    else:
                        st.error("Failed to generate SOAP note.")

                else:
                    st.error(f"Transcription failed: {t_res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

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
                
                # Call backend to process this transcript through the agent
                with st.spinner("Agent is analyzing transcript..."):
                    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    try:
                        response = requests.post(f"{backend_url}/generate", json={"transcript": transcript}, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.soap_note = result["soap_note"]
                            st.session_state.flags = result.get("flags", [])
                            st.session_state.icd_codes = result.get("icd_codes", [])
                        else:
                            st.error(f"Backend error ({response.status_code}). Running agent locally...")
                            raise Exception("Backend error")
                    except Exception as e:
                        # Fallback: Run agent directly in Streamlit
                        from backend.agent.graph import scribe_agent
                        result = scribe_agent.invoke({"transcript": transcript, "doctor_turns": "", "patient_turns": "", "entities": {}, "icd_codes": [], "drug_corrections": [], "soap_note": {}, "flags": []})
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
            
            st.button("📋 Copy SOAP Note")
            st.download_button("💾 Download .txt", "\n".join([f"{k.upper()}: {v}" for k,v in soap.items()]), file_name="soap_note.txt")
