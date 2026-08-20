from pathlib import Path
import tempfile
import streamlit as st

from backend.ai_service import analyze_meeting
from backend.speech_to_text import transcribe_audio_bytes, transcribe_audio_file

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MeetMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

    :root {
        --bg: #0b0e14;
        --panel: #12161f;
        --panel-soft: #161b26;
        --line: rgba(201, 178, 138, 0.16);
        --line-soft: rgba(255, 255, 255, 0.06);
        --text: #e9e7e1;
        --muted: #9a9a9a;
        --gold: #c9a962;
        --gold-soft: rgba(201, 169, 98, 0.12);
        --ink: #7d8ba1;
        --success: #6fae8f;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(201, 169, 98, 0.05), transparent),
            linear-gradient(180deg, #08090d 0%, #0b0e14 45%, #0d1117 100%);
        color: var(--text);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        max-width: 980px;
    }

    /* ---------- Hero ---------- */
    .hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.6rem;
        margin-bottom: 2.2rem;
    }

    .kicker {
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--gold);
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .main-title {
        font-family: 'Fraunces', serif;
        font-size: 46px;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #f5f3ee;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }

    .tagline {
        color: var(--muted);
        font-size: 15.5px;
        max-width: 620px;
        line-height: 1.6;
        font-weight: 400;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        font-family: 'Fraunces', serif;
        font-size: 21px;
        font-weight: 600;
        color: #f0eee8;
        margin: 2.4rem 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-title .num {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        color: var(--gold);
        letter-spacing: 0.1em;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 3px 9px;
        font-weight: 600;
    }

    /* ---------- Panels / cards ---------- */
    .panel {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    .info-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-left: 2px solid var(--gold);
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.6rem;
        color: #ddd9cf;
        line-height: 1.65;
        font-size: 14.5px;
    }

    .summary-card {
        background: var(--panel);
        border: 1px solid var(--line-soft);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        line-height: 1.75;
        font-size: 15px;
        color: #ddd9cf;
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: linear-gradient(160deg, var(--gold-soft), transparent 70%);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        height: 100%;
    }

    .metric-label {
        font-size: 10.5px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 600;
    }

    .metric-value {
        font-family: 'Fraunces', serif;
        font-size: 34px;
        font-weight: 600;
        margin-top: 0.35rem;
        color: #f5f3ee;
    }

    /* ---------- Topic pills ---------- */
    .pill {
        display: inline-block;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        background: var(--gold-soft);
        border: 1px solid var(--line);
        color: #e6d9b8;
        margin: 0.2rem 0.4rem 0.2rem 0;
        font-size: 12.5px;
        font-weight: 500;
    }

    /* ---------- Inputs / buttons ---------- */
    .stTextArea textarea {
        background: var(--panel) !important;
        border: 1px solid var(--line-soft) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        font-size: 14.5px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
    }

    .stButton > button[kind="primary"],
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #c9a962, #b8934f) !important;
        color: #14110a !important;
        border: none !important;
    }

    div[data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed var(--line);
        border-radius: 12px;
        padding: 0.4rem;
    }

    div[data-testid="stFileUploader"] button,
    div[data-testid="stFileUploader"] button span {
        color: #14110a !important;
    }

    .stSelectbox > div > div {
        background: var(--panel) !important;
        border-radius: 10px !important;
        border: 1px solid var(--line-soft) !important;
    }

    /* ---------- Table ---------- */
    .stTable, .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--line-soft);
    }

    hr {
        border-color: var(--line-soft) !important;
    }

    .toolbar-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.9rem;
        color: var(--muted);
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_sample_transcript(name: str) -> str:
    sample_path = Path(__file__).resolve().parent / "sample_data" / f"{name}.txt"
    return sample_path.read_text(encoding="utf-8") if sample_path.exists() else ""


def render_voice_recorder() -> str:
    return """
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.5rem; flex-wrap:wrap;">
        <button id="voice-recorder-button" type="button" style="width:100%; cursor:pointer; border-radius:10px; background:linear-gradient(135deg,#c9a962,#b8934f); color:#14110a; border:none; font-weight:600; padding:0.7rem 1rem; font-size:0.95rem;">🎙 Record voice</button>
        <div id="voice-recording-indicator" style="display:none; color:#ef4444; font-weight:600; font-size:0.85rem;">
            🔴 Recording... <span id="voice-timer">00:00</span>
        </div>
    </div>
    <div id="voice-preview-wrap" style="display:none; margin-top:0.75rem;">
        <audio id="voice-preview" controls style="width:100%;"></audio>
    </div>
    <script type="module">
        import { Streamlit } from "https://cdn.jsdelivr.net/npm/streamlit-component-lib@2.0.0/dist/index.js";

        Streamlit.setComponentReady();

        const button = document.getElementById("voice-recorder-button");
        const indicator = document.getElementById("voice-recording-indicator");
        const timerText = document.getElementById("voice-timer");
        const previewWrap = document.getElementById("voice-preview-wrap");
        const preview = document.getElementById("voice-preview");

        let mediaRecorder = null;
        let stream = null;
        let chunks = [];
        let isRecording = false;
        let timerInterval = null;
        let recordingStartMs = 0;

        function formatTimer(totalSeconds) {
            const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
            const seconds = (totalSeconds % 60).toString().padStart(2, "0");
            return `${minutes}:${seconds}`;
        }

        function resetTimer() {
            if (timerInterval) clearInterval(timerInterval);
            timerInterval = null;
            recordingStartMs = 0;
            timerText.textContent = "00:00";
        }

        function startTimer() {
            resetTimer();
            recordingStartMs = Date.now();
            timerInterval = setInterval(() => {
                const elapsedSeconds = Math.floor((Date.now() - recordingStartMs) / 1000);
                timerText.textContent = formatTimer(elapsedSeconds);
            }, 1000);
        }

        function setButtonLabel(label) {
            button.textContent = label;
        }

        function sendError(message) {
            try {
                Streamlit.setComponentValue({ error: message });
            } catch (error) {
                console.error("Streamlit component error:", error);
            }
        }

        function audioBufferToWav(audioBuffer) {
            const numChannels = audioBuffer.numberOfChannels;
            const sampleRate = audioBuffer.sampleRate;
            const bitDepth = 16;
            const bytesPerSample = bitDepth / 8;
            const blockAlign = numChannels * bytesPerSample;
            const dataLength = audioBuffer.length * blockAlign;
            const buffer = new ArrayBuffer(44 + dataLength);
            const view = new DataView(buffer);
            const channels = [];

            for (let i = 0; i < numChannels; i++) {
                channels.push(audioBuffer.getChannelData(i));
            }

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            writeString(view, 0, "RIFF");
            view.setUint32(4, 36 + dataLength, true);
            writeString(view, 8, "WAVE");
            writeString(view, 12, "fmt ");
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitDepth, true);
            writeString(view, 36, "data");
            view.setUint32(40, dataLength, true);

            let offset = 44;
            for (let sampleIndex = 0; sampleIndex < audioBuffer.length; sampleIndex++) {
                for (let channelIndex = 0; channelIndex < numChannels; channelIndex++) {
                    const sample = Math.max(-1, Math.min(1, channels[channelIndex][sampleIndex]));
                    const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                    view.setInt16(offset, intSample, true);
                    offset += 2;
                }
            }

            return buffer;
        }

        async function blobToWav(blob) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const buffer = await blob.arrayBuffer();
            const decoded = await audioContext.decodeAudioData(buffer.slice(0));
            const wavBuffer = audioBufferToWav(decoded);
            return new Blob([wavBuffer], { type: "audio/wav" });
        }

        button.addEventListener("click", async () => {
            if (isRecording) {
                try {
                    mediaRecorder.stop();
                } catch (error) {
                    sendError("Recording could not be stopped. Please try again.");
                }
                return;
            }

            if (!window.MediaRecorder || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                sendError("This browser does not support audio recording.");
                return;
            }

            try {
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                chunks = [];
                mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data && event.data.size > 0) {
                        chunks.push(event.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    if (stream) {
                        stream.getTracks().forEach((track) => track.stop());
                    }
                    if (timerInterval) clearInterval(timerInterval);
                    timerInterval = null;

                    if (!chunks.length) {
                        previewWrap.style.display = "none";
                        preview.src = "";
                        indicator.style.display = "none";
                        isRecording = false;
                        setButtonLabel("🎙 Record voice");
                        timerText.textContent = "00:00";
                        sendError("No audio was captured. Please try again.");
                        return;
                    }

                    try {
                        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
                        const wavBlob = await blobToWav(blob);
                        const wavBuffer = await wavBlob.arrayBuffer();
                        const bytes = new Uint8Array(wavBuffer);
                        let binary = "";
                        bytes.forEach((byte) => {
                            binary += String.fromCharCode(byte);
                        });
                        const base64 = btoa(binary);
                        const dataUrl = "data:audio/wav;base64," + base64;
                        preview.src = dataUrl;
                        previewWrap.style.display = "block";
                        indicator.style.display = "none";
                        isRecording = false;
                        setButtonLabel("🎙 Record voice");
                        timerText.textContent = "00:00";
                        try {
                            Streamlit.setComponentValue({ audio_data: dataUrl, filename: "recording.wav" });
                        } catch (error) {
                            console.error("Set component value failed:", error);
                        }
                    } catch (error) {
                        console.error(error);
                        previewWrap.style.display = "none";
                        preview.src = "";
                        indicator.style.display = "none";
                        isRecording = false;
                        setButtonLabel("🎙 Record voice");
                        timerText.textContent = "00:00";
                        sendError("Recording failed while preparing the audio preview. Please try again.");
                    }
                };

                mediaRecorder.start(250);
                isRecording = true;
                indicator.style.display = "inline-block";
                setButtonLabel("⏹ Stop recording");
                startTimer();
                Streamlit.setComponentValue({ status: "recording" });
            } catch (error) {
                console.error(error);
                let message = "Microphone access is unavailable. Please allow microphone permission and try again.";
                if (error && error.name === "NotAllowedError") {
                    message = "Microphone permission was denied. Please allow access to record your voice.";
                } else if (error && error.name === "NotFoundError") {
                    message = "No microphone was found on this device.";
                }
                sendError(message);
            }
        });

        window.addEventListener("beforeunload", () => {
            if (stream) {
                stream.getTracks().forEach((track) => track.stop());
            }
        });
    </script>
    """


# ---------------- HERO ----------------
st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="kicker">Meeting Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">MeetMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Turn conversations into clear summaries, decisions, '
    'owners, deadlines, and practical next steps — in seconds.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TRANSCRIPT INPUT ----------------
st.markdown('<div class="section-title"><span class="num">01</span> Meeting transcript</div>', unsafe_allow_html=True)

with st.container():
    top_col1, top_col2, top_col3 = st.columns([2, 1.2, 1])
    with top_col1:
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=[
                "wav", "mp3", "m4a", "aac", "ogg", "opus", "flac", "wma",
                "aiff", "aif", "caf", "webm", "mp4",
            ],
            label_visibility="collapsed",
        )
    with top_col2:
        sample_options = ["team_meeting", "project_meeting", "hackathon_meeting"]
        selected_sample = st.selectbox("Sample transcript", sample_options, label_visibility="collapsed")
    with top_col3:
        if st.button("Load sample", use_container_width=True):
            st.session_state["manual_text"] = load_sample_transcript(selected_sample)

    rec_col, _ = st.columns([1, 3])
    with rec_col:
        recorded_audio = st.audio_input("Record voice", label_visibility="collapsed")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()
            recording_id = hash(audio_bytes)
            if st.session_state.get("last_recording_id") != recording_id:
                try:
                    transcript_text = transcribe_audio_bytes(audio_bytes)
                    st.session_state["manual_text"] = transcript_text
                    st.session_state["meeting_transcript"] = transcript_text
                    st.session_state["last_recording_id"] = recording_id
                    st.success("Voice recording transcribed into the meeting transcript.")
                except Exception as exc:  # pragma: no cover - UI error handling
                    st.error(f"Could not transcribe the recording: {exc}")

    st.session_state.setdefault("manual_text", "")
    if uploaded_file is not None:
        upload_id = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("last_upload_id") != upload_id:
            file_suffix = Path(uploaded_file.name).suffix.lower()
            try:
                supported_audio = {
                    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac",
                    ".wma", ".aiff", ".aif", ".caf", ".webm", ".mp4",
                }
                if file_suffix not in supported_audio:
                    raise ValueError("Unsupported file type. Please choose a supported audio file.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    upload_path = temp_file.name
                meeting_text = transcribe_audio_file(upload_path)

                st.session_state["manual_text"] = meeting_text
                st.session_state["meeting_transcript"] = meeting_text
                st.session_state["last_upload_id"] = upload_id
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(
                    f"Could not process '{uploaded_file.name}'. "
                    "Upload a readable audio file such as WAV, MP3, M4A, AAC, OGG, FLAC, or WEBM. "
                    f"Details: {exc}"
                )
            finally:
                if "upload_path" in locals():
                    Path(upload_path).unlink(missing_ok=True)
        meeting_text = st.session_state.get("manual_text", "")
    else:
        meeting_text = st.session_state.get("manual_text", "")

    st.session_state.setdefault("meeting_transcript", meeting_text)
    meeting_text = st.text_area(
        "Paste a meeting transcript",
        key="meeting_transcript",
        height=240,
        placeholder="Example: We reviewed product progress, assigned follow-ups, and confirmed the final delivery deadline...",
        label_visibility="collapsed",
    )

    analyze = st.button("✨  Analyze meeting", use_container_width=False, type="primary")

# ---------------- ANALYSIS ----------------
if analyze:
    if meeting_text.strip():
        st.session_state["analysis"] = analyze_meeting(meeting_text)
    else:
        st.warning("Please provide a meeting transcript before analyzing.")

analysis = st.session_state.get("analysis")
if analysis:

    st.markdown('<div class="section-title"><span class="num">02</span> Meeting overview</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Words</div>'
            f'<div class="metric-value">{analysis["word_count"]}</div></div>',
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Decisions</div>'
            f'<div class="metric-value">{len(analysis["decisions"])}</div></div>',
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Actions</div>'
            f'<div class="metric-value">{len(analysis["action_items"])}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title"><span class="num">03</span> Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-card">{analysis["summary"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span class="num">04</span> Key decisions</div>', unsafe_allow_html=True)
    if analysis["decisions"]:
        decisions_html = "".join(
            f"<div class='info-card'>{decision}</div>" for decision in analysis["decisions"]
        )
        st.markdown(decisions_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-card">No decisions were detected in this transcript.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span class="num">05</span> Action items</div>', unsafe_allow_html=True)
    if analysis["action_items"]:
        action_df = {
            "Task": [item[0] for item in analysis["action_items"]],
            "Owner": [item[1] for item in analysis["action_items"]],
            "Deadline": [item[2] for item in analysis["action_items"]],
        }
        st.table(action_df)
    else:
        st.markdown('<div class="info-card">No action items were detected in this transcript.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span class="num">06</span> Key topics</div>', unsafe_allow_html=True)
    topic_html = "".join(f"<span class='pill'>{topic}</span>" for topic in analysis["topics"])
    st.markdown(f'<div class="panel">{topic_html}</div>', unsafe_allow_html=True)

        with st.expander("Transcript preview"):
            st.write(meeting_text)
    else:
        st.warning("Please provide a meeting transcript before analyzing.")
        