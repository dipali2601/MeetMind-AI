"""
Speech-to-Text Module for MeetMind AI

This module converts audio files to text using Azure Cognitive Services Speech-to-Text SDK.
It reads Azure Speech credentials from environment variables and provides a simple interface
for transcribing audio files.
"""

import base64
import io
import os
import re
import subprocess
import tempfile
import urllib.parse
import wave
from typing import Optional

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover - optional dependency in local runs
    sr = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency in local runs
    pass

try:
    from pydub import AudioSegment
except ImportError:  # pragma: no cover - optional dependency in local runs
    AudioSegment = None

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - optional dependency in local runs
    imageio_ffmpeg = None

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - optional dependency in local runs
    genai = None
    types = None

from config import GEMINI_API_KEY, GEMINI_MODEL


def format_speaker_transcript(text: str) -> str:
    """Restore conservative sentence punctuation and speaker line boundaries."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return ""

    normalized = re.sub(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+"
        r"(?:said|mentioned|explained|added|asked)\s+",
        r"\n\1: ",
        normalized,
    )
    normalized = re.sub(
        r"\s+(?=([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?):\s)",
        "\n",
        normalized,
    )
    normalized = re.sub(r"(:\s+)([a-z])", lambda match: f"{match.group(1)}{match.group(2).upper()}", normalized)
    normalized = normalized.replace(" .", ".").replace(" ,", ",")
    normalized = normalized.replace(" ?", "?").replace(" !", "!")

    # Split before full names first, then protect their internal space while
    # handling single-name speakers so a surname is never split onto a line.
    normalized = re.sub(
        r"\s+(?=(?:[A-Z][a-z]+\s+[A-Z][a-z]+)\s+"
        r"(?:will|shall|said|mentioned|handle|take|fix|update|prepare|review|test|complete)\b)",
        ".\n",
        normalized,
    )
    normalized = re.sub(
        r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)(?=\s+"
        r"(?:will|shall|said|mentioned|handle|take|fix|update|prepare|review|test|complete)\b)",
        lambda match: f"{match.group(1)}\x00{match.group(2)}",
        normalized,
    )
    normalized = re.sub(
        r"\s+(?=(?:[A-Z][a-z]+)\s+"
        r"(?:will|shall|said|mentioned|handle|take|fix|update|prepare|review|test|complete)\b)",
        ".\n",
        normalized,
    )
    normalized = normalized.replace("\x00", " ")
    # These discourse markers start a new sentence in meeting transcripts.
    normalized = re.sub(
        r"\s+(?=(?:finally|once|today|tomorrow|thank you)\b)",
        ".\n",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b(Good morning everyone)\b", r"\1,", normalized, flags=re.IGNORECASE)

    lines = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line[0].upper() + line[1:]
        if line[-1].isalnum():
            line += "."
        lines.append(line)
    normalized = "\n".join(lines)
    normalized = re.sub(r"([.!?])\s+", r"\1\n", normalized)
    return "\n".join(line.strip() for line in normalized.splitlines() if line.strip())


def data_url_to_audio_bytes(data_url: str) -> bytes:
    """Convert a browser data URL into raw audio bytes."""
    if not data_url or not isinstance(data_url, str):
        raise ValueError("No audio recording was captured.")

    if not data_url.startswith("data:"):
        raise ValueError("The recording is in an unsupported format.")

    header, _, encoded = data_url.partition(",")
    if not encoded:
        raise ValueError("The recording payload is empty.")

    if ";base64" in header:
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("The recording payload is not valid base64 data.") from exc

    return urllib.parse.unquote_to_bytes(encoded)


def _to_wav_bytes(audio_bytes: bytes) -> bytes:
    """Normalize common audio containers to PCM WAV using FFmpeg."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as source:
            frames = source.readframes(source.getnframes())
            output = io.BytesIO()
            with wave.open(output, "wb") as target:
                target.setnchannels(source.getnchannels())
                target.setsampwidth(source.getsampwidth())
                target.setframerate(source.getframerate())
                target.writeframes(frames)
            return output.getvalue()
    except (wave.Error, EOFError):
        if AudioSegment is None and imageio_ffmpeg is None:
            raise ValueError("Install the project requirements to enable M4A/MP3 conversion.")
        try:
            if imageio_ffmpeg is not None:
                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                result = subprocess.run(
                    [
                        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
                        "-acodec", "pcm_s16le", "pipe:1",
                    ],
                    input=audio_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                return result.stdout

            converted = AudioSegment.from_file(io.BytesIO(audio_bytes))
            output = io.BytesIO()
            converted.export(output, format="wav")
            return output.getvalue()
        except Exception as exc:
            raise ValueError(
                "The audio format could not be decoded. Make sure the file is not corrupted "
                "and that the bundled FFmpeg dependency is installed."
            ) from exc


def _transcribe_with_gemini(wav_bytes: bytes) -> str:
    """Transcribe normalized audio through the configured Gemini model."""
    if not GEMINI_API_KEY or not GEMINI_MODEL or genai is None or types is None:
        raise RuntimeError("Gemini transcription is not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
            (
                "Transcribe this recording exactly. Add full stops, commas, question marks, and exclamation "
                "marks from context. When a speaker name is spoken, start a new line with 'Name: ' and keep "
                "that person's related speech on the same line until the next speaker name. Do not invent names. "
                "Return only the final transcript."
            ),
        ],
    )
    transcript = format_speaker_transcript(response.text or "")
    if not transcript:
        raise RuntimeError("Gemini returned an empty transcript.")
    return transcript


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe common audio formats after normalizing them to WAV."""
    if not audio_bytes:
        raise ValueError("The recording is empty. Please try again.")

    wav_bytes = _to_wav_bytes(audio_bytes)
    gemini_error = None
    try:
        return _transcribe_with_gemini(wav_bytes)
    except Exception as exc:
        gemini_error = str(exc)

    if sr is None:
        raise RuntimeError(f"Gemini transcription failed: {gemini_error}")

    try:
        with io.BytesIO(wav_bytes) as buffer:
            with wave.open(buffer, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()

        audio_data = sr.AudioData(frames, sample_rate, sample_width)
        recognizer = sr.Recognizer()
        return format_speaker_transcript(recognizer.recognize_google(audio_data))
    except (wave.Error, EOFError) as exc:
        raise ValueError("The recording could not be decoded as valid WAV audio.") from exc
    except sr.UnknownValueError as exc:
        raise RuntimeError(
            f"Gemini transcription failed: {gemini_error}. "
            "The Google fallback could not understand the recording."
        ) from exc
    except sr.RequestError as exc:
        raise RuntimeError(
            f"Gemini transcription failed: {gemini_error}. "
            "The Google fallback could not connect."
        ) from exc


def transcribe_audio_data_url(data_url: str) -> str:
    """Convert a browser-generated audio data URL into transcript text."""
    audio_bytes = data_url_to_audio_bytes(data_url)
    return transcribe_audio_bytes(audio_bytes)


def transcribe_audio_file(audio_file_path: str) -> str:
    """Transcribe an uploaded audio/video file."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    wav_bytes = _to_wav_bytes(audio_bytes)

    azure_error = None
    try:
        api_key, region = get_speech_credentials()
    except ValueError:
        api_key = region = ""

    if api_key and region:
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                wav_file.write(wav_bytes)
                wav_path = wav_file.name
            return format_speaker_transcript(transcribe_audio(wav_path))
        except (RuntimeError, ValueError) as exc:
            azure_error = str(exc)
        finally:
            if wav_path:
                os.unlink(wav_path)

    try:
        return transcribe_audio_bytes(wav_bytes)
    except RuntimeError as fallback_error:
        if not api_key or not region:
            raise RuntimeError(
                f"Audio was converted, but transcription failed: {fallback_error}"
            ) from fallback_error
        if azure_error:
            raise RuntimeError(f"Azure Speech failed: {azure_error}") from fallback_error
        raise


def get_speech_credentials() -> tuple[str, str]:
    """
    Retrieve Azure Speech credentials from environment variables.

    Returns:
        tuple: (api_key, region) for Azure Speech service

    Raises:
        ValueError: If required environment variables are not set
    """
    api_key = (os.getenv("AZURE_SPEECH_KEY") or "").strip().strip('"').strip("'")
    region = (os.getenv("AZURE_SPEECH_REGION") or "").strip().strip('"').strip("'")

    if not api_key or not region or api_key.lower() in {
        "your_azure_speech_key",
        "your-speech-key",
        "replace_me",
        "changeme",
    }:
        raise ValueError(
            "Missing Azure Speech credentials. Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION "
            "environment variables."
        )

    return api_key, region


def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe an audio file to text using Azure Speech-to-Text service.

    Args:
        audio_file_path (str): Path to the audio file (supports .wav, .mp3, .m4a, etc.)

    Returns:
        str: Transcribed text from the audio file

    Raises:
        FileNotFoundError: If the audio file does not exist
        ValueError: If Azure credentials are missing or invalid
        RuntimeError: If transcription fails
    """
    # Validate audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Get Azure credentials
    try:
        api_key, region = get_speech_credentials()
    except ValueError as e:
        raise ValueError(f"Credential error: {str(e)}")

    # Import Azure Speech SDK
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise RuntimeError(
            "Azure Speech SDK not found. Please install it using: pip install azure-cognitiveservices-speech"
        )

    # Create speech config
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_recognition_language = "en-US"

    # Create audio config from file
    try:
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to configure audio input: {str(e)}")

    # Create speech recognizer
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Perform transcription
    try:
        result = recognizer.recognize_once()
    except Exception as e:
        raise RuntimeError(f"Transcription service error: {str(e)}")

    # Handle recognition results
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        transcript = result.text
        if not transcript.strip():
            raise RuntimeError("Audio file was recognized but contains no recognizable speech")
        return transcript

    elif result.reason == speechsdk.ResultReason.NoMatch:
        raise RuntimeError(
            "No speech could be recognized. Please check the audio quality and language settings."
        )

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        raise RuntimeError(
            f"Transcription canceled. Reason: {cancellation.reason}. "
            f"Error details: {cancellation.error_details}"
        )

    else:
        raise RuntimeError(f"Unexpected recognition result: {result.reason}")


def transcribe_audio_with_details(audio_file_path: str) -> dict:
    """
    Transcribe an audio file and return transcript along with metadata.

    Args:
        audio_file_path (str): Path to the audio file

    Returns:
        dict: Dictionary containing:
            - 'transcript' (str): Transcribed text
            - 'file_path' (str): Input audio file path
            - 'status' (str): Status of transcription ('success' or 'error')
            - 'error_message' (str, optional): Error message if transcription failed

    Example:
        result = transcribe_audio_with_details('meeting.wav')
        if result['status'] == 'success':
            print(result['transcript'])
        else:
            print(result['error_message'])
    """
    try:
        transcript = transcribe_audio(audio_file_path)
        return {
            "transcript": transcript,
            "file_path": audio_file_path,
            "status": "success",
        }
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return {
            "transcript": "",
            "file_path": audio_file_path,
            "status": "error",
            "error_message": str(e),
        }


def main(audio_file_path: str) -> Optional[str]:
    """
    Simple main function to transcribe an audio file.

    This is the entry point for converting audio to text. The returned transcript
    can be directly passed to the text_analysis.py pipeline.

    Args:
        audio_file_path (str): Path to the audio file to transcribe

    Returns:
        str: Transcribed text, or None if transcription failed

    Example:
        transcript = main("path/to/meeting_recording.wav")
        if transcript:
            print("Transcription successful!")
            print(transcript)
        else:
            print("Transcription failed!")
    """
    try:
        transcript = transcribe_audio(audio_file_path)
        print(f"✓ Transcription completed successfully")
        return transcript
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"✗ Transcription failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python speech_to_text.py <audio_file_path>")
        print("\nExample: python speech_to_text.py meeting_recording.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = main(audio_path)

    if result:
        print("\n--- Transcription Result ---")
        print(result)
