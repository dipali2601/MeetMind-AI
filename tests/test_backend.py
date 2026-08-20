import base64

from backend.ai_service import analyze_meeting
from backend.speech_to_text import data_url_to_audio_bytes, format_speaker_transcript


def test_analyze_meeting_returns_expected_sections():
    transcript = """
    Team update: We completed the project structure and finished the dashboard prototype.
    Priya will handle the design review on Friday, and Daniel will test the login flow before the launch.
    We decided to finalize the API integration and prepare the demo for next Monday.
    """

    result = analyze_meeting(transcript)

    assert "summary" in result
    assert "decisions" in result
    assert "action_items" in result
    assert "topics" in result
    assert isinstance(result["action_items"], list)
    assert len(result["action_items"]) >= 1


def test_data_url_to_audio_bytes_decodes_wav_payload():
    wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x00\x00\x00\x00\x00data\x00\x00\x00\x00"
    encoded = base64.b64encode(wav_bytes).decode("ascii")

    result = data_url_to_audio_bytes(f"data:audio/wav;base64,{encoded}")

    assert result == wav_bytes


def test_format_speaker_transcript_adds_boundaries_to_unpunctuated_text():
    transcript = (
        "good morning everyone let's start with a quick update on our project "
        "Rahul Sharma will take care of the login module and complete it by Friday 21st August "
        "Priya Das will handle the dashboard testing by Monday 24th August "
        "thank you"
    )

    assert format_speaker_transcript(transcript) == (
        "Good morning everyone, let's start with a quick update on our project.\n"
        "Rahul Sharma will take care of the login module and complete it by Friday 21st August.\n"
        "Priya Das will handle the dashboard testing by Monday 24th August.\n"
        "Thank you."
    )
