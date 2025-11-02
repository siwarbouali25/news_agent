# tts.py (Windows offline TTS using SAPI5 via pyttsx3)
import os, tempfile

def tts_pyttsx3(text: str, voice_substr: str | None = None, rate: int = 175, volume: float = 1.0) -> bytes:
    import pyttsx3
    engine = pyttsx3.init()  # uses SAPI5 on Windows
    engine.setProperty("rate", rate)       # typical range ~125–200
    engine.setProperty("volume", volume)   # 0.0–1.0

    if voice_substr:
        for v in engine.getProperty("voices"):
            name = (v.name or "").lower()
            if voice_substr.lower() in name:
                engine.setProperty("voice", v.id)
                break

    # save to wav file then return bytes
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    engine.save_to_file(text, path)
    engine.runAndWait()
    with open(path, "rb") as f:
        audio = f.read()
    os.remove(path)
    return audio

def synthesize_tts(text: str, voice_substr: str | None = None, rate: int = 175) -> tuple[bytes, str]:
    """Returns (audio_bytes, mime). Offline WAV."""
    data = tts_pyttsx3(text, voice_substr=voice_substr, rate=rate)
    return data, "audio/wav"
