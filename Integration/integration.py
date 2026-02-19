from Voice_to_text.voice_to_text import voicetotxt
from Sign_to_text.sign_to_text import detect_sign

def get_combined_output():
    voice = voicetotxt()
    sign = detect_sign()

    return {
        "voice_text": voice,
        "sign_text": sign
    }