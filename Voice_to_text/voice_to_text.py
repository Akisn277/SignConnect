import speech_recognition as sr

def voicetotxt():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)

        try:
            audio = recognizer.listen(source, timeout=5)
            print("Processing...")

            text = recognizer.recognize_google(audio)
            return text

        except sr.WaitTimeoutError:
            return "No speech detected"

        except sr.UnknownValueError:
            return "Could not understand audio"

        except sr.RequestError:
            return "API unavailable"