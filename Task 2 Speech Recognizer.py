import speech_recognition as sr

def transcribe_audio(audio_file_path):
    """
    Transcribes speech from an audio file to text using Google's speech recognition API.
    :param audio_file_path: str, path to the .wav audio file
    :return: str, transcribed text or error message
    """
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except FileNotFoundError:
        return "Audio file not found. Please check the path."

# Example usage
if __name__ == "__main__":
    path_to_audio = "sample.wav"  # Replace with your actual audio file
    print("Transcribed Text:\n", transcribe_audio(path_to_audio))
