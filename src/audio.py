import speech_recognition as sr
import pyaudio
import tts


class AudioHandler:

    language = "en-US"

    def __init__(self, app) -> None:
        self.app = app
        self.recognizer = AudioRecognizer(AudioHandler.language)
        self.speaker = AudioSpeaker(AudioHandler.language)


    def recognize_audio(self):
        input_audio = self.recognizer.recognize_input_audio()
        return input_audio

    def respond(self, response):
        self.speaker.respond(response)



class AudioSpeaker(tts.Voice):

    def __init__(self, language):
        super().__init__()
        self.language = language

    def respond(self, response):
        self.speak(sequence=response)


class AudioRecognizer(sr.Recognizer):
    
    def __init__(self, language) -> None:
        super().__init__()
        self.language = language


    def recognize_input_audio(self):
        while True:
            your_text = ""
            with sr.Microphone(device_index=1) as source:
                print("Speak something...")
                self.adjust_for_ambient_noise(source)
                audio = self.listen(source)

            try:
                your_text = self.recognize_google(audio, language=self.language)
                
            except sr.UnknownValueError:
                your_text = "Sorry, I could not understand your audio."
            except sr.RequestError as e:
                your_text = "I could not request results from Google Speech Recognition service; {0}".format(e)

            your_text.lower()
            print(your_text)
            return your_text