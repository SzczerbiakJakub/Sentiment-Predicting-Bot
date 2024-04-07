import pyttsx3


class Voice:

    voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"

    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', Voice.voice_id)
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)



    def speak(self, sequence: str=""):

        self.engine.say(sequence)
        self.engine.runAndWait()





