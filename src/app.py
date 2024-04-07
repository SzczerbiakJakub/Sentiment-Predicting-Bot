import sys
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QScrollArea
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from bot import Chatbot
import audio


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("My PyQt App")
        self.setGeometry(0, 0, 400, 200)
        self.build_app()
        self.chatbot = self.get_chatbot()
        self.audio_handler = self.get_audio_handler()
        

    def build_right_layout(self):
        right_layout = QVBoxLayout()
        self.chat_widget = ChatWidget(self)
        self.message_line_edit_widget = MessageLineEdit(self)
        right_layout.addWidget(self.chat_widget)
        right_layout.addWidget(self.message_line_edit_widget)
        return right_layout
    
    def build_left_layout(self):
        left_layout = QVBoxLayout()
        self.microphone_button = MicrophoneButton(self)
        left_layout.addWidget(self.microphone_button)
        return left_layout

    def build_app(self):
        self.main_widget = QWidget()
        self.main_layout =QHBoxLayout()
        self.layout3 = QHBoxLayout()

        self.right_layout = self.build_right_layout()
        self.left_layout = self.build_left_layout()
        
        left_widget, right_widget = QWidget(), QWidget()
        left_widget.setLayout(self.left_layout)
        right_widget.setLayout(self.right_layout)

        #layout2.setLayout
        self.main_layout.addWidget(left_widget)
        self.main_layout.addWidget(right_widget)
        self.main_widget.setLayout(self.main_layout)
        self.main_widget.setGeometry(0, 0, 100, 100)
        #widget.show()

        self.setCentralWidget(self.main_widget)

    def get_chatbot(self):
        return Chatbot()
    
    def get_audio_handler(self):
        return audio.AudioHandler(self)



class AppLayout(QHBoxLayout):

    def __init__(self):
        super().__init__()


class MessageLineEdit(QLineEdit):

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.returnPressed.connect(self.on_enter_pressed)

    def on_enter_pressed(self):
        #text = self.line_edit.text()
        user_text = self.text()
        print(user_text)
        self.app.chat_widget.add_user_message(user_text)
        bot_response = self.app.chatbot.predict_sentiment(user_text)
        self.app.chat_widget.add_bot_message(bot_response)


class MicrophoneButton(QPushButton):

    def __init__(self, app):
        super().__init__("Unmute")
        self.app = app
        self.muted = True
        self.setGeometry(0, 0, 50, 50)
        self.clicked.connect(self.unmute)

    def unmute(self):
        self.setText("Mute")
        self.muted = False
        self.clicked.disconnect(self.unmute)
        self.clicked.connect(self.mute)
        self.interact_with_audio()

    def mute(self):
        self.setText("Unmute")
        self.muted = True
        self.clicked.disconnect(self.mute)
        self.clicked.connect(self.unmute)

    def interact_with_audio(self):
        response = self.app.audio_handler.recognize_audio()
        self.app.chat_widget.add_user_message(response)
        self.mute()
        self.app.audio_handler.respond(response)
        self.app.chat_widget.add_bot_message(response)


class ChatWidget(QWidget):
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.chat_layout = QVBoxLayout()
        self.chat_screen = ChatScreen()
        self.chat_layout.addWidget(self.chat_screen.scroll_area)
        self.setLayout(self.chat_layout)
        
    def add_user_message(self, message):
        self.chat_screen.add_user_message(message)

    def add_bot_message(self, message):
        self.chat_screen.add_bot_message(message)
        self.app.audio_handler.respond(message)

    
class ChatScreen(QWidget):

    def __init__(self):
        super().__init__()
        self.scroll_layout = QVBoxLayout()
        self.scroll_area = self.build_chat_scroll_area()
        self.build_chat_widget()

    def build_chat_scroll_area(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("background-color: white;")
        return scroll_area

    def build_chat_widget(self):
        self.scroll_area.setWidget(self)
        self.setLayout(self.scroll_layout)

    def add_user_message(self, message):
        label1 = QLabel(message)
        label1.setWordWrap(True)
        label1.setFixedWidth(500)
        label1.setMaximumWidth(400)
        font = QFont("Arial", 30)
        label1.setFont(font)
        label1.setStyleSheet("color: white; background-color: darkblue; border-radius: 10px;")
        #label1.setFixedHeight(50)
        ChatScreen.adjust_text_height(label1)
        self.scroll_layout.addWidget(label1, alignment=Qt.AlignRight)

    def add_bot_message(self, message):
        label2 = QLabel(message)
        label2.setWordWrap(True)
        label2.setFixedWidth(500)
        label2.setMaximumWidth(400)
        font = QFont("Arial", 30)
        label2.setFont(font)
        label2.setStyleSheet("color: white; background-color: darkgrey; border-radius: 10px;")
        #label2.setFixedHeight(50)
        ChatScreen.adjust_text_height(label2)
        self.scroll_layout.addWidget(label2)

    @staticmethod
    def adjust_text_height(label):
        if label.sizeHint().width() < label.contentsRect().width():
            ...
        else:
            ...



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())