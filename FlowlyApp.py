import sys
import threading
import socket
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, QObject
import speech_recognition as sr
import pyaudio
import pyttsx3

class SocketClient:
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))

    def send_message(self, message):
        self.client_socket.send(message.encode('utf-8'))

    def receive_message(self):
        return self.client_socket.recv(1024).decode('utf-8')
    
    def close(self):
        self.client_socket.close()

class TaskManagerApp(QWidget):
    response_received = pyqtSignal(str)
    task_text_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        
        self.client = SocketClient('127.0.0.1', 4320)
        self.recognizer = sr.Recognizer()
        
        self.setWindowTitle("Flowly")
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()

        self.text_field = QLineEdit(self)
        self.text_field.setPlaceholderText("Please enter task description...")

        self.sendTask_btn = QPushButton("Send Task", self)
        self.sendTask_btn.clicked.connect(self.send_task)

        self.record_btn = QPushButton("Record", self)
        self.record_btn.clicked.connect(self.record_task)

        self.response_label = QLabel("Server response will be displayed here.", self)

        layout.addWidget(self.text_field)
        layout.addWidget(self.sendTask_btn)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.response_label)

        self.setLayout(layout)

        # Connect signals
        self.response_received.connect(self.response_label.setText)
        self.task_text_updated.connect(self.text_field.setText)

        # Start server listener thread
        self.listen_thread = threading.Thread(target=self.listen_for_server_messages, daemon=True)
        self.listen_thread.start()

    def send_task(self):
        task = self.text_field.text()
        if task:
            self.client.send_message(task)
            self.text_field.clear()

    def record_task(self):
        # Disable button to prevent multiple clicks
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Recording...")
        # Start recording in a background thread
        threading.Thread(target=self._record_task_thread, daemon=True).start()

    def _record_task_thread(self):
        try:
            with sr.Microphone() as mic:
                self.recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = self.recognizer.listen(mic)
                text = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")
                self.task_text_updated.emit(text)
        except sr.UnknownValueError:
            print("Could not understand audio.")
            self.response_received.emit("Could not understand audio. Please try again.")
        except Exception as e:
            print(f"Error: {e}")
            self.response_received.emit("Recording failed. Please try again.")
        finally:
            # Re-enable the button
            self.record_btn.setEnabled(True)
            self.record_btn.setText("Record")

    def listen_for_server_messages(self):
        while True:
            try:
                message = self.client.receive_message()
                if message:
                    self.response_received.emit(message)
            except ConnectionAbortedError:
                break
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

    def closeEvent(self, event):
        self.client.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TaskManagerApp()
    window.show()
    sys.exit(app.exec_())