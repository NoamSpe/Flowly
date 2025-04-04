import sys
import threading
import socket
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem, QInputDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, Qt
import speech_recognition as sr
import pyaudio
import pyttsx3
import json

class SocketClient:
    def __init__(self, host, port):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        self.user = 'test'

    def send_message(self, message):
        self.client_socket.send(json.dumps(message).encode('utf-8'))

    def receive_message(self):
        return json.loads(self.client_socket.recv(4096).decode('utf-8'))
    
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
        self.setGeometry(300, 300, 600, 400)
        self.initUI()
        self.showLogin()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.text_field = QLineEdit(self)
        self.text_field.setPlaceholderText("Please enter task description...")
        self.sendTask_btn = QPushButton("Send Task", self)
        self.sendTask_btn.clicked.connect(self.send_task)
        self.record_btn = QPushButton("Record", self)
        self.record_btn.clicked.connect(self.record_task)
        self.response_label = QLabel("Server response will be displayed here.", self)
        self.task_list = QListWidget(self)
        self.layout.addWidget(self.text_field)
        self.layout.addWidget(self.sendTask_btn)
        self.layout.addWidget(self.record_btn)
        self.layout.addWidget(self.response_label)
        self.layout.addWidget(self.task_list)
        self.setLayout(self.layout)
        self.response_received.connect(self.response_label.setText)
        self.task_text_updated.connect(self.text_field.setText)
        self.listen_thread = threading.Thread(target=self.listen_for_server_messages, daemon=True)
        self.listen_thread.start()

    def showLogin(self):
        username, ok = QInputDialog.getText(self, "Login", "Username:")
        print(username)
        if ok:
            self.client.send_message({'action': 'login', 'user': username})
            response = self.client.receive_message()
            if response.get('status') == 'success':
                self.client.user = username
                self.get_tasks()
            # else:
            #     QMessageBox.critical(self, "Login Failed", "Invalid username.")
            #     self.showLogin()

    def get_tasks(self):
        print(self.client.user)
        self.client.send_message({'action': 'get_tasks', 'user': self.client.user})
        print('sent tasks request')
        response = self.client.receive_message()
        print('response:', response)
        if 'tasks' in response:
            self.task_list.clear()
            for task_id, desc, date, time, category, urgency, status in response['tasks']:
                item = QListWidgetItem(f"{desc} - {date} {time} - {status}")
                item.setData(Qt.UserRole, task_id)
                self.task_list.addItem(item)
                
    def send_task(self):
        task = self.text_field.text()
        if task:
            self.client.send_message({'action': 'add_task', 'task_desc': task, 'user': self.client.user})
            self.text_field.clear()
            self.get_tasks()

    def record_task(self):
        self.record_btn.setEnabled(False)
        self.record_btn.setText("Recording...")
        threading.Thread(target=self._record_task_thread, daemon=True).start()

    def _record_task_thread(self):
        try:
            with sr.Microphone() as mic:
                self.recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = self.recognizer.listen(mic)
                text = self.recognizer.recognize_google(audio).lower()
                self.task_text_updated.emit(text)
        except sr.UnknownValueError:
            self.response_received.emit("Could not understand audio.")
        except Exception as e:
            self.response_received.emit("Recording failed.")
        finally:
            self.record_btn.setEnabled(True)
            self.record_btn.setText("Record")

    def listen_for_server_messages(self):
        while True:
            try:
                message = self.client.receive_message()
                if message.get('status'):
                    self.get_tasks()
                if message.get('error'):
                    self.response_received.emit(message.get('error'))
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