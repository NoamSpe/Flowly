import sys
import threading
import socket
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem, QInputDialog, QMessageBox
from PyQt5.QtWidgets import QMenu, QDialog, QComboBox, QDialogButtonBox
from PyQt5.QtCore import pyqtSignal, QObject, Qt
import speech_recognition as sr
import pyaudio
import pyttsx3
import json

class SocketClient:
    def __init__(self, host, port):
        self.client_socket = None
        self.host = host
        self.port = port
        self.connect()
        # self.client_socket.connect((host, port))
        # self.user = 'test'
    
    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def send_message(self, message):
        try:
            self.client_socket.send(json.dumps(message).encode('utf-8'))
        except (BrokenPipeError, ConnectionResetError):
            self.connect()
            self.client_socket.send(json.dumps(message).encode('utf-8'))

    def receive_message(self):
        try:
            data = b''
            while True:
                chunk = self.client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                try:
                    return json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    continue
        except (ConnectionResetError, OSError):
            self.connect()
            return {'error': 'Connection lost, please try again'}
    
    def close(self):
        self.client_socket.close()

class FlowlyApp(QWidget):
    response_received = pyqtSignal(str)
    task_text_updated = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client = None
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
        self.refreshList_btn = QPushButton("Refresh", self)
        self.refreshList_btn.clicked.connect(self.get_tasks)
        self.response_label = QLabel("Server response will be displayed here.", self)
        self.task_list = QListWidget(self)
        self.task_list.itemChanged.connect(self.handle_task_checkbox)
        self.layout.addWidget(self.text_field)
        self.layout.addWidget(self.sendTask_btn)
        self.layout.addWidget(self.record_btn)
        self.layout.addWidget(self.refreshList_btn)
        self.layout.addWidget(self.response_label)
        self.layout.addWidget(self.task_list)
        self.setLayout(self.layout)
        self.response_received.connect(self.response_label.setText)
        self.task_text_updated.connect(self.text_field.setText)
        self.listen_thread = threading.Thread(target=self.listen_for_server_messages, daemon=True)
        self.listen_thread.start()
        self.hide()

    def showLogin(self):
        if self.client:
            self.client.close()

        dialog = QDialog(self)
        dialog.setWindowTitle("Login")
        layout = QVBoxLayout()

        self.username_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)

        login_btn = QPushButton("Login")
        signup_btn = QPushButton("Sign Up")

        layout.addWidget(QLabel("Username:"))
        layout.addWidget(self.username_edit)
        layout.addWidget(QLabel("Password:"))
        layout.addWidget(self.password_edit)
        layout.addWidget(login_btn)
        layout.addWidget(signup_btn)

        login_btn.clicked.connect(self.handle_login)
        signup_btn.clicked.connect(self.handle_signup)

        dialog.setLayout(layout)
        dialog.exec_()

    def handle_login(self):
        username = self.username_edit.text()
        password = self.password_edit.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password")
            return

        self.client = SocketClient('127.0.0.1', 4320)
        self.client.send_message({'action': 'login', 'user': username, 'password': password})
        response = self.client.receive_message()

        if response.get('status') == 'success':
            self.client.user = response['username']
            self.show()
            self.get_tasks()
            self.sender().parent().accept()

            self.listen_thread = threading.Thread(target=self.listen_for_server_messages, daemon=True)
            self.listen_thread.start()
        else:
            QMessageBox.critical(self, "Login Failed", response.get('message', 'Unknown error'))
            self.client = None

    def handle_signup(self):
        username = self.username_edit.text()
        password = self.password_edit.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Please enter both username and password")
            return

        email, ok = QInputDialog.getText(self, "Sign Up", "Email:")
        if not ok or not email:
            return

        self.client.send_message({'action': 'signup', 'username': username, 'email': email, 'password': password})
        response = self.client.receive_message()

        if response.get('status') == 'success':
            QMessageBox.information(self, "Success", "Account created! Please log in.")
        else:
            QMessageBox.critical(self, "Signup Failed", response.get('message', 'Unknown error'))

    def show_task_context_menu(self, pos):
        item = self.task_list.itemAt(pos)
        if not item:
            return

        menu = QMenu()
        edit_action = menu.addAction("Edit Task")
        delete_action = menu.addAction("Delete Task")

        action = menu.exec_(self.task_list.mapToGlobal(pos))

        if action == edit_action:
            self.edit_task(item)
        elif action == delete_action:
            self.delete_task(item)

    def get_tasks(self):
        if not self.client:
            return

        print(self.client.user)
        self.client.send_message({'action': 'get_tasks', 'user': self.client.user})
        print('sent tasks request')
        try:
            response = self.client.receive_message()
        except Exception as e:
            return
        print('response:', response)

        if 'tasks' in response:
            self.task_list.clear()
            for task in response['tasks']:
                task_id, desc, date, time, category, urgency, status = task
                item_text = str(desc)
                if date: item_text += f" due: {date}"
                if time: item_text += f" at {time}"

                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, task_id)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if status=='done' else Qt.Unchecked)
                self.task_list.addItem(item)

    def delete_task(self, item):
        confirm = QMessageBox.question(self,
                                       "Confirm Delete", "Are you sure you want to delete this task?",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            task_id = item.data(Qt.UserRole)
            self.client.send_message({'action': 'delete_task', 'task_id': task_id, 'user': self.client.user})
            self.get_tasks()

    def edit_task(self, item):
        task_id = item.data(Qt.UserRole)
        self.client.send_message({'action': 'get_task', 'task_id': task_id, 'user': self.client.user})
        response = self.client.receive_message()

        if response.get('status') != 'success':
            QMessageBox.critical(self, "Error", "Could not fetch task details")
            return

        task_data = response['task']
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Task")
        dialog.setFixedSize(400,300)
        layout = QVBoxLayout()

        form_layout = QVBoxLayout()

        # Task Description
        desc_edit = QLineEdit(task_data[1])
        form_layout.addWidget(QLabel("Description:"))
        form_layout.addWidget(desc_edit)

        # Date
        date_edit = QLineEdit(task_data[2] if task_data[2] != 'None' else '')
        form_layout.addWidget(QLabel("Date (YYYY-MM-DD):"))
        form_layout.addWidget(date_edit)

        # Time
        time_edit = QLineEdit(task_data[3] if task_data[3] != 'None' else '')
        form_layout.addWidget(QLabel("Time (HH:MM):"))
        form_layout.addWidget(time_edit)

        # Category
        category_edit = QLineEdit(task_data[4] if task_data[4] else '')
        form_layout.addWidget(QLabel("Category:"))
        form_layout.addWidget(category_edit)

        # Urgency
        urgency_combo = QComboBox()
        urgency_combo.addItems(["1", "2", "3", "4", "5"])
        urgency_combo.setCurrentText(str(task_data[5]))
        form_layout.addWidget(QLabel("Urgency (1-5):"))
        form_layout.addWidget(urgency_combo)

        # Status
        status_combo = QComboBox()
        status_combo.addItems(["pending", "done"])
        status_combo.setCurrentText(task_data[6])
        form_layout.addWidget(QLabel("Status:"))
        form_layout.addWidget(status_combo)

        # Dialog buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addLayout(form_layout)
        layout.addWidget(btn_box)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            update_data = {
                'TaskDesc': desc_edit.text(),
                'Date': date_edit.text() or None,
                'Time': time_edit.text() or None,
                'Category': category_edit.text(),
                'Urgency': int(urgency_combo.currentText()),
                'Status': status_combo.currentText()
            }
            self.client.send_message({
                'action': 'update_task',
                'task_id': task_id,
                'update_data': update_data,
                'user': self.client.user
            })
            self.get_tasks()
       
    def handle_task_checkbox(self, item):
        task_id = item.data(Qt.UserRole)
        new_status = 'done' if item.checkState() == Qt.Checked else 'pending'
        self.client.send_message({'action':'update_task_status', 'task_id':task_id, 'status':new_status, 'user':self.client.user})

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
        if self.client:
            self.client.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FlowlyApp()
    window.show()
    sys.exit(app.exec_())