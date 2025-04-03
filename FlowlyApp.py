import sys
import threading
import socket
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel


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
    def __init__(self):
        super().__init__()
        
        self.client = SocketClient('127.0.0.1', 4320)  # Connect to the server
        
        self.setWindowTitle("Task Manager")
        self.setGeometry(300, 300, 400, 300)

        # Create layout
        layout = QVBoxLayout()

        # Text field for user input
        self.text_field = QLineEdit(self)
        self.text_field.setPlaceholderText("Enter task...")

        # Button to send input to server
        self.send_button = QPushButton("Send Task", self)
        self.send_button.clicked.connect(self.send_task)

        # Label to display responses
        self.response_label = QLabel("Server response will be displayed here.", self)

        # Add widgets to layout
        layout.addWidget(self.text_field)
        layout.addWidget(self.send_button)
        layout.addWidget(self.response_label)

        # Set layout for the window
        self.setLayout(layout)

        # Start a background thread to listen for messages from the server
        self.listen_thread = threading.Thread(target=self.listen_for_server_messages, daemon=True)
        self.listen_thread.start()

    def send_task(self):
        task = self.text_field.text()
        if task:
            print(task)
            self.client.send_message(task)
            self.text_field.clear()

    def listen_for_server_messages(self):
        while True:
            message = self.client.receive_message()
            if message:
                # Update the GUI with the message from the server
                self.response_label.setText(message)
                
    def closeEvent(self, event):
        self.client.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TaskManagerApp()
    window.show()
    sys.exit(app.exec_())
