# FlowlyApp.py
import sys
import threading
import socket
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QLabel, QListWidget, QListWidgetItem, QInputDialog,
                             QMessageBox, QMenu, QDialog, QComboBox, QDialogButtonBox,
                             QCheckBox, QSpacerItem, QSizePolicy, QFrame, QFormLayout)
# Make sure all QtCore imports are present
from PyQt5.QtCore import (pyqtSignal, QObject, Qt, QThread, pyqtSlot,
                          QMetaObject, Q_ARG, QPoint, QTimer) # Added QTimer
from PyQt5.QtGui import QFont, QIcon
import speech_recognition as sr
import pyaudio # Implicitly used by sr.Microphone
import pyttsx3
import json
import time
import datetime
import math

# --- NetworkWorker Class (Keep As Is) ---
# No changes needed here based on the current problem
class NetworkWorker(QObject):
    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    _request_finished = pyqtSignal() # Internal signal

    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.client_socket = None
        self.receive_buffer = b""
        self.is_connected = False
        self.current_user = None
        self.lock = threading.Lock()

    # --- connect_socket Method ---
    def connect_socket(self):
        with self.lock:
            if self.client_socket:
                try: self.client_socket.close()
                except Exception: pass
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.settimeout(10.0)
                self.client_socket.connect((self.host, self.port))
                self.is_connected = True
                self.receive_buffer = b""
                print("DEBUG: Socket connected")
                return True
            except (socket.error, OSError) as e:
                print(f"ERROR: Socket connection failed: {e}")
                self.client_socket = None; self.is_connected = False
                self.error_occurred.emit(f"Connection failed: {e}")
                return False

    # --- _send_request_internal Method ---
    def _send_request_internal(self, request_data):
        with self.lock:
            if not self.is_connected or self.client_socket is None:
                if not self.connect_socket():
                    self.error_occurred.emit("Failed to connect to server.")
                    self._request_finished.emit(); return False
            action = request_data.get('action')
            if self.current_user and 'user' not in request_data and action not in ['login', 'signup']:
                 request_data['user'] = self.current_user
            try:
                message = json.dumps(request_data) + '\n'
                self.client_socket.sendall(message.encode('utf-8'))
                print(f"DEBUG: Sent -> {request_data}")
                return True
            except (socket.error, OSError, BrokenPipeError, ConnectionResetError) as e:
                print(f"ERROR: Send failed: {e}. Attempting reconnect...")
                self.is_connected = False
                if self.connect_socket():
                    try:
                        message = json.dumps(request_data) + '\n'
                        self.client_socket.sendall(message.encode('utf-8'))
                        print(f"DEBUG: Sent (after retry) -> {request_data}")
                        return True
                    except (socket.error, OSError) as e2:
                         print(f"ERROR: Send failed on retry: {e2}")
                         self.error_occurred.emit(f"Failed to send message: {e2}")
                         self.is_connected = False; self._request_finished.emit(); return False
                else:
                    self.error_occurred.emit("Failed to send message and reconnect.")
                    self._request_finished.emit(); return False

    # --- _receive_response_internal Method ---
    def _receive_response_internal(self):
        with self.lock:
            if not self.is_connected or self.client_socket is None:
                self.error_occurred.emit("Not connected."); self._request_finished.emit(); return None
            start_time = time.time(); timeout = 15.0
            while True:
                if b'\n' in self.receive_buffer:
                    message_data, self.receive_buffer = self.receive_buffer.split(b'\n', 1)
                    message_str = message_data.decode('utf-8').strip()
                    if message_str:
                        try:
                            response = json.loads(message_str)
                            print(f"DEBUG: Received <- {response}"); return response
                        except json.JSONDecodeError as e:
                             print(f"ERROR: JSON Decode Error: '{e}' on data: '{message_str}'")
                             self.error_occurred.emit(f"Received invalid data."); continue
                try:
                    chunk = self.client_socket.recv(4096)
                    if not chunk:
                        print("ERROR: Server closed connection."); self.error_occurred.emit("Server disconnected.")
                        self.is_connected = False; self._request_finished.emit(); return None
                    self.receive_buffer += chunk
                except socket.timeout:
                     if not self.receive_buffer and time.time() - start_time > timeout:
                          self.error_occurred.emit("No response (timeout)."); self._request_finished.emit(); return None
                except (socket.error, OSError, ConnectionResetError) as e:
                     print(f"ERROR: Receive failed: {e}"); self.error_occurred.emit(f"Network error: {e}")
                     self.is_connected = False; self._request_finished.emit(); return None
                if time.time() - start_time > timeout:
                    print(f"ERROR: Overall receive timeout."); self.error_occurred.emit("Receive timeout.")
                    self._request_finished.emit(); return None

    # --- process_request Method ---
    @pyqtSlot(dict)
    def process_request(self, request_data):
        print(f"DEBUG: Worker received request: {request_data}")
        action = request_data.get('action') # Get action before modifying dict
        # Add user if known and not login/signup
        if self.current_user and 'user' not in request_data and action not in ['login', 'signup']:
             request_data['user'] = self.current_user

        response = None
        if self._send_request_internal(request_data):
            response = self._receive_response_internal()

        final_response = {}
        if isinstance(response, dict): final_response = response.copy()
        elif response is not None: final_response['raw_response'] = response

        if action: final_response['action_echo'] = action # Add echo

        if final_response: self.response_received.emit(final_response)
        # else: error already emitted

        self._request_finished.emit()

    # --- set_current_user Method ---
    @pyqtSlot(str)
    def set_current_user(self, username):
        print(f"DEBUG: Worker received set_current_user: '{username}'")
        self.current_user = username if username else None

    # --- close_connection Method ---
    def close_connection(self):
         with self.lock:
            if self.client_socket:
                print("DEBUG: Worker closing socket.")
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                    self.client_socket.close()
                except (socket.error, OSError) as e: print(f"WARN: Error closing socket: {e}")
                finally: self.client_socket = None; self.is_connected = False; self.current_user = None

# --- TaskItemWidget Class (Keep As Is) ---
# No changes needed here
class TaskItemWidget(QWidget):
    status_changed = pyqtSignal(int, bool) # task_id, is_done
    edit_requested = pyqtSignal(int)       # task_id
    delete_requested = pyqtSignal(int)     # task_id

    def __init__(self, task_id, desc, date, time, category, urgency, status, parent=None):
        super().__init__(parent)
        self.task_id = task_id
        self.is_done = (status == 'done')
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5)
        self.checkbox = QCheckBox(); self.checkbox.setChecked(self.is_done)
        self.checkbox.stateChanged.connect(self.on_checkbox_change)
        self.desc_label = QLabel(desc); self.desc_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        datetime_str = ""
        if date and date != 'None': datetime_str += f"{date} "
        if time and time != 'None': datetime_str += f"{time}"
        self.datetime_label = QLabel(datetime_str.strip()); self.datetime_label.setStyleSheet("color: gray; font-size: 9pt;")
        urgency_label = QLabel(f"Urg: {urgency}"); urgency_label.setStyleSheet("color: #888;")
        self.edit_button = QPushButton("Edit"); self.edit_button.setFixedSize(40, 25)
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self.task_id))
        self.delete_button = QPushButton("Del"); self.delete_button.setFixedSize(40, 25); self.delete_button.setStyleSheet("color: red;")
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.task_id))
        layout.addWidget(self.checkbox); layout.addWidget(self.desc_label, 1); layout.addWidget(self.datetime_label)
        layout.addSpacerItem(QSpacerItem(10, 0)); layout.addWidget(urgency_label); layout.addSpacerItem(QSpacerItem(10, 0))
        layout.addWidget(self.edit_button); layout.addWidget(self.delete_button)
        self.update_appearance()

    def on_checkbox_change(self, state):
        self.is_done = (state == Qt.Checked)
        self.update_appearance()
        self.status_changed.emit(self.task_id, self.is_done)

    def update_appearance(self):
        font = self.desc_label.font(); font.setStrikeOut(self.is_done); self.desc_label.setFont(font)
        palette = self.desc_label.palette(); palette.setColor(self.desc_label.foregroundRole(), Qt.gray if self.is_done else Qt.black)
        self.desc_label.setPalette(palette); self.datetime_label.setVisible(not self.is_done)

# --- Main Application ---
class FlowlyApp(QWidget):
    request_network_action = pyqtSignal(dict)
    set_worker_user = pyqtSignal(str) # Signal to set user in worker

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.setWindowTitle("Flowly - Task Manager")
        self.setGeometry(200, 200, 700, 500)

        # urgency sorting factors setup
        self.CATEGORY_FACTORS = {
            "Work":0.8,
            "School":0.65,
            "Personal":0.6,
            "Household":0.4,
            "Health":0.7,
            "_default_":0.5 # Default for unknown categories
        }
        self.DTIME_WEIGHT = 0.6
        self.CATEGORY_WEIGHT = 0.4
        self.TIME_URGENCY_K=0.05
        self.DF_TIME = 0.1

        self.current_sort_mode = "datetime"
        self.tasks_cache = []

        # --- Network Thread Setup ---
        self.network_thread = QThread()
        self.network_worker = NetworkWorker('127.0.0.1', 4320)
        self.network_worker.moveToThread(self.network_thread)
        # Connect signals/slots
        self.request_network_action.connect(self.network_worker.process_request)
        self.network_worker.response_received.connect(self.handle_server_response)
        self.network_worker.error_occurred.connect(self.handle_network_error)
        self.network_worker._request_finished.connect(self.on_request_finished)
        self.set_worker_user.connect(self.network_worker.set_current_user) # Connect signal
        # Start thread
        self.network_thread.started.connect(self.network_worker.connect_socket)
        self.network_thread.finished.connect(self.network_worker.close_connection)
        self.network_thread.start()

        # --- Initialize State ---
        self.logged_in_user = None
        self.is_request_pending = False
        self._pending_dialog = None # Reference to active login/signup dialog
        self._editing_task_id = None # Track task being edited

        # --- Initialize UI ---
        self.initUI() # Build widgets

        # --- Start Login Flow ---
        # Use QTimer to call showLogin after __init__ completes and event loop starts
        QTimer.singleShot(0, self.showLogin)

    def initUI(self):
        """Sets up the main window widgets. Does NOT show the window."""
        self.layout = QVBoxLayout(self)
        # Input Area
        input_layout = QHBoxLayout()
        self.text_field = QLineEdit(); self.text_field.setPlaceholderText("Enter new task...")
        self.text_field.returnPressed.connect(self.send_task)
        self.sendTask_btn = QPushButton("Add Task"); self.sendTask_btn.clicked.connect(self.send_task)
        self.record_btn = QPushButton("Record"); self.record_btn.clicked.connect(self.record_task)
        input_layout.addWidget(self.text_field, 1); input_layout.addWidget(self.sendTask_btn); input_layout.addWidget(self.record_btn)
        # Sort Control
        sort_layout = QHBoxLayout()
        sort_label = QLabel("Sort by:")
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Due Date", "Urgency"])
        self.sort_combo.currentIndexChanged.connect(self.change_sort_mode)
        sort_layout.addStretch() # Push to the right
        sort_layout.addWidget(sort_label)
        sort_layout.addWidget(self.sort_combo)
        # Task List
        self.task_list = QListWidget(); self.task_list.setSelectionMode(QListWidget.NoSelection)
        self.task_list.setFocusPolicy(Qt.NoFocus); self.task_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.task_list.customContextMenuRequested.connect(self.show_task_context_menu)
        # Button Layout
        button_layout = QHBoxLayout()
        self.refreshList_btn = QPushButton("Refresh List"); self.refreshList_btn.clicked.connect(self.request_get_tasks)
        self.logout_btn = QPushButton("Logout"); self.logout_btn.clicked.connect(self.handle_logout)
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button_layout.addWidget(self.refreshList_btn); button_layout.addWidget(self.logout_btn)
        # Status Bar
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setStyleSheet("color: gray;")
        # Add layouts/widgets
        self.layout.addLayout(input_layout); self.layout.addLayout(sort_layout); self.layout.addWidget(self.task_list)
        self.layout.addLayout(button_layout); self.layout.addWidget(self.status_label)
        # Window starts hidden by default

    def showLogin(self):
        """Shows the modal login dialog."""
        print("DEBUG: showLogin called.")
        # If somehow called when already logged in, just ensure main window is visible
        if self.logged_in_user:
            print("DEBUG: Already logged in, ensuring main window is visible.")
            self.show_main_window()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Login / Sign Up")
        layout = QFormLayout(dialog)
        # Username/Password fields
        self.username_edit = QLineEdit(dialog)
        self.password_edit = QLineEdit(dialog)
        self.password_edit.setEchoMode(QLineEdit.Password)
        layout.addRow(QLabel("Username:"), self.username_edit)
        layout.addRow(QLabel("Password:"), self.password_edit)
        # Buttons
        buttonBox = QDialogButtonBox(dialog)
        login_btn = buttonBox.addButton("Login", QDialogButtonBox.AcceptRole)
        signup_btn = buttonBox.addButton("Sign Up", QDialogButtonBox.ActionRole)
        cancel_btn = buttonBox.addButton(QDialogButtonBox.Cancel)
        layout.addRow(buttonBox)
        # Connect buttons
        login_btn.clicked.connect(lambda: self.attempt_login(dialog))
        signup_btn.clicked.connect(lambda: self.attempt_signup(dialog))
        cancel_btn.clicked.connect(dialog.reject)
        # Clear fields and reference
        self.username_edit.clear(); self.password_edit.clear()
        self._pending_dialog = None # Ensure reference is clear initially

        print("DEBUG: Executing login dialog...")
        self.status_label.setText("Status: Please log in or sign up.")
        result = dialog.exec_() # BLOCKING CALL

        # --- This code runs AFTER dialog is closed ---
        if result == QDialog.Accepted:
            # Login was successful, dialog.accept() was called by handle_server_response
            print("DEBUG: Login dialog finished with Accepted.")
            # Main window should already be visible due to logic in handle_server_response
        else:
            # Dialog was rejected (Cancel button or closed) or login failed
            print(f"DEBUG: Login dialog finished with Rejected (Result code: {result}).")
            if not self.logged_in_user:
                # If we are still not logged in after dialog closes, exit app.
                print("No user logged in. Exiting application.")
                QApplication.instance().quit()
            # Else (if user was logged in before, e.g., logout -> cancel), main window stays.

    def attempt_login(self, dialog):
        """Validates input and sends login request."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            QMessageBox.warning(dialog, "Input Error", "Username and password required.")
            return
        if self.is_request_pending:
            QMessageBox.information(dialog, "Busy", "Processing previous request.")
            return

        self.set_busy_status("Logging in...")
        request = {'action': 'login', 'user': username, 'password': password}
        self._pending_dialog = dialog # Store reference BEFORE sending
        print(f"DEBUG: Stored pending dialog ref: {id(self._pending_dialog)} for login")
        self.request_network_action.emit(request)

    def attempt_signup(self, dialog):
        """Validates input, gets email, and sends signup request."""
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            QMessageBox.warning(dialog, "Input Error", "Username and password required.")
            return

        email, ok = QInputDialog.getText(dialog, "Sign Up - Email", "Enter email:")
        if not ok or not email.strip():
            QMessageBox.warning(dialog, "Input Error", "Email required for sign up.")
            return

        if self.is_request_pending:
            QMessageBox.information(dialog, "Busy", "Processing previous request.")
            return

        self.set_busy_status("Signing up...")
        request = {'action': 'signup', 'username': username, 'email': email.strip(), 'password': password}
        self._pending_dialog = dialog # Store reference BEFORE sending
        print(f"DEBUG: Stored pending dialog ref: {id(self._pending_dialog)} for signup")
        self.request_network_action.emit(request)

    @pyqtSlot(dict)
    def handle_server_response(self, response):
        """Processes responses from the NetworkWorker."""
        print(f"DEBUG: handle_server_response received: {response}")
        action = response.get('action_echo', 'unknown')
        status = response.get('status')
        message = response.get('message', '')

        # --- Check if response is related to the active login/signup dialog ---
        is_pending_dialog_relevant = False
        local_dialog_ref = None # Use a local variable
        if hasattr(self, '_pending_dialog') and self._pending_dialog and action in ['login','signup']:
            print(f"DEBUG: Response action '{action}' matches pending dialog {id(self._pending_dialog)}")
            is_pending_dialog_relevant = True
            local_dialog_ref = self._pending_dialog # Capture reference

        # --- Handle LOGIN response for the dialog ---
        if is_pending_dialog_relevant and action == 'login':
            if status == 'success':
                print(f"DEBUG: Login success in response for user: {response.get('username')}")
                self.logged_in_user = response.get('username')
                if self.logged_in_user:
                    # 1. Inform worker
                    print(f"DEBUG: Emitting set_worker_user for {self.logged_in_user}")
                    self.set_worker_user.emit(self.logged_in_user)
                    # 2. Update main window UI
                    print("DEBUG: Updating main UI (title, status).")
                    self.setWindowTitle(f"Flowly - {self.logged_in_user}")
                    self.status_label.setText(f"Status: Logged in as {self.logged_in_user}.")
                    # 3. Show main window
                    print("DEBUG: Calling show_main_window.")
                    self.show_main_window()
                    # 4. Accept the dialog (must happen AFTER showing main window)
                    print(f"DEBUG: Accepting dialog {id(local_dialog_ref)}.")
                    local_dialog_ref.accept() # Use the local reference
                    # 5. Fetch initial data
                    print("DEBUG: Requesting initial tasks after login.")
                    self.request_get_tasks()
                else:
                    print("ERROR: Login success response missing username.")
                    QMessageBox.critical(local_dialog_ref, "Login Error", "Login incomplete.")
                    self.logged_in_user = None
            else: # Login failed
                print(f"DEBUG: Login failed in response: {message}")
                QMessageBox.critical(local_dialog_ref, "Login Failed", message or "Invalid credentials.")
                self.logged_in_user = None

            # Clear reference *after* handling response for this dialog
            print(f"DEBUG: Clearing _pending_dialog reference {id(self._pending_dialog)} after login handling.")
            self._pending_dialog = None

        # --- Handle SIGNUP response for the dialog ---
        elif is_pending_dialog_relevant and action == 'signup':
            if status == 'success':
                QMessageBox.information(local_dialog_ref, "Signup Successful", message or "Account created. Please log in.")
                if hasattr(self,'username_edit'): self.username_edit.clear()
                if hasattr(self,'password_edit'): self.password_edit.clear()
                if hasattr(self,'username_edit'): self.username_edit.setFocus()
            else: # Signup failed
                QMessageBox.critical(local_dialog_ref, "Signup Failed", message or "Could not create account.")

            # Clear reference *after* handling response for this dialog
            print(f"DEBUG: Clearing _pending_dialog reference {id(self._pending_dialog)} after signup handling.")
            self._pending_dialog = None

        # --- Handle OTHER responses (only if logged in) ---
        elif self.logged_in_user:
            if action == 'get_tasks' and status == 'success':
                print("DEBUG: Handling get_tasks response.")
                tasks = response.get('tasks', [])
                self.sort_and_display_tasks(tasks)
                if not self.is_request_pending: self.status_label.setText(f"Tasks loaded ({len(tasks)}). sorted by {self.current_sort_mode.replace('_', ' ').title()}")
            elif action == 'add_task' and status == 'task_added':
                QMessageBox.information(self, "Success", message or "Task added.")
                self.request_get_tasks() # Refresh
            elif action == 'update_task_status' and status == 'status_updated':
                print(f"Task {response.get('task_id')} status updated.")
                # Optionally update status label briefly
            elif action == 'update_task' and status == 'task_updated':
                 QMessageBox.information(self, "Success", "Task details updated.")
                 self.request_get_tasks() # Refresh
            elif action == 'delete_task' and status == 'task_deleted':
                 QMessageBox.information(self, "Success", f"Task deleted.")
                 self.request_get_tasks() # Refresh
            elif action in ['add_task', 'update_task_status', 'update_task', 'delete_task'] and 'success' in status or 'updated' in status or 'deleted' in status:
                 QMessageBox.information(self, "Success", response.get('message', f"Action {action} successful."))
                 self.request_get_tasks()
            # Handle get_task response for editing
            elif action == 'get_task' and status == 'success' and 'task' in response:
                 if hasattr(self, '_editing_task_id') and self._editing_task_id == response['task'].get('TaskID'):
                     print(f"DEBUG: Received details for edit task {self._editing_task_id}.")
                     self.show_edit_dialog(response['task'])
                     return # Edit dialog handles its own flow
                 else:
                     print("WARN: Received unexpected get_task details.")
            # Handle general errors for logged-in users
            elif status and status != 'success': # Catch specific error statuses
                 QMessageBox.warning(self, f"Action Failed ({action})", message or "An error occurred.")
            elif 'error' in response and not status: # Catch generic errors
                 QMessageBox.critical(self, "Server Error", response['error'])

        # --- Ignore responses if not logged in and not login/signup response ---
        elif not self.logged_in_user and action not in ['login', 'signup']:
            print(f"WARN: Ignored action '{action}' response; user not logged in.")

    def show_main_window(self):
        """Helper function to ensure the main window is visible and active."""
        print("DEBUG: show_main_window() called.")
        if self.isHidden():
            print("DEBUG: Main window is hidden, calling self.show().")
            self.show()
        else:
            print("DEBUG: Main window already visible, ensuring raised/activated.")
        self.raise_()
        self.activateWindow()

    @pyqtSlot(str)
    def handle_network_error(self, error_message):
        """Handles network errors reported by the worker."""
        print(f"ERROR: Network error reported: {error_message}")
        # If error happens during login/signup attempt
        if hasattr(self, '_pending_dialog') and self._pending_dialog:
             QMessageBox.critical(self._pending_dialog, "Network Error", error_message)
             self._pending_dialog = None # Clear ref as the attempt failed
        else: # General error when app is running
             QMessageBox.critical(self, "Network Error", f"{error_message}\nCheck connection/server.")
        self.on_request_finished() # Ensure controls are re-enabled

    @pyqtSlot()
    def on_request_finished(self):
        """Called when network request cycle completes (success or error)."""
        print("DEBUG: Request finished.")
        self.is_request_pending = False
        self.set_controls_enabled(True)
        # Reset status bar if it was showing a busy message, but only if logged in
        if self.logged_in_user and self.status_label.text().startswith("Working..."):
             self.status_label.setText(f"Status: Logged in as {self.logged_in_user}.")
        elif not self.logged_in_user and self.status_label.text().startswith("Working..."):
             self.status_label.setText(f"Status: Please log in or sign up.")

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    def change_sort_mode(self, index):
        mode = self.sort_combo.currentText().lower().replace(" ", "_") # "due_date" or "urgency"
        if mode != self.current_sort_mode:
            self.current_sort_mode = mode
            print(f"DEBUG: Sort mode changed to {self.current_sort_mode}")
            self.sort_and_display_tasks(self.tasks_cache)
    
    def calculate_urgency(self, task_data):
        """Calculates dynamic urgency score for a single task."""
        # Unpack carefully, assuming task_data tuple structure:
        # (TaskID, TaskDesc, DateStr, TimeStr, Category, UrgencyDB, Status)
        try:
            _, _, date_str, time_str, category, _, _ = task_data
        except ValueError:
            print(f"WARN: Could not unpack task data for urgency: {task_data}")
            return 0 # Or some default low score

        now = datetime.datetime.now()
        due_datetime = None

        # --- Combine Date and Time ---
        if date_str and date_str != 'None':
            try:
                parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                parsed_time = datetime.time.min # Default to start of day
                if time_str and time_str != 'None':
                    try:
                        # Handle potential microseconds if your DB stores them
                        time_str_clean = time_str.split('.')[0]
                        parsed_time = datetime.datetime.strptime(time_str_clean, '%H:%M:%S').time()
                    except ValueError:
                        print(f"WARN: Invalid time format '{time_str}', using 00:00:00")
                due_datetime = datetime.datetime.combine(parsed_date, parsed_time)
            except ValueError:
                print(f"WARN: Invalid date format '{date_str}', cannot determine due date.")

        # --- Calculate TimeFactor ---
        time_factor = self.DF_TIME
        if due_datetime:
            delta = due_datetime - now
            delta_hours = delta.total_seconds() / 3600.0
            if delta_hours <= 0: # Overdue or due now
                time_factor = 1.0
            else:
                time_factor = math.exp(-self.TIME_URGENCY_K * delta_hours)

        # --- Calculate CategoryFactor ---
        category_factor = self.CATEGORY_FACTORS.get(category, self.CATEGORY_FACTORS["_default_"]) if category else self.CATEGORY_FACTORS["_default_"]

        # --- Final Score ---
        urgency_score = (self.DTIME_WEIGHT * time_factor) + (self.CATEGORY_WEIGHT * category_factor)
        # print(f"DEBUG: TaskID {task_data[0]}, Due: {due_datetime}, DeltaH: {delta_hours if due_datetime else 'N/A'}, TF: {time_factor:.3f}, CF: {category_factor:.3f}, Score: {urgency_score:.3f}")
        return urgency_score

    def sort_and_display_tasks(self, tasks):
        """Sorts the raw task data based on current mode and updates list."""
        self.tasks_cache = tasks # Update cache
        print(f"DEBUG: Sorting {len(tasks)} tasks by {self.current_sort_mode}")

        if not tasks:
            self.populate_task_list([]) # Clear list if no tasks
            return

        sorted_tasks = []
        if self.current_sort_mode == "urgency":
            # Create list of (score, task_data) tuples for sorting
            tasks_with_scores = []
            for task in tasks:
                score = self.calculate_urgency(task)
                tasks_with_scores.append((score, task))
            # Sort descending by score
            tasks_with_scores.sort(key=lambda item: item[0], reverse=True)
            sorted_tasks = [item[1] for item in tasks_with_scores] # Extract sorted task data

        elif self.current_sort_mode == "due_date":
            # Sort by date, time (earliest first). Handle None values (put them last).
            def get_sort_key(task):
                # TaskID, TaskDesc, DateStr, TimeStr, Category, UrgencyDB, Status
                date_str = task[2]
                time_str = task[3]
                # Represent tasks with no date/time as very far in the future
                far_future_datetime = datetime.datetime.max

                if date_str and date_str != 'None':
                    try:
                        parsed_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                        parsed_time = datetime.time.min
                        if time_str and time_str != 'None':
                             try:
                                 time_str_clean = time_str.split('.')[0]
                                 parsed_time = datetime.datetime.strptime(time_str_clean, '%H:%M:%S').time()
                             except ValueError: pass # Keep min time
                        return datetime.datetime.combine(parsed_date, parsed_time)
                    except ValueError:
                        return far_future_datetime # Invalid date format -> last
                else:
                    return far_future_datetime # No date -> last

            sorted_tasks = sorted(tasks, key=get_sort_key)
        else: # Default or unknown sort
            sorted_tasks = tasks # Keep original order

        self.populate_task_list(sorted_tasks)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # --- set_controls_enabled Method ---
    def set_controls_enabled(self, enabled):
        self.sendTask_btn.setEnabled(enabled); self.record_btn.setEnabled(enabled)
        self.refreshList_btn.setEnabled(enabled); self.task_list.setEnabled(enabled)

    # --- set_busy_status Method ---
    def set_busy_status(self, message="Working..."):
        self.status_label.setText(message)
        self.is_request_pending = True
        self.set_controls_enabled(False)

    # --- populate_task_list Method (Keep As Is) ---
    def populate_task_list(self, sorted_tasks):
        self.task_list.clear()
        if not sorted_tasks:
            item = QListWidgetItem("No tasks found."); item.setFlags(Qt.NoItemFlags); self.task_list.addItem(item)
            return
        print(f"DEBUG: Populating task list with {len(sorted_tasks)} sorted tasks.")
        for task_data in sorted_tasks:
            try:
                task_id, desc, date, time, cat, urg, stat = task_data # Unpack carefully
                task_widget = TaskItemWidget(task_id, desc, date, time, cat, urg, stat)
                task_widget.status_changed.connect(self.handle_task_status_change)
                task_widget.edit_requested.connect(self.request_edit_task)
                task_widget.delete_requested.connect(self.request_delete_task)
                list_item = QListWidgetItem(self.task_list); list_item.setSizeHint(task_widget.sizeHint())
                list_item.setData(Qt.UserRole, task_id); self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, task_widget)
            except Exception as e: print(f"Error creating widget for task {task_data}: {e}")

    # --- request_get_tasks Method (Keep As Is) ---
    def request_get_tasks(self):
        if not self.logged_in_user: QMessageBox.warning(self, "Not Logged In", "Please log in."); return
        if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
        self.set_busy_status("Refreshing tasks...")
        self.request_network_action.emit({'action': 'get_tasks'})

    # --- handle_task_status_change Method (Keep As Is) ---
    @pyqtSlot(int, bool)
    def handle_task_status_change(self, task_id, is_done):
        if self.is_request_pending: print("WARN: Ignoring status change during request."); return
        new_status = 'done' if is_done else 'pending'
        self.set_busy_status(f"Updating task {task_id}...")
        self.request_network_action.emit({'action': 'update_task_status', 'task_id': task_id, 'status': new_status})

    # --- send_task Method (Keep As Is) ---
    def send_task(self):
        task_desc = self.text_field.text().strip()
        if not task_desc: QMessageBox.warning(self, "Input Error", "Task description empty."); return
        if not self.logged_in_user: QMessageBox.warning(self, "Error", "Not logged in."); return
        if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
        self.set_busy_status("Adding task...")
        self.request_network_action.emit({'action': 'add_task', 'task_desc': task_desc})
        self.text_field.clear()

    # --- record_task Method (Keep As Is - includes previous fix) ---
    def record_task(self):
        if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
        self.record_btn.setEnabled(False); self.record_btn.setText("Recording...")
        threading.Thread(target=self._record_task_thread, daemon=True).start()

    # --- _record_task_thread Method (Keep As Is - includes previous fix) ---
    def _record_task_thread(self):
        recognized_text = None; error_message = None
        try:
            with sr.Microphone() as mic:
                self.recognizer.adjust_for_ambient_noise(mic, duration=0.3)
                print("DEBUG: Listening..."); audio = self.recognizer.listen(mic, timeout=5, phrase_time_limit=10)
                print("DEBUG: Processing..."); recognized_text = self.recognizer.recognize_google(audio).lower()
        except sr.WaitTimeoutError: error_message = "No speech detected."
        except sr.UnknownValueError: error_message = "Could not understand audio."
        except sr.RequestError as e: error_message = f"Recognition service error; {e}"
        except Exception as e: error_message = f"Recording error: {e}"; print(f"ERROR: {e}")
        finally:
            if recognized_text: QMetaObject.invokeMethod(self.text_field, "setText", Qt.QueuedConnection, Q_ARG(str, recognized_text))
            if error_message: QMetaObject.invokeMethod(self.status_label, "setText", Qt.QueuedConnection, Q_ARG(str, f"Status: {error_message}"))
            QMetaObject.invokeMethod(self.record_btn, "setEnabled", Qt.QueuedConnection, Q_ARG(bool, True))
            QMetaObject.invokeMethod(self.record_btn, "setText", Qt.QueuedConnection, Q_ARG(str, "Record"))

    # --- show_task_context_menu Method (Keep As Is) ---
    @pyqtSlot(QPoint)
    def show_task_context_menu(self, pos):
        list_item = self.task_list.itemAt(pos);
        if not list_item: return
        task_id = list_item.data(Qt.UserRole);
        if not task_id: return
        global_pos = self.task_list.mapToGlobal(pos); menu = QMenu()
        edit_action = menu.addAction("Edit Task"); delete_action = menu.addAction("Delete Task")
        action = menu.exec_(global_pos)
        if action == edit_action: self.request_edit_task(task_id)
        elif action == delete_action: self.request_delete_task(task_id)

    # --- request_delete_task Method (Keep As Is) ---
    @pyqtSlot(int)
    def request_delete_task(self, task_id):
        if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
        confirm = QMessageBox.question(self, "Confirm Delete", f"Delete task {task_id}?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.set_busy_status(f"Deleting task {task_id}...")
            self.request_network_action.emit({'action': 'delete_task', 'task_id': task_id})

    # --- request_edit_task Method (Keep As Is) ---
    @pyqtSlot(int)
    def request_edit_task(self, task_id):
        if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
        self._editing_task_id = task_id; self.set_busy_status(f"Fetching task {task_id}...")
        self.request_network_action.emit({'action': 'get_task', 'task_id': task_id})

    # --- show_edit_dialog Method (Keep As Is) ---
    def show_edit_dialog(self, task_data):
        task_id = task_data.get('TaskID')
        if not hasattr(self, '_editing_task_id') or self._editing_task_id != task_id:
             print(f"WARN: Ignoring edit data for task {task_id}."); return
        dialog = QDialog(self); dialog.setWindowTitle(f"Edit Task {task_id}"); layout = QVBoxLayout(dialog); form_layout = QFormLayout()
        desc_edit = QLineEdit(task_data.get('TaskDesc', ''))
        date_edit = QLineEdit(task_data.get('Date', '') if task_data.get('Date') != 'None' else ''); date_edit.setPlaceholderText("YYYY-MM-DD")
        time_edit = QLineEdit(task_data.get('Time', '') if task_data.get('Time') != 'None' else ''); time_edit.setPlaceholderText("HH:MM:SS")
        category_edit = QLineEdit(task_data.get('Category', 'General'))
        urgency_combo = QComboBox(); urgency_combo.addItems([str(i) for i in range(1, 6)]); urgency_combo.setCurrentText(str(task_data.get('Urgency', 3)))
        status_combo = QComboBox(); status_combo.addItems(["pending", "done"]); status_combo.setCurrentText(task_data.get('Status', 'pending'))
        form_layout.addRow("Desc:", desc_edit); form_layout.addRow("Date:", date_edit); form_layout.addRow("Time:", time_edit)
        form_layout.addRow("Category:", category_edit); form_layout.addRow("Urgency:", urgency_combo); form_layout.addRow("Status:", status_combo)
        layout.addLayout(form_layout)
        btn_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel); btn_box.accepted.connect(dialog.accept); btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)
        del self._editing_task_id # Clear flag before showing

        if dialog.exec_() == QDialog.Accepted:
            update_data = {'TaskDesc': desc_edit.text().strip(), 'Date': date_edit.text().strip() or None, 'Time': time_edit.text().strip() or None,
                           'Category': category_edit.text().strip(), 'Urgency': int(urgency_combo.currentText()), 'Status': status_combo.currentText()}
            if not update_data['TaskDesc']: QMessageBox.warning(self, "Input Error", "Desc empty."); return
            if self.is_request_pending: QMessageBox.information(self, "Busy", "Processing request."); return
            self.set_busy_status(f"Saving task {task_id}...")
            self.request_network_action.emit({'action': 'update_task', 'task_id': task_id, 'update_data': update_data})

    # --- handle_logout Method ---
    def handle_logout(self):
         if self.logged_in_user:
              confirm = QMessageBox.question(self, "Confirm Logout", f"Log out?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
              if confirm == QMessageBox.Yes:
                    print(f"Logging out user {self.logged_in_user}")
                    # Store username before clearing
                    logged_out_user = self.logged_in_user
                    self.logged_in_user = None
                    # Inform worker user is gone - send empty string
                    self.set_worker_user.emit("")
                    # Optionally close/reset connection
                    # self.network_worker.close_connection()  
                    # Clear UI immediately
                    self.task_list.clear()
                    self.setWindowTitle("Flowly - Task Manager")
                    self.status_label.setText("Status: Logged out.")
                    self.hide() # Hide main window

                    # Use QTimer to show login dialog AFTER current event processing
                    QTimer.singleShot(0, self.showLogin)

    # --- closeEvent Method (Keep As Is) ---
    def closeEvent(self, event):
        print("Close event triggered. Stopping network thread...")
        self.network_worker.close_connection()
        self.network_thread.quit()
        if not self.network_thread.wait(3000):
             print("WARN: Network thread termination.")
             self.network_thread.terminate()
        print("Network thread stopped.")
        event.accept()

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # app.setStyle('Fusion')
    window = FlowlyApp()
    sys.exit(app.exec_())