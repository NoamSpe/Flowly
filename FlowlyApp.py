# FlowlyApp.py
import sys
import threading
import speech_recognition as sr # For speech recognition

from PyQt5.QtWidgets import (QApplication, QWidget, QListWidgetItem, QInputDialog,
                             QMessageBox, QMenu, QDialog) # Added QDialog
from PyQt5.QtCore import (pyqtSignal, Qt, QThread, pyqtSlot,
                          QPoint, QTimer, QDate) # Removed Q_ARG, QMetaObject
from PyQt5.QtGui import QIcon

# Local imports
from client_network import NetworkWorker
from gui_task_widget import TaskItemWidget
import gui_components    # For UI setup functions
import task_logic       # For task processing functions
from app_config import (SERVER_HOST, SERVER_PORT, LOGO_SQUARE_PATH, APP_TITLE_BASE,
                        STYLESHEET_PATH)


class FlowlyApp(QWidget):
    request_network_action = pyqtSignal(dict)
    set_worker_user = pyqtSignal(str)

    # Signals for UI updates from threads (like speech recognition)
    update_record_button_signal = pyqtSignal(bool, str) # (enabled, text)
    update_status_text_signal = pyqtSignal(str)         # Renamed for clarity
    update_task_text_field_signal = pyqtSignal(str)     # Renamed for clarity

    def __init__(self):
        super().__init__()
        
        # Speech Recognizer Setup
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.3 # seconds of non-speaking audio before phrase is considered complete
        self.recognizer.energy_threshold = 4000 # higher values mean it's less sensitive to background noise

        self.setWindowTitle(APP_TITLE_BASE)
        self.setWindowIcon(QIcon(LOGO_SQUARE_PATH))
        self.setGeometry(100, 100, 1024, 768) # Adjusted size slightly

        self.current_sort_mode = "due_date" 
        self.tasks_cache = []

        self.selected_calendar_date = None
        self.highlighted_dates = set()

        # --- Network Thread Setup ---
        self.network_thread = QThread(self) # Parent it to self for better lifetime management
        self.network_worker = NetworkWorker(SERVER_HOST, SERVER_PORT)
        self.network_worker.moveToThread(self.network_thread)
        self.request_network_action.connect(self.network_worker.process_request)
        self.network_worker.response_received.connect(self.handle_server_response)
        self.network_worker.error_occurred.connect(self.handle_network_error)
        self.network_worker._request_finished.connect(self.on_request_finished)
        self.set_worker_user.connect(self.network_worker.set_current_user)
        
        self.network_thread.started.connect(self.network_worker.connect_socket)
        # self.network_thread.finished.connect(self.network_worker.close_connection) # Connection closed in closeEvent
        self.network_thread.finished.connect(self.network_thread.deleteLater) # Clean up thread
        self.network_worker.destroyed.connect(self.network_thread.quit) # If worker is deleted, quit thread

        self.network_thread.start()

        # --- Initialize State ---
        self.logged_in_user = None
        self.is_request_pending = False
        self._pending_dialog = None
        self._editing_task_id = None

        # Connect UI update signals to their respective slots
        self.update_record_button_signal.connect(self._handle_record_button_update)
        self.update_status_text_signal.connect(lambda text: self.status_label.setText(text))
        self.update_task_text_field_signal.connect(lambda text: self.text_field.setText(text))

        # --- Initialize UI ---
        gui_components.setup_main_ui(self)

        QTimer.singleShot(0, self.showLogin) # Show login dialog after __init__ completes

    # --- Login/Signup Flow ---
    def showLogin(self):
        if self.logged_in_user:
            self.show_main_window()
            return

        dialog, self.username_edit, self.password_edit, login_btn, signup_btn, cancel_btn = \
            gui_components.create_login_dialog(self)
        
        login_btn.clicked.connect(lambda: self.attempt_login(dialog))
        signup_btn.clicked.connect(lambda: self.attempt_signup(dialog))
        cancel_btn.clicked.connect(dialog.reject)

        self.username_edit.clear()
        self.password_edit.clear()
        self._pending_dialog = None # Clear any stale reference

        self.status_label.setText("Status: Please log in or sign up.")
        result = dialog.exec_()

        if result != QDialog.Accepted and not self.logged_in_user:
            QApplication.instance().quit()
    
    def attempt_login(self, dialog_instance):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            QMessageBox.warning(dialog_instance, "Input Error", "Username and password are required.")
            return
        if self.is_request_pending:
            QMessageBox.information(dialog_instance, "Busy", "Processing a previous request. Please wait.")
            return
        self.set_busy_status("Logging in...")
        request = {'action': 'login', 'user': username, 'password': password}
        self._pending_dialog = dialog_instance # Store reference to active dialog
        self.request_network_action.emit(request)

    def attempt_signup(self, dialog_instance):
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            QMessageBox.warning(dialog_instance, "Input Error", "Username and password are required.")
            return
        if not (len(password) >= 8 and any(c.isupper() for c in password) and \
                any(c.islower() for c in password) and any(c.isdigit() for c in password)):
            QMessageBox.warning(dialog_instance, "Input Error", 
                                "Password not adequate.\n"
                                "Must be at least 8 characters long and include at least one uppercase letter, "
                                "one lowercase letter, and one digit.")
            return
        
        email, ok = QInputDialog.getText(dialog_instance, "Sign Up - Email", "Enter your email address:")
        if not ok or not email.strip(): # Also check if email is empty after stripping
            QMessageBox.warning(dialog_instance, "Input Error", "A valid email address is required for sign up.")
            return
        # Basic email format check (not exhaustive)
        if "@" not in email or "." not in email.split('@')[-1]:
            QMessageBox.warning(dialog_instance, "Input Error", "Please enter a valid email address format.")
            return

        if self.is_request_pending:
            QMessageBox.information(dialog_instance, "Busy", "Processing a previous request. Please wait.")
            return
        self.set_busy_status("Signing up...")
        request = {'action': 'signup', 'username': username, 'email': email.strip(), 'password': password}
        self._pending_dialog = dialog_instance
        self.request_network_action.emit(request)

    # --- Network Callbacks ---
    @pyqtSlot(dict)
    def handle_server_response(self, response):
        print(f"DEBUG (FlowlyApp): UI received server response: {response}")
        action = response.get('action_echo', 'unknown')
        status = response.get('status')
        message = response.get('message', '')

        local_dialog_ref = self._pending_dialog # Capture before it's potentially cleared

        if local_dialog_ref and action in ['login','signup']:
            if action == 'login':
                if status == 'success':
                    self.logged_in_user = response.get('username')
                    if self.logged_in_user:
                        self.set_worker_user.emit(self.logged_in_user)
                        self.setWindowTitle(f"{APP_TITLE_BASE} - {self.logged_in_user}")
                        self.update_status_text_signal.emit(f"Status: Logged in as {self.logged_in_user}.")
                        self.show_main_window()
                        local_dialog_ref.accept() 
                        self.request_get_tasks() # Fetch initial tasks
                        # update_calendar_highlights will be called by sort_and_display_tasks
                    else:
                        QMessageBox.critical(local_dialog_ref, "Login Error", "Login successful, but username was not returned.")
                else:
                    QMessageBox.critical(local_dialog_ref, "Login Failed", message or "Invalid username or password.")
                self._pending_dialog = None # Dialog handled
            
            elif action == 'signup':
                if status == 'success':
                    QMessageBox.information(local_dialog_ref, "Signup Successful", message or "Account created successfully. Please log in.")
                    if hasattr(self,'username_edit'): self.username_edit.clear()
                    if hasattr(self,'password_edit'): self.password_edit.clear()
                    if hasattr(self,'username_edit'): self.username_edit.setFocus()
                else:
                    QMessageBox.critical(local_dialog_ref, "Signup Failed", message or "Could not create account. Username or email may already exist.")
                self._pending_dialog = None # Dialog handled

        elif self.logged_in_user: # Other actions for logged-in users
            if action == 'get_tasks':
                if status == 'success':
                    self.tasks_cache = response.get('tasks', [])
                    self.sort_and_display_tasks() 
                    if not self.is_request_pending: # Only update status if not immediately followed by another busy state
                         self.update_status_text_signal.emit(f"Tasks loaded ({len(self.tasks_cache)}).")
                else:
                    QMessageBox.warning(self, "Task Load Failed", message or "Could not retrieve tasks.")
                    self.update_status_text_signal.emit("Status: Failed to load tasks.")
            
            elif action == 'add_task':
                if status == 'task_added': # Or a more generic 'success'
                    QMessageBox.information(self, "Success", message or "Task added successfully.")
                    self.request_get_tasks() # Refresh list
                else:
                    QMessageBox.warning(self, "Add Task Failed", message or "Could not add the task.")
            
            elif action == 'update_task_status':
                if status == 'status_updated': # Or 'success'
                    print(f"Task {response.get('task_id')} status updated successfully.")
                    # No message box for this, just refresh the list.
                    self.request_get_tasks()
                else:
                    QMessageBox.warning(self, "Update Failed", message or "Could not update task status.")
                    self.request_get_tasks() # Refresh to revert potential optimistic UI changes

            elif action == 'update_task':
                 if status == 'task_updated': # Or 'success'
                     QMessageBox.information(self, "Success", "Task details updated successfully.")
                     self.request_get_tasks()
                 else:
                     QMessageBox.warning(self, "Update Failed", message or "Could not update task details.")
            
            elif action == 'delete_task':
                 if status == 'task_deleted': # Or 'success'
                     QMessageBox.information(self, "Success", f"Task deleted successfully.")
                     self.request_get_tasks()
                 else:
                     QMessageBox.warning(self, "Delete Failed", message or "Could not delete the task.")
            
            elif action == 'get_task' and status == 'success' and 'task' in response:
                 # This is for fetching task details before editing
                 if hasattr(self, '_editing_task_id') and self._editing_task_id == response['task'].get('TaskID'):
                     self.show_edit_dialog_with_data(response['task'])
                     # self.on_request_finished() is called after dialog closes or by network worker
                 else:
                     print("WARN (FlowlyApp): Received unexpected get_task details or mismatched ID.")
                     if not self.is_request_pending and self._editing_task_id: # If we were expecting this but it was wrong
                        self.on_request_finished() # Reset busy state
                     self._editing_task_id = None
            
            # General error handling for other actions if status is not success-like
            elif status and 'success' not in status and 'updated' not in status and 'deleted' not in status and 'added' not in status:
                 QMessageBox.warning(self, f"Action Failed ({action})", message or "An unspecified error occurred.")
            elif 'error' in response and not status: # Generic server error
                 QMessageBox.critical(self, "Server Error", response['error'])
        
        elif not self.logged_in_user and action not in ['login', 'signup']:
            print(f"WARN (FlowlyApp): Ignored action '{action}' response; user not logged in and not a login/signup action.")
        
        # If no specific handling set request_pending to false, do it in on_request_finished
        # self.on_request_finished() will be called by the network worker signal

    def show_main_window(self):
        if self.isHidden(): self.show()
        self.raise_()
        self.activateWindow()

    @pyqtSlot(str)
    def handle_network_error(self, error_message):
        print(f"ERROR (FlowlyApp): Network error received by UI: {error_message}")
        active_dialog = self._pending_dialog if hasattr(self, '_pending_dialog') else self
        
        QMessageBox.critical(active_dialog, "Network Error", 
                             f"{error_message}\nPlease check your connection and server status.")
        
        if self._pending_dialog: # If it was a login/signup dialog error
            self._pending_dialog = None # Clear dialog reference as it failed
        
        self.on_request_finished() # Ensure UI is re-enabled

    @pyqtSlot()
    def on_request_finished(self):
        """Called when network request cycle (send/receive) completes, successfully or with error."""
        print("DEBUG (FlowlyApp): Network request cycle finished.")
        self.is_request_pending = False
        self.set_controls_enabled(True) # Re-enable UI controls

        # Reset status bar if it was showing a busy message
        current_status_text = self.status_label.text()
        if current_status_text.startswith("Working...") or \
           current_status_text.startswith("Logging in...") or \
           current_status_text.startswith("Signing up...") or \
           current_status_text.startswith("Refreshing tasks...") or \
           current_status_text.startswith("Adding task...") or \
           current_status_text.startswith("Updating task...") or \
           current_status_text.startswith("Deleting task...") or \
           current_status_text.startswith("Fetching task..."):
            if self.logged_in_user:
                 self.update_status_text_signal.emit(f"Status: Logged in as {self.logged_in_user}.")
            else:
                 self.update_status_text_signal.emit(f"Status: Please log in or sign up.")
        # _editing_task_id should be cleared by the logic that initiates/handles the edit
        # If a get_task request finishes and _editing_task_id is still set (e.g. error before dialog shown)
        if self._editing_task_id and not self._pending_dialog: # Check _pending_dialog to avoid clearing during active dialog
            print(f"DEBUG (FlowlyApp): Clearing _editing_task_id ({self._editing_task_id}) after request finished without edit dialog.")
            self._editing_task_id = None

    # --- Task and List Management ---
    def change_sort_mode(self, index):
        mode_text = self.sort_combo.currentText().lower().replace(" ", "_")
        if mode_text != self.current_sort_mode:
            self.current_sort_mode = mode_text
            self.sort_and_display_tasks()
    
    def sort_and_display_tasks(self):
        status_filter_state = {
            'all': self.Rstatus_all.isChecked(),
            'pending': self.Rstatus_pending.isChecked(),
            'done': self.Rstatus_done.isChecked()
        }
        category_checkboxes_state = {cat: cb.isChecked() for cat, cb in self.category_checkboxes.items()}
        
        ui_sort_mode = 'due_date' if self.sort_combo.currentIndex() == 0 else 'urgency'

        is_done_filter_active = self.Rstatus_done.isChecked()
        is_all_filter_active_and_no_pending_or_done = self.Rstatus_all.isChecked() and \
                                                    not (self.Rstatus_pending.isChecked() or self.Rstatus_done.isChecked())


        # Enable sort_combo unless "Done" is exclusively selected or "All" is selected
        # and implies a mix that might include "Done"
        if is_done_filter_active:
            self.sort_combo.setEnabled(False)
            if self.sort_combo.currentIndex() != 0: self.sort_combo.setCurrentIndex(0) # Force "Due Date"
            ui_sort_mode = 'due_date'
        else:
            self.sort_combo.setEnabled(True)


        final_sorted_tasks = task_logic.filter_and_sort_tasks(
            self.tasks_cache, self.selected_calendar_date,
            status_filter_state, category_checkboxes_state, ui_sort_mode
        )
        self.populate_task_list(final_sorted_tasks)
        self.update_calendar_highlights()
        self.update_task_list_label_text(final_sorted_tasks) # Renamed for clarity

    def update_task_list_label_text(self, tasks_on_display):
        count = len(tasks_on_display)
        date_info_str = f"for {self.selected_calendar_date.toString('yyyy-MM-dd')}" if self.selected_calendar_date else "All Dates"
        
        status_info_str = "Tasks"
        if self.Rstatus_pending.isChecked(): status_info_str = "Pending Tasks"
        elif self.Rstatus_done.isChecked(): status_info_str = "Completed Tasks"

        # Determine effective sort mode for display label
        sort_display_mode = self.current_sort_mode.replace('_', ' ').title()
        if self.Rstatus_done.isChecked(): # If "Done" filter is active, sorting is always by Due Date
            sort_display_mode = "Due Date"
        
        self.task_list_label.setText(f"Showing {count} {status_info_str} ({date_info_str}) - Sorted by {sort_display_mode}")


    def populate_task_list(self, sorted_tasks_to_display):
        self.task_list.clear()
        if not sorted_tasks_to_display:
            item = QListWidgetItem("No tasks match your filters.")
            item.setFlags(Qt.NoItemFlags)
            self.task_list.addItem(item)
            return
        
        for task_data in sorted_tasks_to_display:
            try:
                # Ensure task_data has enough elements before unpacking
                if len(task_data) < 6:
                    print(f"WARN (FlowlyApp): Malformed task data skipped: {task_data}")
                    continue
                task_id, desc, date, time, cat, stat = task_data
                
                task_widget = TaskItemWidget(task_id, desc, date, time, cat, stat)
                task_widget.status_changed.connect(self.handle_task_status_change)
                task_widget.edit_requested.connect(self.request_edit_task)
                task_widget.delete_requested.connect(self.request_delete_task)
                
                list_item = QListWidgetItem(self.task_list)
                list_item.setSizeHint(task_widget.sizeHint())
                list_item.setData(Qt.UserRole, task_id)
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, task_widget)
            except Exception as e:
                print(f"ERROR (FlowlyApp): Creating task widget for {task_data}: {e}")

    def request_get_tasks(self):
        if not self.logged_in_user:
            QMessageBox.warning(self, "Not Logged In", "Please log in to view tasks.")
            return
        if self.is_request_pending:
            QMessageBox.information(self, "Busy", "Already fetching tasks. Please wait.")
            return
        self.set_busy_status("Refreshing tasks...")
        self.request_network_action.emit({'action': 'get_tasks'})

    @pyqtSlot(int, bool)
    def handle_task_status_change(self, task_id, is_done):
        if self.is_request_pending:
            print("WARN (FlowlyApp): Ignoring task status change, a request is pending.")
            # Ideally, revert the checkbox state here or disable it during requests
            return
        new_status = 'done' if is_done else 'pending'
        self.set_busy_status(f"Updating task {task_id} to {new_status}...")
        self.request_network_action.emit({
            'action': 'update_task_status',
            'task_id': task_id,
            'status': new_status
        })

    # --- UI Controls State ---
    def set_controls_enabled(self, enabled):
        # Main action buttons
        self.sendTask_btn.setEnabled(enabled)
        self.record_btn.setEnabled(enabled)
        self.refreshList_btn.setEnabled(enabled)
        
        # Task list interaction
        self.task_list.setEnabled(enabled)
        
        # Calendar
        if hasattr(self, 'calendar_widget'): self.calendar_widget.setEnabled(enabled)
        self.clearDateFilter_btn.setEnabled(enabled)

        # Filters
        self.Rstatus_all.setEnabled(enabled)
        self.Rstatus_pending.setEnabled(enabled)
        self.Rstatus_done.setEnabled(enabled)
        for cb in self.category_checkboxes.values():
            cb.setEnabled(enabled)
        
        # Sort combo is handled specially by sort_and_display_tasks based on "Done" filter
        # However, if all controls are disabled, sort_combo should also be.
        if not enabled:
            self.sort_combo.setEnabled(False)
        else:
            # Re-evaluate sort_combo enabled state based on current filters
            is_done_filter_active = self.Rstatus_done.isChecked()
            self.sort_combo.setEnabled(not is_done_filter_active)


    def set_busy_status(self, message="Working..."):
        self.update_status_text_signal.emit(message)
        self.is_request_pending = True
        self.set_controls_enabled(False)

    def update_filters_display(self):
        if not self.is_request_pending:
            self.sort_and_display_tasks()


    # --- Calendar Interaction ---
    def clear_calendar_selection(self):
        if self.is_request_pending: return
        if self.selected_calendar_date is not None:
            self.selected_calendar_date = None
            self.calendar_widget.setSelectedDate(QDate()) 
            self.sort_and_display_tasks()

    def update_calendar_highlights(self):
        if hasattr(self, 'calendar_widget'):
            gui_components.update_calendar_display_highlights(
                self.calendar_widget, self.tasks_cache, self.highlighted_dates
            )
    
    def handle_calendar_date_clicked(self, q_date):
        if self.is_request_pending: return

        if self.selected_calendar_date == q_date:
            self.selected_calendar_date = None
            self.calendar_widget.setSelectedDate(QDate()) # Visually deselect
        else:
            self.selected_calendar_date = q_date
            # self.calendar_widget.setSelectedDate(q_date) is done by the click itself
        self.sort_and_display_tasks()


    # --- Task Actions (Add, Record, Edit, Delete) ---
    def send_task(self):
        task_desc = self.text_field.text().strip()
        if not task_desc:
            QMessageBox.warning(self, "Input Error", "Task description cannot be empty.")
            return
        if not self.logged_in_user:
            QMessageBox.warning(self, "Not Logged In", "Please log in to add tasks.")
            return
        if self.is_request_pending:
            QMessageBox.information(self, "Busy", "Processing another request. Please wait.")
            return
        self.set_busy_status("Adding task...")
        self.request_network_action.emit({'action': 'add_task', 'task_desc': task_desc})
        self.text_field.clear()

    @pyqtSlot(bool, str)
    def _handle_record_button_update(self, enabled, text):
        self.record_btn.setEnabled(enabled)
        self.record_btn.setText(text)

    def record_task(self):
        """Initiates voice recording for a task."""
        if self.is_request_pending:
            QMessageBox.information(self, "Busy", "Cannot start recording while another action is pending.")
            return
        
        # Update button state immediately (main thread)
        self.update_record_button_signal.emit(False, "Recording...")
        # Start the recording and recognition in a separate thread
        threading.Thread(target=self._record_task_thread_worker, daemon=True).start()

    def _record_task_thread_worker(self):
        """Worker function for speech recognition. Runs in a separate thread."""
        recognized_text = None
        error_message = None
        try:
            with sr.Microphone() as source:
                self.update_status_text_signal.emit("Status: Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.7)
                self.update_status_text_signal.emit("Status: Listening...")
                print("DEBUG (FlowlyApp Speech): Listening...")
                # Listen for the first phrase and extract it into audio data
                # Increased timeout and phrase_time_limit for potentially longer task descriptions
                audio = self.recognizer.listen(source, timeout=7, phrase_time_limit=20)
            
            self.update_status_text_signal.emit("Status: Recognizing...")
            print("DEBUG (FlowlyApp Speech): Processing audio...")
            recognized_text = self.recognizer.recognize_google(audio).lower()
            print(f"DEBUG (FlowlyApp Speech): Recognized: {recognized_text}")

        except sr.WaitTimeoutError:
            error_message = "No speech detected within the time limit."
            print("DEBUG (FlowlyApp Speech): WaitTimeoutError")
        except sr.UnknownValueError:
            error_message = "Google Speech Recognition could not understand audio."
            print("DEBUG (FlowlyApp Speech): UnknownValueError")
        except sr.RequestError as e:
            error_message = f"Could not request results from Google Speech Recognition service; {e}"
            print(f"DEBUG (FlowlyApp Speech): RequestError: {e}")
        except Exception as e: # Catch any other unexpected errors during recording/recognition
            error_message = f"An unexpected error occurred during voice recording: {e}"
            print(f"ERROR (FlowlyApp Speech): Unexpected error: {e}")
        finally:
            if recognized_text:
                self.update_task_text_field_signal.emit(recognized_text) # Update the text field
                self.update_status_text_signal.emit("Status: Voice input captured.")
            if error_message:
                self.update_status_text_signal.emit(f"Status: {error_message}")
            
            # Signal to re-enable the record button and reset its text
            self.update_record_button_signal.emit(True, "Record")
    
    @pyqtSlot(QPoint)
    def show_task_context_menu(self, pos):
        list_item = self.task_list.itemAt(pos)
        if not list_item: return
        
        task_widget_instance = self.task_list.itemWidget(list_item)
        if not task_widget_instance or not hasattr(task_widget_instance, 'task_id'):
             # Fallback to UserRole if itemWidget is not used or doesn't have task_id
            task_id = list_item.data(Qt.UserRole)
            if task_id is None:
                print("WARN (FlowlyApp): Context menu on item without TaskID.")
                return
        else:
            task_id = task_widget_instance.task_id


        global_pos = self.task_list.mapToGlobal(pos)
        menu = QMenu(self) # Parent menu to self
        edit_action = menu.addAction("Edit Task")
        delete_action = menu.addAction("Delete Task")
        
        # Disable edit if task is done (already handled by TaskItemWidget, but good for context menu too)
        # This requires fetching the task status if not readily available on task_widget_instance
        # For simplicity, we assume TaskItemWidget's edit button enabling is sufficient visual cue.

        action = menu.exec_(global_pos)
        if action == edit_action:
            self.request_edit_task(task_id)
        elif action == delete_action:
            self.request_delete_task(task_id)

    @pyqtSlot(int)
    def request_delete_task(self, task_id):
        if self.is_request_pending:
            QMessageBox.information(self, "Busy", "Cannot delete task now, another action is in progress.")
            return
        confirm = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to permanently delete task {task_id}?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.set_busy_status(f"Deleting task {task_id}...")
            self.request_network_action.emit({'action': 'delete_task', 'task_id': task_id})

    @pyqtSlot(int)
    def request_edit_task(self, task_id):
        if self.is_request_pending:
            QMessageBox.information(self, "Busy", "Cannot edit task now, another action is in progress.")
            return
        
        # Check if this task is 'done', if so, don't allow editing (optional, depends on desired UX)
        # This would require finding the task in tasks_cache to check its status.
        # for task in self.tasks_cache:
        #     if task[0] == task_id and task[5] == 'done':
        #         QMessageBox.information(self, "Edit Disabled", "Completed tasks cannot be edited.")
        #         return

        self._editing_task_id = task_id 
        self.set_busy_status(f"Fetching task {task_id} details for editing...")
        self.request_network_action.emit({'action': 'get_task', 'task_id': task_id})

    def show_edit_dialog_with_data(self, task_data):
        task_id_from_data = task_data.get('TaskID')
        
        if not hasattr(self, '_editing_task_id') or self._editing_task_id != task_id_from_data:
             print(f"WARN (FlowlyApp): Mismatched task ID for edit. Expected {self._editing_task_id}, got {task_id_from_data}.")
             self.on_request_finished() # Reset busy state
             self._editing_task_id = None
             return

        dialog, edit_fields, btn_box = gui_components.create_edit_task_dialog(self, task_data)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        
        original_editing_task_id = self._editing_task_id # Store before clearing
        self._editing_task_id = None # Clear flag, dialog will handle the rest or cancel

        if dialog.exec_() == QDialog.Accepted:
            updated_desc = edit_fields['desc'].text().strip()
            if not updated_desc:
                QMessageBox.warning(self, "Input Error", "Task description cannot be empty.")
                self.on_request_finished() # Ensure UI is re-enabled if validation fails here
                return 

            update_data = {
                'TaskDesc': updated_desc,
                'Date': edit_fields['date'].text().strip() or None, # Empty string becomes None
                'Time': edit_fields['time'].text().strip() or None,
                'Category': edit_fields['category'].currentText().strip() or "Personal", # Default if empty
                'Status': edit_fields['status'].currentText()
            }
            
            if self.is_request_pending: # Should ideally not happen if on_request_finished was called
                QMessageBox.information(self, "Busy", "Cannot update task now, another action is in progress.")
                return
            self.set_busy_status(f"Saving changes to task {original_editing_task_id}...")
            self.request_network_action.emit({
                'action': 'update_task', 
                'task_id': original_editing_task_id, 
                'update_data': update_data
            })
        else: # Dialog was cancelled or closed
            print(f"DEBUG (FlowlyApp): Edit dialog for task {original_editing_task_id} cancelled.")
            # Ensure request state is reset if no other request is pending
            if not self.is_request_pending:
                 self.on_request_finished()


    # --- Session Management ---
    def handle_logout(self):
        if self.logged_in_user:
            confirm = QMessageBox.question(self, "Confirm Logout", 
                                           f"Are you sure you want to log out, {self.logged_in_user}?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if confirm == QMessageBox.Yes:
                print(f"INFO (FlowlyApp): Logging out user {self.logged_in_user}")
                self.logged_in_user = None
                self.set_worker_user.emit("") 
                
                self.tasks_cache = []
                self.task_list.clear() # Visually clear list
                self.populate_task_list([]) # Show "No tasks" message
                self.selected_calendar_date = None
                self.highlighted_dates.clear() 
                if hasattr(self, 'calendar_widget'): self.update_calendar_highlights() 
                
                self.update_task_list_label_text([]) # Reset label with 0 tasks
                self.setWindowTitle(APP_TITLE_BASE)
                self.update_status_text_signal.emit("Status: Logged out successfully.")
                
                self.hide() 
                QTimer.singleShot(100, self.showLogin) # Small delay before showing login

    # --- Application Exit ---
    def closeEvent(self, event):
        print("INFO (FlowlyApp): Application close event triggered.")
        # Ask for confirmation if user is logged in and has pending tasks (optional)
        # if self.logged_in_user and any(task[5] == 'pending' for task in self.tasks_cache):
        #     reply = QMessageBox.question(self, 'Confirm Exit',
        #                                  "You have pending tasks. Are you sure you want to exit?",
        #                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        #     if reply == QMessageBox.No:
        #         event.ignore()
        #         return
        
        if self.network_thread.isRunning():
            print("INFO (FlowlyApp): Requesting network worker to close connection...")
            # Use invokeMethod if close_connection is a slot and worker is in another thread
            # QMetaObject.invokeMethod(self.network_worker, "close_connection", Qt.BlockingQueuedConnection)
            # Or ensure close_connection is thread-safe and call directly if safe.
            # For simplicity, assuming close_connection can be called; worker uses a lock.
            self.network_worker.close_connection()

            print("INFO (FlowlyApp): Quitting network thread...")
            self.network_thread.quit() 
            if not self.network_thread.wait(3000): # Wait up to 3 seconds
                 print("WARN (FlowlyApp): Network thread did not quit gracefully. Terminating.")
                 self.network_thread.terminate() # Force terminate if necessary
            else:
                 print("INFO (FlowlyApp): Network thread stopped.")
        
        event.accept()


# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Attempt to load stylesheet
    try:
        with open(STYLESHEET_PATH, "r") as file:
            app.setStyleSheet(file.read())
            print(f"INFO: Stylesheet '{STYLESHEET_PATH}' loaded.")
    except FileNotFoundError:
        print(f"WARN: Stylesheet '{STYLESHEET_PATH}' not found. Using default Qt styling.")
    except Exception as e:
        print(f"ERROR: Could not load stylesheet '{STYLESHEET_PATH}': {e}")

    window = FlowlyApp()
    # window.show() # showLogin will handle initial visibility or main window show
    sys.exit(app.exec_())