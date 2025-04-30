import socket
import ssl
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
import json
import threading
import time

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
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                context.load_verify_locations('server.crt')

                plain_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket = context.wrap_socket(plain_sock, server_hostname=self.host)

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
