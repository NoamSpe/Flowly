# Server.py
import socket
import threading
import json
import pickle
import dateparser as dp
import bcrypt
import ssl
import numpy as np

import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict

from database import Database
from models import ModelsLoader

# Models loading
models = ModelsLoader()


class TaskServer:
    def __init__(self, host='0.0.0.0', port=4320):
        self.host = host
        self.port = port
        self.db = Database()
        self._setup_server()
        self.active_users = {} # {username: userid}

        # SSL context
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ssl_context.load_cert_chain(certfile='server.crt', keyfile='server.key')

    def _setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind((self.host, self.port))
        except OSError as e:
             print(f"Error binding to {self.host}:{self.port} - {e}")
             print("Is another instance running? Exiting.")
             exit()
        self.sock.listen()
        print(f"Server listening on {self.host}:{self.port}")

    def handle_client(self, client, addr): # addr for logging
        print(f"Handling connection from {addr}")
        current_user_id = None
        current_username = None
        conn = None
        try:
            # Wrap socket with SSL
            conn = self.ssl_context.wrap_socket(client, server_side=True)
            print(f"SSL handshake completed with {addr}")

            while True:
                # Buffer until newline
                buffer = b""
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                        raise ConnectionResetError("Client disconnected")
                    buffer += chunk
                    # check for newline (message delimiter)
                    if b'\n' in buffer:
                        message_data, buffer = buffer.split(b'\n', 1)
                        message_str = message_data.decode('utf-8').strip()
                        if message_str: # not processing empty strings
                            break # got a message
                    if len(buffer) > 1024 * 10: # 10KB limit
                        print(f"Buffer exceeded limit for {addr}. Closing connection.")
                        conn.close()
                        return # exit handler

                if not message_str:
                     print(f"Received empty message or only whitespace from {addr}, continuing...")
                     continue

                try:
                    request = json.loads(message_str)
                    print(f"Received from {addr}: {request}")
                    action = request.get('action')

                    # Actions accessible without login
                    if action == 'login':
                        username_attempt = request.get('user')
                        password_attempt = request.get('password', '').encode('utf-8') # Get password as bytes

                        if not username_attempt or not password_attempt:
                            self._send_response(conn, {'status': 'failed', 'message': 'Username and password required'})
                            continue

                        print(f"Login attempt for username: '{username_attempt}' from {addr}")
                        user_info = self.db.get_user_by_username(username_attempt)
                        print(f"Retrieved user info: {user_info} for {addr}")

                        if user_info:
                            user_id, db_username, stored_hash_str = user_info
                            stored_hash = stored_hash_str.encode('utf-8') # Get hash as bytes

                            # Verify Password
                            if bcrypt.checkpw(password_attempt, stored_hash):
                                print(f"Login successful for User ID: {user_id}, Username: {db_username} from {addr}")
                                self.active_users[db_username] = user_id # track active user
                                current_user_id = user_id # Track for this connection
                                current_username = db_username
                                self._send_response(conn, {'status': 'success', 'username': db_username, 'action_echo':'login'})
                            else:
                                print(f"Invalid password for username: '{username_attempt}' from {addr}")
                                self._send_response(conn, {'status': 'failed', 'message': 'Invalid username or password', 'action_echo':'login'})
                        else:
                            error_msg = f"User '{username_attempt}' not found"
                            print(f"{error_msg} for {addr}")
                            self._send_response(conn, {'status': 'failed', 'message': 'Invalid username or password', 'action_echo':'login'}) # Generic message for security
                        continue # move to next message

                    elif action == 'signup':
                        username = request.get('username')
                        password = request.get('password')

                        if not all([username, password]):
                             self._send_response(conn, {'status':'error', 'message':'Missing username or password'})
                             continue

                        # Check for existing username
                        existing_user = self.db.get_user_by_username(username)

                        if existing_user:
                            self._send_response(conn, {'status':'error', 'message':'Username already exists'})
                            continue

                        try:
                            # Hash the password
                            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                            user_id = self.db.create_user(username, hashed_password.decode('utf-8')) # Store hash as string
                            print(f"User created: ID {user_id}, Username {username} from {addr}")
                            self._send_response(conn, {'status':'success', 'message': 'Signup successful! Please log in.', 'action_echo':'signup'}) # Don't send user_id back directly
                        except Exception as e:
                            print(f"Signup failed for {username} from {addr} - {e}")
                            self._send_response(conn, {'status':'error', 'message':'Signup failed. Please try again.', 'action_echo':'signup'})
                        continue # Move to next message

                    # Actions requiring login
                    # Check if user is logged in for next actions
                    if current_user_id is None:
                         print(f"Action '{action}' attempted without login from {addr}. Denying.")
                         self._send_response(conn, {'status': 'error', 'message': 'Not logged in'})
                         continue # skip to next message

                    request_username = request.get('user') # Client might still send username
                    if request_username != current_username:
                         print(f"Mismatched username in request ('{request_username}') vs logged in user ('{current_username}') for action '{action}' from {addr}. Using logged in user.")
                         # Proceed using current_username and current_user_id

                    if action == 'get_tasks':
                        tasks = self.db.get_tasks(user_id = current_user_id)
                        self._send_response(conn, {'status': 'success', 'tasks': tasks, 'action_echo':'get_tasks'})

                    elif action == 'add_task':
                        task_desc_raw = request.get('task_desc')
                        if not task_desc_raw:
                            self._send_response(conn, {'status':'error', 'message': 'Task description cannot be empty', 'action_echo':'add_task'})
                            continue

                        print(f"Got raw task description: '{task_desc_raw}' from {current_username} ({addr})")
                        try:
                            # Process with NER
                            prediction = models.NER_predict(task_desc_raw)
                            print(f"NER Prediction: {prediction}")

                            words = task_desc_raw.split(' ')
                            task_parts = [word for ind, word in enumerate(words) if prediction[ind] in ['B-Task', 'I-Task']]
                            date_parts = [word for ind, word in enumerate(words) if prediction[ind] in ['B-Date', 'I-Date']]
                            time_parts = [word for ind, word in enumerate(words) if prediction[ind] in ['B-Time', 'I-Time']]

                            task_desc_processed = ' '.join(task_parts)
                            date_str_raw = ' '.join(date_parts)
                            time_str_raw = ' '.join(time_parts)

                            # Parse Date/Time
                            date_obj = None
                            time_obj = None
                            try:
                                if date_str_raw:
                                     # parse date expression with dateparser
                                     parsed_dt = dp.parse(date_str_raw, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'})
                                     if parsed_dt: date_obj = parsed_dt.date() # have only the date data
                            except Exception as dp_err:
                                print(f"Date parsing failed for '{date_str_raw}': {dp_err}")

                            try:
                                if time_str_raw:
                                     # combined with date for context
                                     parse_context = f"{date_obj} {time_str_raw}" if date_obj else time_str_raw
                                     parsed_dt = dp.parse(parse_context, languages=['en'])
                                     if parsed_dt: time_obj = parsed_dt.time() # have only the time data
                            except Exception as dp_err:
                                print(f"Time parsing failed for '{time_str_raw}': {dp_err}")

                            # Classify task description for category
                            predicted_category = models.category_predict(task_desc_raw)

                            task_data = {
                                'TaskDesc': task_desc_processed if task_desc_processed else task_desc_raw, # back to raw description if NER fails
                                'Date': str(date_obj) if date_obj else None,
                                'Time': str(time_obj) if time_obj else None,
                                'Category': predicted_category, 
                                'Status': 'pending' # pending by default
                            }

                            self.db.create_task(user_id=current_user_id, **task_data) # Store the new task in the database
                            print(f"Task stored in database for user {current_user_id} ({current_username})!")
                            self._send_response(conn, {'status':'task_added', 'message': 'Task added successfully', 'action_echo':'add_task'})
                        except Exception as e:
                            print(f"Failed processing/adding task for user {current_user_id} ({current_username}): {e}")
                            self._send_response(conn, {'status':'error', 'message': f'Error adding task: {e}'})

                    elif action == 'update_task_status':
                        task_id = request.get('task_id')
                        status = request.get('status')
                        if not task_id or status not in ['pending', 'done']:
                            self._send_response(conn, {'status': 'error', 'message': 'Invalid task ID or status', 'action_echo':'update_task_status'})
                            continue
                        # Verify ownership before updating
                        task_info = self.db.get_task(task_id) # Fetch task to check UserID
                        if task_info and task_info[1] == current_user_id: # Assuming UserID is in index 1
                           self.db.update_task_status(task_id, status)
                           self._send_response(conn, {'status': 'status_updated', 'task_id': task_id, 'new_status': status, 'action_echo':'update_task_status'})
                        elif task_info:
                           print(f"WARN: User {current_user_id} tried to update status for task {task_id} owned by {task_info[1]} from {addr}")
                           self._send_response(conn, {'status': 'error', 'message': 'Unauthorized: You do not own this task', 'action_echo':'update_task_status'})
                        else:
                           self._send_response(conn, {'status': 'error', 'message': 'Task not found', 'action_echo':'update_task_status'})

                    elif action == 'update_task':
                        task_id = request.get('task_id')
                        update_data = request.get('update_data')
                        if not task_id or not isinstance(update_data, dict):
                             self._send_response(conn, {'status':'error', 'message':'Invalid task ID or update data format', 'action_echo':'update_task'})
                             continue

                        # Verify ownership
                        task_info = self.db.get_task(task_id)
                        if task_info and task_info[1] == current_user_id: # Check UserID at index 1
                            try:
                                self.db.update_task(task_id, **update_data)
                                self._send_response(conn, {'status':'task_updated', 'task_id': task_id, 'action_echo':'update_task'})
                            except Exception as e:
                                print(f"ERROR: Updating task {task_id} failed: {e}")
                                self._send_response(conn, {'status':'error', 'message':f'Error updating task: {e}', 'action_echo':'update_task'})
                        elif task_info:
                            print(f"WARN: User {current_user_id} tried to update task {task_id} owned by {task_info[1]} from {addr}")
                            self._send_response(conn, {'status':'error', 'message':'Unauthorized: You do not own this task', 'action_echo':'update_task'})
                        else:
                            self._send_response(conn, {'status': 'error', 'message': 'Task not found', 'action_echo':'update_task'})

                    elif action == 'delete_task':
                        task_id = request.get('task_id')
                        if not task_id:
                            self._send_response(conn, {'status':'error', 'message':'Task ID required', 'action_echo':'delete_task'})
                            continue
                        # Verify ownership
                        task_info = self.db.get_task(task_id)
                        if task_info and task_info[1] == current_user_id: # Check UserID at index 1
                            self.db.delete_task(task_id)
                            self._send_response(conn, {'status':'task_deleted', 'task_id': task_id, 'action_echo':'delete_task'})
                        elif task_info:
                            print(f"WARN: User {current_user_id} tried to delete task {task_id} owned by {task_info[1]} from {addr}")
                            self._send_response(conn, {'status':'error', 'message':'Unauthorized: You do not own this task', 'action_echo':'delete_task'})
                        else:
                            self._send_response(conn, {'status': 'error', 'message': 'Task not found', 'action_echo':'delete_task'})

                    elif action == 'get_task': # for fetching details for edit form
                        task_id = request.get('task_id')
                        if not task_id:
                             self._send_response(conn, {'status':'error', 'message':'Task ID required', 'action_echo':'get_task'})
                             continue
                        task_info = self.db.get_task(task_id)
                        if task_info and task_info[1] == current_user_id: # Check UserID at index 1
                             # Convert tuple to dict for easier client handling
                             task_dict = {
                                'TaskID': task_info[0],
                                'UserID': task_info[1],
                                'TaskDesc': task_info[2],
                                'Date': task_info[3],
                                'Time': task_info[4],
                                'Category': task_info[5],
                                'Status': task_info[6]
                             }
                             self._send_response(conn, {'status': 'success', 'task': task_dict, 'action_echo':'get_task'})
                        elif task_info:
                             self._send_response(conn, {'status':'error', 'message':'Unauthorized', 'action_echo':'get_task'})
                        else:
                             self._send_response(conn, {'status': 'error', 'message': 'Task not found', 'action_echo':'get_task'})

                    else:
                        print(f"WARN: Unknown action '{action}' received from {current_username or addr}")
                        self._send_response(conn, {'status':'error', 'message':f'Unknown action: {action}', 'action_echo':'get_task'})

                except json.JSONDecodeError:
                    print(f"ERROR: Invalid JSON received from {current_username or addr}: {message_str}")
                    self._send_response(conn, {'status':'error','message':'Invalid request format (bad JSON)'})
                except Exception as e: # any other error
                    print(f"ERROR: Exception processing request from {current_username or addr}: {e}")
                    self._send_response(conn, {'status':'error','message':f'Server error processing request: {e}'})

        except (ssl.SSLError, ConnectionResetError, BrokenPipeError) as e:
            print(f"Client {addr} SSL/connection error: {e}")
        except Exception as e:
            print(f"ERROR: Unhandled exception in client handler for {addr}: {e}")
            # Try sending a generic error if possible
            try:
                self._send_response(conn, {'status':'error','message':'An unexpected server error occurred.'})
            except:
                pass
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing SSL socket for {addr}: {e}")
            # remove active user entry if they were logged in
            if current_username and current_username in self.active_users:
                 # Check if the stored user_id matches the one for this session before deleting
                 if self.active_users[current_username] == current_user_id:
                    del self.active_users[current_username]
                    print(f"User '{current_username}' logged out from {addr}.")
                 else:
                    print(f"WARN: User '{current_username}' from {addr} disconnected, but active session belongs to a different connection.")

            conn.close()
            print(f"Connection closed for {addr}")

    def _send_response(self, conn, response_data):
        """Helper to send JSON response with newline delimiter."""
        try:
            message = json.dumps(response_data) + '\n' # Add newline
            conn.sendall(message.encode('utf-8'))
        except (OSError, ConnectionResetError, BrokenPipeError) as e:
            print(f"ERROR: Failed to send response: {e} - Data: {response_data}")
    
    def run(self):
        while True: # main server loop
            try:
                conn, addr = self.sock.accept() # accept a new connection
                print(f"Accepted connection from {addr}")
                client_thread = threading.Thread(
                    target=self.handle_client, args=(conn, addr), daemon=True
                ) # create a new thread for each client
                client_thread.start()
            except KeyboardInterrupt:
                print("\nShutting down server...")
                break
            except Exception as e:
                print(f"error accepting connection: {e}")

        self.sock.close()
        print("Server socket closed.")


if __name__ == "__main__":
    server = TaskServer()
    server.run()
