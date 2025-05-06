# Server.py
import socket
import threading
from database import Database
import json
import pickle
import dateparser as dp
import bcrypt # <-- Import bcrypt
import ssl
import numpy as np

import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict

from models import ModelsLoader

# Models definition
models = ModelsLoader()
# --- NER Model ---
# class BiLSTM_NER(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
#         super(BiLSTM_NER, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.dropout = nn.Dropout(0.25)
#         self.fc = nn.Linear(hidden_dim * 2, num_classes)  # BiLSTM doubles hidden size
#         self.crf = CRF(num_classes)

#     def forward(self, x, tags=None, mask=None):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         x = self.dropout(x)
#         emissions = self.fc(x)

#         if tags is not None:  # Training
#             loss = -self.crf(emissions, tags, mask=mask)
#             return loss
#         else:  # Prediction
#             return self.crf.viterbi_decode(emissions, mask=mask)

# MAX_SEQUENCE_LENGTH = 25
# EMBEDDING_DIM = 100
# VOCAB_SIZE = 8000
# HIDDEN_DIM = 64
# LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
# NUM_CLASSES = len(LABELS)
# NerModel = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
# NerModel.load_state_dict(torch.load('NERModel.pth'))
# NerModel.eval()


# # Load label mappings
# label2idx = {label: idx for idx, label in enumerate(LABELS)}
# idx2label = {idx: label for label, idx in label2idx.items()}

# # Load tokenizer (add this after model loading)
# with open('NERtokenizer.pkl', 'rb') as f:  # saved during training
#     tokenizer_dict = pickle.load(f)
# tokenizer = defaultdict(lambda: 1, tokenizer_dict)

# def NER_predict(sentence):
#     print("start predicting")
#     tokens = [tokenizer[word.strip().lower()] for word in sentence.split()]
#     padded = tokens + [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
#     input_tensor = torch.tensor([padded], dtype=torch.long)
#     mask = (input_tensor != 0)
#     with torch.no_grad():
#         preds = NerModel(input_tensor, mask=mask)[0]  # CRF decode returns list
#     return [idx2label[idx] for idx in preds[:len(tokens)]]
    
# # --- Category Classifier Model ---
# with open('CategoryClassification/CategoryClassifier_Model.pkl', 'rb') as f:
#     CatClassifier = pickle.load(f)
# with open('CategoryClassification/CategoryClassifier_Vectorizer.pkl', 'rb') as f:
#     CatVectorizer = pickle.load(f)


# ---------------------------------------- SERVER ----------------------------------------

class TaskServer:
    def __init__(self, host='127.0.0.1', port=4320):
        self.host = host
        self.port = port
        self.db = Database()
        self.db.debug_print_all_users()
        self._setup_server()
        self.active_users = {} # {username: userid}
        # self._ensure_test_user_exists() # Let's rely on signup/login for testing now

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

    def handle_client(self, client, addr): # Add addr for better logging
        print(f"Handling connection from {addr}")
        current_user_id = None
        current_username = None
        conn = None
        try:
            # Wrap socket with SSL
            conn = self.ssl_context.wrap_socket(client, server_side=True)
            print(f"SSL handshake completed with {addr}")

            while True:
                # Improved message receiving: Buffer until newline
                buffer = b""
                while True:
                    chunk = conn.recv(1024)
                    if not chunk:
                         # Client disconnected gracefully
                         raise ConnectionResetError("Client disconnected")
                    buffer += chunk
                    # Use newline as a simple message delimiter
                    if b'\n' in buffer:
                        message_data, buffer = buffer.split(b'\n', 1)
                        message_str = message_data.decode('utf-8').strip()
                        if message_str: # Avoid processing empty strings
                             break # Got a message
                    # Optional: Add a timeout or max buffer size here to prevent memory issues
                    if len(buffer) > 1024 * 10: # e.g., 10KB limit
                        print(f"Warning: Buffer exceeded limit for {addr}. Closing connection.")
                        conn.close()
                        return # Exit handler

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

                        print(f"DEBUG: Login attempt for username: '{username_attempt}' from {addr}")
                        user_info = self.db.get_user_by_username(username_attempt)
                        print(f"DEBUG: Retrieved user info: {user_info} for {addr}")

                        if user_info:
                            user_id, db_username, stored_hash_str = user_info
                            stored_hash = stored_hash_str.encode('utf-8') # Get hash as bytes

                            # *** Verify Password ***
                            if bcrypt.checkpw(password_attempt, stored_hash):
                                print(f"DEBUG: Login successful for User ID: {user_id}, Username: {db_username} from {addr}")
                                self.active_users[db_username] = user_id # Store mapping
                                current_user_id = user_id             # Track for this connection
                                current_username = db_username
                                self._send_response(conn, {'status': 'success', 'username': db_username, 'action_echo':'login'})
                            else:
                                print(f"DEBUG: Invalid password for username: '{username_attempt}' from {addr}")
                                self._send_response(conn, {'status': 'failed', 'message': 'Invalid username or password', 'action_echo':'login'})
                        else:
                            error_msg = f"User '{username_attempt}' not found"
                            print(f"DEBUG: {error_msg} for {addr}")
                            self._send_response(conn, {'status': 'failed', 'message': 'Invalid username or password', 'action_echo':'login'}) # Generic message for security
                        continue # Move to next message

                    elif action == 'signup':
                        username = request.get('username')
                        email = request.get('email')
                        password = request.get('password')

                        if not all([username, email, password]):
                             self._send_response(conn, {'status':'error', 'message':'Missing username, email, or password'})
                             continue

                        # Check for existing user/email (case-insensitive)
                        existing_user = self.db.get_user_by_username(username)
                        existing_email = self.db.get_user_by_email(email)

                        if existing_user:
                            self._send_response(conn, {'status':'error', 'message':'Username already exists'})
                            continue
                        if existing_email:
                            self._send_response(conn, {'status':'error', 'message':'Email already exists'})
                            continue

                        try:
                            # *** Hash the password ***
                            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                            user_id = self.db.create_user(username, email, hashed_password.decode('utf-8')) # Store hash as string
                            print(f"DEBUG: User created: ID {user_id}, Username {username} from {addr}")
                            self._send_response(conn, {'status':'success', 'message': 'Signup successful! Please log in.', 'action_echo':'signup'}) # Don't send user_id back directly
                        except Exception as e:
                            print(f"ERROR: Signup failed for {username} from {addr} - {e}")
                            self._send_response(conn, {'status':'error', 'message':'Signup failed. Please try again.', 'action_echo':'signup'})
                        continue # Move to next message

                    # --- Actions requiring login ---
                    # Check if user is logged in for subsequent actions
                    if current_user_id is None:
                         print(f"WARN: Action '{action}' attempted without login from {addr}. Denying.")
                         self._send_response(conn, {'status': 'error', 'message': 'Not logged in'})
                         continue # Skip to next message

                    # Get username associated with this connection's user_id if needed
                    request_username = request.get('user') # Client might still send username
                    if request_username != current_username:
                         print(f"WARN: Mismatched username in request ('{request_username}') vs logged in user ('{current_username}') for action '{action}' from {addr}. Using logged in user.")
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

                            # Parse Date/Time safely
                            date_obj = None
                            time_obj = None
                            try:
                                if date_str_raw:
                                     # Use fuzzy=True for better flexibility if needed
                                     parsed_dt = dp.parse(date_str_raw, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'})
                                     if parsed_dt: date_obj = parsed_dt.date()
                            except Exception as dp_err:
                                print(f"WARN: Date parsing failed for '{date_str_raw}': {dp_err}")

                            try:
                                if time_str_raw:
                                     # Combine with date if available for context, or parse time alone
                                     parse_context = f"{date_obj} {time_str_raw}" if date_obj else time_str_raw
                                     parsed_dt = dp.parse(parse_context, languages=['en'])
                                     if parsed_dt: time_obj = parsed_dt.time()
                            except Exception as dp_err:
                                print(f"WARN: Time parsing failed for '{time_str_raw}': {dp_err}")

                            # Classify for Category
                            predicted_category = models.category_predict(task_desc_raw)
                            # vectorized_task_desc = CatVectorizer.transform([task_desc_raw])
                            # predicted_category = CatClassifier.predict(vectorized_task_desc)
                            # predicted_category = None if np.max(CatClassifier.predict_proba(vectorized_task_desc)) < 0.3 else str(predicted_category[0])

                            task_data = {
                                'TaskDesc': task_desc_processed if task_desc_processed else task_desc_raw, # Fallback to raw if NER fails
                                'Date': str(date_obj) if date_obj else None,
                                'Time': str(time_obj) if time_obj else None,
                                'Category': predicted_category, 
                                # 'Urgency': request.get('Urgency', 3),   # Allow override or default
                                'Status': 'pending' # Always start as pending
                            }
                            print(f"Processed task data: {task_data}")

                            self.db.create_task(user_id=current_user_id, **task_data)
                            print(f"Task stored in database for user {current_user_id} ({current_username})!")
                            self._send_response(conn, {'status':'task_added', 'message': 'Task added successfully', 'action_echo':'add_task'})
                        except Exception as e:
                            print(f"ERROR: Failed processing/adding task for user {current_user_id} ({current_username}): {e}")
                            self._send_response(conn, {'status':'error', 'message': f'Error adding task: {e}'})


                    elif action == 'update_task_status': # Renamed from 'update_status' for clarity
                        task_id = request.get('task_id')
                        status = request.get('status')
                        if not task_id or status not in ['pending', 'done']:
                            self._send_response(conn, {'status': 'error', 'message': 'Invalid task ID or status', 'action_echo':'update_task_status'})
                            continue
                        # Verify ownership before updating
                        task_info = self.db.get_task(task_id) # Fetch task to check UserID
                        if task_info and task_info[1] == current_user_id: # Assuming UserID is the second column (index 1) in get_task result
                           self.db.update_task_status(task_id, status)
                           self._send_response(conn, {'status': 'status_updated', 'task_id': task_id, 'new_status': status, 'action_echo':'update_task_status'})
                        elif task_info:
                           print(f"WARN: User {current_user_id} tried to update status for task {task_id} owned by {task_info[1]} from {addr}")
                           self._send_response(conn, {'status': 'error', 'message': 'Unauthorized: You do not own this task', 'action_echo':'update_task_status'})
                        else:
                           self._send_response(conn, {'status': 'error', 'message': 'Task not found', 'action_echo':'update_task_status'})


                    elif action == 'update_task':
                        task_id = request.get('task_id')
                        update_data = request.get('update_data') # This should be a dict
                        if not task_id or not isinstance(update_data, dict):
                             self._send_response(conn, {'status':'error', 'message':'Invalid task ID or update data format', 'action_echo':'update_task'})
                             continue

                        # Verify ownership
                        task_info = self.db.get_task(task_id)
                        if task_info and task_info[1] == current_user_id: # Check UserID at index 1
                            try:
                                # Optional: Sanitize/validate update_data keys/values here
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

                    elif action == 'get_task': # Keep for fetching details for edit form
                        task_id = request.get('task_id')
                        if not task_id:
                             self._send_response(conn, {'status':'error', 'message':'Task ID required', 'action_echo':'get_task'})
                             continue
                        task_info = self.db.get_task(task_id)
                        if task_info and task_info[1] == current_user_id: # Check UserID at index 1
                             # Convert tuple to dict for easier client handling
                             task_dict = {
                                'TaskID': task_info[0],
                                'UserID': task_info[1], # Include UserID
                                'TaskDesc': task_info[2],
                                'Date': task_info[3],
                                'Time': task_info[4],
                                'Category': task_info[5],
                                # 'Urgency': task_info[6],
                                'Status': task_info[6]
                                # Add CreatedAt, UpdatedAt if needed
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
                except Exception as e: # Catch broader errors during request processing
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
                pass # Ignore if sending fails
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    print(f"Error closing SSL socket for {addr}: {e}")
            # Clean up active user entry if they were logged in
            if current_username and current_username in self.active_users:
                 # Check if the stored user_id matches the one for this session before deleting
                 if self.active_users[current_username] == current_user_id:
                    del self.active_users[current_username]
                    print(f"User '{current_username}' logged out from {addr}.")
                 else:
                     # This might happen if the user logs in again from elsewhere
                     print(f"WARN: User '{current_username}' from {addr} disconnected, but active session belongs to a different connection.")

            conn.close()
            print(f"Connection closed for {addr}")

    def _send_response(self, conn, response_data):
        """Helper to send JSON response with newline delimiter."""
        try:
            message = json.dumps(response_data) + '\n' # Add newline
            conn.sendall(message.encode('utf-8'))
            # print(f"Sent: {response_data}") # Optional: Debug logging
        except (OSError, ConnectionResetError, BrokenPipeError) as e:
            print(f"ERROR: Failed to send response: {e} - Data: {response_data}")
    
    def run(self):
        while True:
            try:
                conn, addr = self.sock.accept()
                print(f"Accepted connection from {addr}")
                client_thread = threading.Thread(
                    target=self.handle_client, args=(conn, addr), daemon=True # Use daemon threads
                )
                client_thread.start()
            except KeyboardInterrupt:
                 print("\nShutting down server...")
                 break
            except Exception as e:
                 print(f"ERROR accepting connection: {e}")

        self.sock.close()
        print("Server socket closed.")


if __name__ == "__main__":
    server = TaskServer()
    server.run()
