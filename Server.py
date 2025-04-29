import socket
import threading
from database import Database
import json
import pickle
import dateparser as dp
import bcrypt

import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict


# ---------------------------------------- LOADING NER MODEL ----------------------------------------
# Model definition
class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # BiLSTM doubles hidden size
        self.crf = CRF(num_classes)

    def forward(self, x, tags=None, mask=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.fc(x)

        if tags is not None:  # Training
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:  # Prediction
            return self.crf.viterbi_decode(emissions, mask=mask)

MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 100
VOCAB_SIZE = 8000
HIDDEN_DIM = 64
LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
NUM_CLASSES = len(LABELS)
NerModel = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
NerModel.load_state_dict(torch.load('NERModel.pth'))
NerModel.eval()


# Load label mappings
label2idx = {label: idx for idx, label in enumerate(LABELS)}
idx2label = {idx: label for label, idx in label2idx.items()}

# Load tokenizer (add this after model loading)
with open('NERtokenizer.pkl', 'rb') as f:  # saved during training
    tokenizer_dict = pickle.load(f)
tokenizer = defaultdict(lambda: 1, tokenizer_dict)

def NER_predict(sentence):
    print("start predicting")
    tokens = [tokenizer[word.strip().lower()] for word in sentence.split()]
    padded = tokens + [0] * (MAX_SEQUENCE_LENGTH - len(tokens))
    input_tensor = torch.tensor([padded], dtype=torch.long)
    mask = (input_tensor != 0)
    with torch.no_grad():
        preds = NerModel(input_tensor, mask=mask)[0]  # CRF decode returns list
    return [idx2label[idx] for idx in preds[:len(tokens)]]
    

# ---------------------------------------- SERVER ----------------------------------------

class TaskServer:
    def __init__(self, host='127.0.0.1', port=4320):
        self.host = host
        self.port = port
        self.db = Database()
        self.db.debug_print_all_users()
        self._setup_server()
        self.active_users = {} # {username: userid}
        self._ensure_test_user_exists()

    def _ensure_test_user_exists(self):
        # Check if 'test' user exists, create if not
        user = self.db.get_user_by_username('test')
        if not user:
            # For testing, create 'test' user with dummy data
            password_hash = bcrypt.hashpw('testpass'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            self.db.create_user('test', 'test@example.com', password_hash)
            print("Created 'test' user with password 'testpass' for testing.")

    def _setup_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allow reuse of local addresses
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print(f"Server listening on {self.host}:{self.port}")

    def handle_client(self, conn):
        print("starting handling")
        try:
            while True:
                data = conn.recv(1024).decode('utf-8')
                if not data:
                    print("client disconnected")
                    break
                
                try:
                    request = json.loads(data)
                    print(request)
                    action = request.get('action')

                    if action == 'login':
                        username = request.get('user')
                        password = request.get('password')
                        print(f"DEBUG: Login attempt for username: '{username}'")  # Debug print

                        user_info = self.db.get_user_by_username(username)
                        print(f"DEBUG: Retrieved user info: {user_info}")  # Debug print

                        if user_info:  # User exists
                            stored_hash = user_info[2]
                            if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                                user_id = user_info[0]  # UserID is first column
                                username_from_db = user_info[1]  # Actual username from DB
                                print(f"DEBUG: Logging in user ID: {user_id}, Username: {username_from_db}")
                                self.active_users[username_from_db] = user_id
                                conn.send(json.dumps({'status': 'success', 'username':username_from_db}).encode('utf-8'))
                            else:
                                conn.send(json.dumps({'status': 'failed', 'message': 'Invalid password'}).encode('utf-8'))
                        else:
                            error_msg = f"User '{username}' not found"
                            print(f"DEBUG: {error_msg}")
                            conn.send(json.dumps({'status': 'failed', 'message': error_msg}).encode('utf-8'))
                        continue
                    if action == 'signup':
                        username = request.get('username')
                        email = request.get('email')
                        password = request.get('password')

                        existing_user = self.db.get_user_by_username(username)
                        existing_email = self.db.get_user_by_email(email)

                        if existing_user:
                            conn.send(json.dumps({'status':'error', 'message':'Username already exists'}).encode('utf-8'))
                            continue
                        elif existing_email:
                            conn.send(json.dumps({'status':'error', 'message':'Email already exists'}).encode('utf-8'))
                            continue
                        else:
                            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                            user_id = self.db.create_user(username, email, password_hash)
                            conn.send(json.dumps({'status':'success', 'user_id':user_id}).encode('utf-8'))
                    elif action == 'get_task':
                        task_id = request.get('task_id')
                        task = self.db.get_task(task_id)
                        if task:
                            conn.send(json.dumps({'status': 'success', 'task': task}).encode('utf-8'))
                        else:
                            conn.send(json.dumps({'status': 'error', 'message': 'Task not found'}).encode('utf-8'))
                        continue
                    if action == 'add_task':
                        task_desc = request.get('task_desc')
                        print("got data!", task_desc)
                        # Process with NER
                        prediction = NER_predict(task_desc)
                        print(prediction)
                        DateExp = ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Date', 'I-Date']])
                        Date = dp.parse(DateExp, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'}).date() if DateExp else ''
                        TimeExp = ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Time', 'I-Time']])
                        Time = dp.parse(TimeExp, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'}).time() if TimeExp else ''
                        task_data = {
                            'TaskDesc': ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Task', 'I-Task']]),
                            'Date': str(Date),
                            'Time': str(Time),
                            'Category': 'General', # change when category prediction is implemented
                            'Urgency': 3 # change when urgency prediction is implemented
                        }
                        print(task_data)

                        user_id = self.active_users.get(username)
                        if not user_id:
                            conn.send(json.dumps({'status':'error','message':'User not logged in'}).encode('utf-8'))
                            continue
                        # Store in database
                        self.db.create_task(
                            user_id=user_id,
                            **task_data
                        )

                        print("task stored in database!")
                        conn.send(json.dumps({'status':'task_added'}).encode('utf-8'))
                    elif action == 'get_tasks':
                        tasks = self.db.get_tasks(user_id = self.active_users[username])
                        conn.send(json.dumps({'tasks':tasks}).encode('utf-8'))
                    elif action == 'update_status':
                        task_id = request.get('task_id')
                        status = request.get('status')
                        self.db.update_task_status(task_id, status)
                        conn.send(json.dumps({'status': 'status_updated'}).encode('utf-8'))
                    elif action == 'update_task':
                        task_id = request.get('task_id')
                        update_data = request.get('update_data')
                        user_id = self.active_users.get(username)
                        task = self.db.get_task(task_id)
                        if task and task[1] == user_id:
                            self.db.update_task(task_id, **update_data)
                            conn.send(json.dumps({'status':'task_updated'}).encode('utf-8'))
                        else:
                            conn.send(json.dumps({'status':'error', 'message':'Unauthorized'}).encode('utf-8'))
                    elif action == 'delete_task':
                        task_id = request.get('task_id')
                        user_id = self.active_users.get(username)
                        task = self.db.get_task(task_id)
                        if task and task[1] == user_id:
                            self.db.delete_task(task_id)
                            conn.send(json.dumps({'status':'task_deleted'}).encode('utf-8'))
                        else:
                            conn.send(json.dumps({'status':'error', 'message':'Unauthorized'}).encode('utf-8'))
                except json.JSONDecodeError:
                    conn.send(json.dumps({'error':'Invalid request format'}).encode('utf-8'))
        except Exception as e:
            conn.send(json.dumps({'error':str(e)}).encode('utf-8'))
        finally:
            conn.close()
            print("connection closed")

    def run(self):
        while True:
            conn, addr = self.sock.accept()
            print(f"Connected by {addr}")
            client_thread = threading.Thread(
                target=self.handle_client, args=(conn,)
            )
            client_thread.start()

if __name__ == "__main__":
    server = TaskServer()
    server.run()
