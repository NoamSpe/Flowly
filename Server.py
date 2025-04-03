import socket
import threading
from database import Database
import dateparser as dp

import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict
import pickle

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
        self._setup_server()

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
                task_desc = conn.recv(1024).decode('utf-8')
                if not task_desc:
                    print("client disconnected")
                    break

                print("got data!", task_desc)

                # Process with NER
                prediction = NER_predict(task_desc)
                print(prediction)
                DateExp = ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Date', 'I-Date']])
                if DateExp: Date = dp.parse(DateExp, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'}).date()
                else: Date = ''
                TimeExp = ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Time', 'I-Time']])
                if TimeExp: Time = dp.parse(TimeExp, languages=['en'], settings={'DATE_ORDER': 'DMY', 'PREFER_DATES_FROM': 'future'}).time()
                else: Time = ''
                task_data = {
                    'TaskDesc': ' '.join([word for ind, word in enumerate(task_desc.split(' ')) if prediction[ind] in ['B-Task', 'I-Task']]),
                    'Date': str(Date),
                    'Time': str(Time),
                    'Category': 'General', # change when category prediction is implemented
                    'Urgency': 3 # change when urgency prediction is implemented
                }
                print(task_data)

                # Store in database
                self.db.create_task(
                    user_id=int(1),
                    **task_data
                )
                print("task stored in database!")

                conn.send(b"Task processed successfully")
        except Exception as e:
            conn.send(f"Error: {str(e)}".encode())
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
