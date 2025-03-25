import asyncio
import websockets
import json
import torch
import torch.nn as nn
from TorchCRF import CRF

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

EMBEDDING_DIM = 100
VOCAB_SIZE = 8000
HIDDEN_DIM = 64
LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
NUM_CLASSES = len(LABELS)
NerModel = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
NerModel.load_state_dict(torch.load('NERModel.pth'))
NerModel.eval()


# ---------------------------------------- SERVER ----------------------------------------

# Simulate model prediction (replace with actual model)
def predict_categoty(task_name, task_date, task_time):
    category = "work"  # Example mock prediction
    return category

def predict_urgency(task_name, task_date, task_time, task_category):
    urgency = "medium"  # Example mock prediction
    return category

async def handle_task_description(websocket, path):
    try:
        # Receive data from the client (task details)
        task_data = await websocket.recv()
        task_labels = NerModel(task_data)

        task_name = [x for x in task_labels if x in ['B-Task', 'I-Task']]
        task_date = [x for x in task_labels if x in ['B-Date', 'I-Date']]
        task_time = [x for x in task_labels if x in ['B-Time', 'I-Time']]
        
        # Get predictions from the "model"
        task_category = predict_categoty (task_name, task_date, task_time)
        task_urgency = predict_categoty (task_name, task_date, task_time, task_urgency)
        
        # Create a response with predicted properties
        response = {
            "category": task_category,
            "urgency": task_urgency,
            "task_name": task_name,
            "task_date": task_date,
            "task_time": task_time
        }
        
        # Send the response back to the client
        await websocket.send(json.dumps(response))
        
    except Exception as e:
        # Handle errors
        print(f"Error: {e}")

# Start the WebSocket server
async def main():
    server = await websockets.serve(handle_task_request, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

# Run the WebSocket server
if __name__ == "__main__":
    asyncio.run(main())
