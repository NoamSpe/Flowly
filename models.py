import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict
import pickle
import numpy as np

class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        
        super(BiLSTM_NER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # word at index 0 in the vocabulary is padding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True) # batch first - (batch, seq, feature) instead of (seq, batch, feature)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # BiLSTM doubles hidden size
        self.crf = CRF(num_classes)

    def forward(self, x, tags=None, mask=None):
        x = self.embedding(x)
        x, _ = self.lstm(x) # _ is the hidden state
        x = self.dropout(x)
        emissions = self.fc(x)

        if tags is not None:  # Training
            loss = -self.crf(emissions, tags, mask=mask) # CRF loss is negative log likelihood
            return loss
        else:  # Prediction
            return self.crf.viterbi_decode(emissions, mask=mask)

class ModelsLoader():
    def __init__(self):
        self.paths = {
            'ner_model': 'NER/NERModel.pth',
            'ner_vocabulary': 'NER/NERVocabulary.pkl',
            'cat_classifier': 'CategoryClassification/CategoryClassifier_Model.pkl',
            'cat_vectorizer': 'CategoryClassification/CategoryClassifier_Vectorizer.pkl'
        }
        self.LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
        self.label2idx = {label: idx for idx, label in enumerate(self.LABELS)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.load_ner_model()
        self.load_ner_vocabulary()
        self.load_cat_classifier()

    def load_ner_model(self):
        self.MAX_SEQUENCE_LENGTH = 25
        EMBEDDING_DIM = 100
        VOCAB_SIZE = 8000
        HIDDEN_DIM = 64
        NUM_CLASSES = len(self.LABELS)
        self.NerModel = BiLSTM_NER(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
        self.NerModel.load_state_dict(torch.load(self.paths['ner_model']))
        self.NerModel.eval()
    def load_ner_vocabulary(self):
        with open(self.paths['ner_vocabulary'], 'rb') as f:  # saved during training
            vocabulary_dict = pickle.load(f)
        self.vocab = defaultdict(lambda: 1, vocabulary_dict)
    
    def load_cat_classifier(self):
        with open(self.paths['cat_classifier'], 'rb') as f:
            self.CatClassifier = pickle.load(f)
        with open(self.paths['cat_vectorizer'], 'rb') as f:
            self.CatVectorizer = pickle.load(f)

    def NER_predict(self, sentence):
        print("start predicting")
        tokens = [self.vocab[word.strip().lower()] for word in sentence.split()]
        padded = tokens + [0] * (self.MAX_SEQUENCE_LENGTH - len(tokens)) # pad to max length
        input_tensor = torch.tensor([padded], dtype=torch.long) # required format for input to neural network
        mask = (input_tensor != 0)
        with torch.no_grad(): # ensure inference mode - no gradients are computed
            preds = self.NerModel(input_tensor, mask=mask)[0]  # CRF decode returns list
        return [self.idx2label[idx] for idx in preds[:len(tokens)]]
    
    def category_predict(self, task_desc):
        vectorized_task_desc = self.CatVectorizer.transform([task_desc]) # vectorize the given task description
        predicted_category = self.CatClassifier.predict(vectorized_task_desc)
        predicted_category = None if np.max(self.CatClassifier.predict_proba(vectorized_task_desc)) < 0.4 else str(predicted_category[0])
        return predicted_category
