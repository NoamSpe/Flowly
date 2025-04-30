import torch
import torch.nn as nn
from TorchCRF import CRF
from collections import defaultdict
import pickle
import numpy as np

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

class ModelsLoader():
    def __init__(self):
        self.model = None
        self.paths = {
            'ner_model': 'NERModel.pth',
            'ner_tokenizer': 'NERTokenizer.pkl',
            'cat_classifier': 'CategoryClassification/CategoryClassifier_Model.pkl',
            'cat_vectorizer': 'CategoryClassification/CategoryClassifier_Vectorizer.pkl'
        }
        self.LABELS = ['O', 'B-Task', 'I-Task', 'B-Date', 'I-Date', 'B-Time', 'I-Time']
        self.label2idx = {label: idx for idx, label in enumerate(self.LABELS)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.load_ner_model()
        self.load_ner_tokenizer()
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
    def load_ner_tokenizer(self):
        with open('NERtokenizer.pkl', 'rb') as f:  # saved during training
            tokenizer_dict = pickle.load(f)
        self.tokenizer = defaultdict(lambda: 1, tokenizer_dict)
    def load_cat_classifier(self):
        with open('CategoryClassification/CategoryClassifier_Model.pkl', 'rb') as f:
            self.CatClassifier = pickle.load(f)
        with open('CategoryClassification/CategoryClassifier_Vectorizer.pkl', 'rb') as f:
            self.CatVectorizer = pickle.load(f)

    def NER_predict(self, sentence):
        print("start predicting")
        tokens = [self.tokenizer[word.strip().lower()] for word in sentence.split()]
        padded = tokens + [0] * (self.MAX_SEQUENCE_LENGTH - len(tokens))
        input_tensor = torch.tensor([padded], dtype=torch.long)
        mask = (input_tensor != 0)
        with torch.no_grad():
            preds = self.NerModel(input_tensor, mask=mask)[0]  # CRF decode returns list
        return [self.idx2label[idx] for idx in preds[:len(tokens)]]
    def category_predict(self, task_desc):
        vectorized_task_desc = self.CatVectorizer.transform([task_desc])
        predicted_category = self.CatClassifier.predict(vectorized_task_desc)
        predicted_category = None if np.max(self.CatClassifier.predict_proba(vectorized_task_desc)) < 0.3 else str(predicted_category[0])
        return predicted_category
