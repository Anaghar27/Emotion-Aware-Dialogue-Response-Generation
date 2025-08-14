import os
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier:
    labels = []
    def __init__(self, ckpt_path=None):
        self.ckpt_path = ckpt_path
    def predict(self, text):
        return self.predict_heuristic(text)
    def predict_heuristic(self, text):
        raise NotImplementedError

#Bi-directional LSTM model for emotion classification.
class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        avg_pool = torch.mean(lstm_out, dim=1)
        dropped = self.dropout(avg_pool)
        output = self.fc(dropped)
        return self.softmax(output)
    
# Wrapper for the BiLSTMEmotionClassifier with vocabulary loading, encoding, and inference logic.
class EmotionClassifier(BaseClassifier):
    def __init__(self, ckpt_path=None, vocab_path=None, label2id=None,
                 max_len=50, lower=True,
                 embed_dim=128, hidden_dim=64, dropout=0.5,
                 pad_token="<PAD>", unk_token="<UNK>"):
        super().__init__(ckpt_path)

        # Label mappings
        self.label2id = label2id or {"no emotion":0,"anger":1,"disgust":2,"fear":3,"happiness":4,"sadness":5,"surprise":6}
        self.label2id = {k: int(v) for k, v in self.label2id.items()}
        self.id2label = {v: k for k, v in self.label2id.items()}


        self.vocab_path = vocab_path
        self.max_len = max_len
        self.lower = lower
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = None
        self.model = None
        self.pad_id = 0
        self.unk_id = 1
        self._load_all()

    def _load_all(self):
        if not self.vocab_path or not os.path.exists(self.vocab_path):
            print(f"[EmotionClassifier] Missing vocab at {self.vocab_path}")
            return

        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.unk_id = self.vocab.get(self.unk_token, 1)
        vocab_size = max(self.vocab.values()) + 1

        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            print(f"[EmotionClassifier] Missing checkpoint at {self.ckpt_path}")
            return

        output_dim = max(self.label2id.values()) + 1

        self.model = BiLSTMEmotionClassifier(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            pad_idx=self.pad_id
        ).to(self.device)

        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception as e:
            print("[EmotionClassifier] load_state_dict warning:", e)

        self.model.eval()
        print(f"[EmotionClassifier] Loaded {self.ckpt_path} (vocab={vocab_size}, labels={output_dim})")

    def _tok(self, s):
        if not s: return []
        t = s.lower() if self.lower else s
        return re.findall(r"[A-Za-z']+", t)

    def _encode(self, text):
        toks = self._tok(text)
        ids = [self.vocab.get(tok, self.unk_id) for tok in toks][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return x

    def predict(self, text):
        if self.model is None or self.vocab is None:
            print(f"[DEBUG] EmotionClassifier fallback (heuristic) for: {text!r}")
            return self.predict_heuristic(text)

        with torch.no_grad():
            print(f"[DEBUG] EmotionClassifier model called for: {text!r}")
            x = self._encode(text)
            print(f"[DEBUG] Emotion token IDs: {x.tolist()}")
            logp = self.model(x)              # (1, C) log-probs
            print(f"[DEBUG] Emotion log-probs: {logp.tolist()}")
            idx = int(torch.argmax(logp, dim=1).item())
            pred = self.id2label.get(idx, str(idx))
            print(f"[DEBUG] Emotion predicted: {pred}")
            return pred


    def predict_heuristic(self, text):
        t = (text or "").lower()
        if any(w in t for w in ["angry","mad","furious","annoyed","irritated"]): return "0"
        if any(w in t for w in ["sad","upset","depressed","cry","alone","unhappy"]): return "1"
        if any(w in t for w in ["happy","great","awesome","love","glad","yay"]): return "2"
        if any(w in t for w in ["scared","afraid","worried","anxious","panic"]): return "3"
        if any(w in t for w in ["wow","surprised","unexpected","no way","really?"]): return "4"
        return "5"

# LSTM model for dialogue act classification.
class LSTMActClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        emb = self.embedding(x)            
        out, _ = self.lstm(emb)            
        mean_pool = out.mean(dim=1)        
        dropped = self.dropout(mean_pool)  
        logits = self.fc(dropped)          
        return self.log_softmax(logits)    

# Wrapper for LSTMActClassifier with vocabulary loading, encoding, and inference logic for dialogue act classification.
class ActClassifier(BaseClassifier):
    def __init__(self, ckpt_path=None, vocab_path=None, label2id=None,
                 max_len=50, lower=True,
                 embed_dim=128, hidden_dim=64, dropout=0.5,
                 pad_token="<PAD>", unk_token="<UNK>"):
        super().__init__(ckpt_path)
        self.vocab_path = vocab_path
        self.label2id = label2id or {"other":0,"inform":1,"question":2,"directive":3,"commissive":4}
        self.id2label = {v:k for k,v in self.label2id.items()}

        self.max_len = max_len
        self.lower = lower
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = None
        self.model = None
        self.pad_id = 0
        self.unk_id = 1
        self._load_all()

    def _load_all(self):
        if not self.vocab_path or not os.path.exists(self.vocab_path):
            print(f"[ActClassifier] Missing vocab at {self.vocab_path}.")
            return
        with open(self.vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.pad_id = self.vocab.get(self.pad_token, 0)
        self.unk_id = self.vocab.get(self.unk_token, 1)
        vocab_size = max(self.vocab.values()) + 1

        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            print(f"[ActClassifier] Missing checkpoint at {self.ckpt_path}.")
            return

        output_dim = max(self.label2id.values()) + 1
        self.model = LSTMActClassifier(
            vocab_size=vocab_size,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            output_dim=output_dim,
            pad_idx=self.pad_id,
            dropout=self.dropout
        ).to(self.device)

        state = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception as e:
            print("[ActClassifier] load_state_dict warning:", e)
        self.model.eval()
        print(f"[ActClassifier] Loaded {self.ckpt_path} (vocab={vocab_size}, labels={output_dim})")

    def _tok(self, s):
        if not s: return []
        t = s.lower() if self.lower else s
        return re.findall(r"[A-Za-z']+", t)

    def _encode(self, text):
        toks = self._tok(text)
        ids = [self.vocab.get(tok, self.unk_id) for tok in toks][:self.max_len]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        return x

    def predict(self, text):
        if self.model is None or self.vocab is None:
            print(f"[DEBUG] ActClassifier fallback (heuristic) for: {text!r}")
            return self.predict_heuristic(text)

        with torch.no_grad():
            print(f"[DEBUG] ActClassifier model called for: {text!r}")
            x = self._encode(text)
            print(f"[DEBUG] Token IDs: {x.tolist()}")
            logp = self.model(x)           # (1, C) log-probs
            print(f"[DEBUG] Log probabilities: {logp.tolist()}")
            idx = int(torch.argmax(logp, dim=1).item())
            pred_label = self.id2label.get(idx, str(idx))
            print(f"[DEBUG] Predicted label: {pred_label}")
            return pred_label

    def predict_heuristic(self, text):
        t = (text or "").strip().lower()
        if not t: return "other"
        if t.endswith("?") or t.split(" ")[0] in {"what","why","how","when","where","who"}: return "question"
        if any(p in t for p in ["please ", "can you", "could you", "help me"]): return "directive"
        if any(p in t for p in ["i will","i'll","i can","i am going to"]): return "commissive"
        return "inform"
