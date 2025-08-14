import os
from my_model import EmotionAwareOpenAI

class ResponseRuntime:
    def __init__(self):
        BASE = os.path.dirname(os.path.abspath(__file__))

        
        EMO_CKPT  = os.path.join(BASE, "models","emotion_classifier_best.pt")
        EMO_VOCAB = os.path.join(BASE, "data", "processed", "vocab.pkl")
        
        # Mapping from emotion label to numeric ID
        EMO_LABEL2ID = {
            "no emotion": 0,
            "anger": 1,
            "disgust": 2,
            "fear": 3,
            "happiness": 4,
            "sadness": 5,
            "surprise": 6,
        }

        ACT_CKPT  = os.path.join(BASE, "models", "act_classifier_best.pt")
        ACT_VOCAB = os.path.join(BASE, "data", "processed", "vocab.pkl")
        
        # Mapping from dialogue act label to numeric ID
        ACT_LABEL2ID = {
            "other": 0,
            "inform": 1,
            "question": 2,
            "directive": 3,
            "commissive": 4
        }


        print("[DEBUG] EMO_CKPT =", EMO_CKPT)
        print("[DEBUG] EMO_VOCAB =", EMO_VOCAB)
        print("[DEBUG] ACT_CKPT =", ACT_CKPT)
        print("[DEBUG] ACT_VOCAB =", ACT_VOCAB)
        print("[DEBUG] CWD =", os.getcwd())

        # Initialize the emotion-aware dialogue engine
        self.engine = EmotionAwareOpenAI(
            model="gpt-4o-mini",
            max_context_chars=12000,

            # Emotion classification parameters
            emotion_ckpt=EMO_CKPT,
            emotion_vocab_path=EMO_VOCAB,
            emotion_label2id=EMO_LABEL2ID,
            emotion_max_len=50,
            emotion_embed_dim=128,
            emotion_hidden_dim=64,

            # Dialogue act classification parameters
            act_ckpt=ACT_CKPT,
            act_vocab_path=ACT_VOCAB,
            act_label2id=ACT_LABEL2ID,
            act_max_len=50
        )

    def generate(self, history, temperature=0.7, top_p=0.9, max_new_tokens=256):
        turns = [(m["role"], m["content"]) for m in history]
        # Stream the generated output
        for chunk in self.engine.generate_stream(
            turns=turns,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        ):
            yield chunk
