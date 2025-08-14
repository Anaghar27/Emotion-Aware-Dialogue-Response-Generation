import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from classifiers import EmotionClassifier, ActClassifier

# Mapping from detected emotion label to desired response tone
EMOTION_TONE = {
    "no emotion": "matter-of-fact, clear",
    "anger":      "calm, validating, non-confrontational",
    "disgust":    "patient, reflective, balanced",
    "fear":       "gentle, grounding, confidence-building",
    "happiness":  "positive, concise, appreciative",
    "sadness":    "warm, reassuring, supportive",
    "surprise":   "curious, clarifying, steady",
    "neutral":    "matter-of-fact, clear",
}

# Mapping from detected dialogue act to response style
DIALOGUE_ACT_STYLE = {
    "question": "answer clearly, then ask one helpful follow-up if useful",
    "inform": "be concise and structured.",
    "other":      "be concise, helpful, and clarify if needed.",
    "directive": "acknowledge and follow whats specified.",
    "commisive": "confirm next steps, express encouragement, and offer assistance if needed.",
}

class EmotionAwareOpenAI:
    def __init__(self, model="gpt-4o-mini",
                 max_context_chars=12000,
                
                # Emotion classifier parameters
                 emotion_ckpt=None, emotion_vocab_path=None, emotion_label2id=None,
                 emotion_max_len=50, emotion_embed_dim=128, emotion_hidden_dim=64, emotion_dropout=0.5,

                # Act classifier parameters
                 act_ckpt=None, act_vocab_path=None, act_label2id=None,
                 act_max_len=50, act_embed_dim=128, act_hidden_dim=64, act_dropout=0.5):
        # Load API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env")
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_context_chars = max_context_chars

        # Initialize emotion classifier if model paths are provided
        if emotion_ckpt and emotion_vocab_path:
            self.emotion = EmotionClassifier(
                ckpt_path=emotion_ckpt,
                vocab_path=emotion_vocab_path,
                label2id=emotion_label2id or {"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6},
                max_len=emotion_max_len,
                embed_dim=emotion_embed_dim,
                hidden_dim=emotion_hidden_dim,
                dropout=emotion_dropout
            )
        else:
            self.emotion = EmotionClassifier(ckpt_path=None, vocab_path=None)


        # Initialize act classifier if model paths are provided
        self.act = None
        if act_ckpt and act_vocab_path:
            self.act = ActClassifier(
                ckpt_path=act_ckpt,
                vocab_path=act_vocab_path,
                label2id=act_label2id or {"other":0,"inform":1,"question":2,"directive":3,"commissive":4},
                max_len=act_max_len,
                embed_dim=act_embed_dim,
                hidden_dim=act_hidden_dim,
                dropout=act_dropout
            )

    def classify_last(self, turns):
        last_user = ""
        for role, content in reversed(turns):
            if role == "user":
                last_user = content
                break

        emo = self.emotion.predict(last_user) or "0"
        
        # Predict act using classifier if available, else heuristic rules
        if self.act is not None:
            dact = self.act.predict(last_user)
        else:
            tl = (last_user or "").strip().lower()
            if not tl:
                dact = "other"
            elif tl.endswith("?") or tl.split(" ")[0] in {"what","why","how","when","where","who"}:
                dact = "question"
            elif any(p in tl for p in ["please ", "can you", "could you", "help me"]):
                dact = "directive"
            elif any(p in tl for p in ["i will","i'll","i can","i am going to"]):
                dact = "commissive"
            else:
                dact = "inform"

        return emo, dact

    def _build_messages(self, turns):
        emo, dact = self.classify_last(turns)
        tone = EMOTION_TONE.get(emo, EMOTION_TONE["neutral"])
        act_style = DIALOGUE_ACT_STYLE.get(dact, DIALOGUE_ACT_STYLE["other"])

        # System prompt to enforce tone and dialogue act style
        system = (
            "You are an emotionally intelligent assistant. "
            "Strictly follow the user emotion detected and use tone that is specified without fail."
            "Strictly follow the dialogue act detected and reply accordingly"
            "Dont use your pre-trained emotion detection ability, use the detected emotion and act from this system."
            "Be concise, kind, and maintain continuity with prior turns. "
            f"Detected user emotion: {emo}. Use a tone that is {tone}. "
            f"Detected dialogue act: {dact}. Please {act_style}."
        )

        # Build messages list starting with system prompt
        messages = [{"role": "system", "content": system}]
        for role, content in turns:
            r = role if role in ("system", "user", "assistant") else "user"
            messages.append({"role": r, "content": content})

        # Trim messages from the start if total characters exceed limit
        total = 0
        trimmed = []
        for m in reversed(messages):
            total += len(m["content"])
            trimmed.append(m)
            if total > self.max_context_chars:
                break
        return list(reversed(trimmed))

    def generate_stream(self, turns, temperature=0.7, top_p=0.9, max_new_tokens=256):
        messages = self._build_messages(turns)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stream=True,
        )
        # Yield generated chunks
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                continue
