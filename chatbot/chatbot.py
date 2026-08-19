"""
chatbot/chatbot.py
==================
MindBridge — Privacy-Preserving Emotion-Aware Chat Assistant

NOVELTY IMPLEMENTATIONS:
  1. Emotion-Aware Federated Personalization
     - Global FL model + Personal Emotion Calibration Layer per user
     - System learns each user's emotional language patterns
     - "I'm fine" → detects hidden sadness for users who understate

  2. Emotion Drift Detection
     - Monitors emotional trajectory over time
     - Detects sudden drops: happy→happy→neutral→sad→very sad
     - Triggers proactive drift alerts

  3. Privacy-Preserving Emotional Memory (STRONGEST NOVELTY)
     - Stores emotion probability vectors, NOT raw text
     - "I feel lonely today" → stored as {sadness:0.8, loneliness:0.7}
     - Full conversation text is never persisted to disk

  6. Crisis Risk Scoring System
     - risk_score = emotion_weight + keyword_score + history_score
     - Graduated responses: concern → alert → crisis
     - Not binary — understands degree of risk

  7. Federated Emotional Knowledge Graph
     - Builds emotion→topic relationships locally
     - Tracks which topics co-occur with which emotions for each user
     - Graph grows per session, used to personalize responses

  8. Emotion-Adaptive Response Style
     - Response tone conditioned on emotion category
     - Sadness → slow, empathetic | Anxiety → grounding
     - Anger → calming | Joy → celebratory
"""

import os
import sys
import json
import random
import datetime
import re
import subprocess
import math
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from collections import deque, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH     = "saved_model/model_final.pt"
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LEN        = 32
NUM_LABELS     = 28
DEVICE         = torch.device("cpu")
HISTORY_DIR    = "user_histories"
os.makedirs(HISTORY_DIR, exist_ok=True)

GROQ_API_KEY = ""              # not used — Ollama handles responses
# GROQ_MODEL   = "llama3-8b-8192"
OLLAMA_MODEL = "llama3"        # model you pulled
OLLAMA_URL   = "http://localhost:11434/api/generate"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# Emotion index map for vector operations
EMOTION_INDEX = {e: i for i, e in enumerate(EMOTION_LABELS)}

POSITIVE_EMOTIONS = {
    "joy", "love", "gratitude", "excitement", "admiration",
    "amusement", "approval", "caring", "optimism", "pride",
    "relief", "surprise"
}
NEGATIVE_EMOTIONS = {
    "sadness", "grief", "anger", "annoyance", "fear",
    "disappointment", "disapproval", "disgust", "embarrassment",
    "nervousness", "remorse"
}

# Emotion weights for risk scoring (Point 6)
RISK_WEIGHTS = {
    "grief": 0.9, "sadness": 0.8, "fear": 0.7, "remorse": 0.7,
    "nervousness": 0.6, "disappointment": 0.6, "embarrassment": 0.5,
    "anger": 0.5, "disgust": 0.4, "annoyance": 0.3,
    "disapproval": 0.3, "neutral": 0.1, "confusion": 0.2,
    "joy": -0.3, "love": -0.3, "gratitude": -0.3,
    "excitement": -0.3, "optimism": -0.2, "pride": -0.2,
    "admiration": -0.1, "amusement": -0.1, "approval": -0.1,
    "caring": -0.1, "relief": -0.2, "surprise": 0.0,
    "curiosity": 0.0, "desire": 0.1, "realization": 0.1,
}

EMOTION_COLORS = {
    "joy": "#22C55E", "love": "#EC4899", "gratitude": "#10B981",
    "excitement": "#F59E0B", "admiration": "#6366F1",
    "amusement": "#8B5CF6", "approval": "#14B8A6",
    "caring": "#F97316", "optimism": "#EAB308", "pride": "#84CC16",
    "relief": "#06B6D4", "surprise": "#A855F7",
    "sadness": "#3B82F6", "grief": "#1E3A5F", "anger": "#EF4444",
    "annoyance": "#F87171", "fear": "#7C3AED",
    "disappointment": "#6B7280", "disapproval": "#DC2626",
    "disgust": "#92400E", "embarrassment": "#DB2777",
    "nervousness": "#9333EA", "remorse": "#1D4ED8",
    "confusion": "#94A3B8", "curiosity": "#38BDF8",
    "desire": "#FB923C", "realization": "#A3E635", "neutral": "#CBD5E1",
}

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL SPELL CORRECTOR
# ══════════════════════════════════════════════════════════════════════════════
SPELLING_FIXES = {
    "sucide":"suicide","suicde":"suicide","suicied":"suicide",
    "suiside":"suicide","suiicde":"suicide","suiced":"suicide",
    "sucidal":"suicidal","suicidel":"suicidal","suicidall":"suicidal",
    "killl":"kill","kil":"kill","dye":"die","dyeing":"dying",
    "dieing":"dying","mself":"myself","myslef":"myself","mysef":"myself",
    "hurrt":"hurt","hert":"hurt","selfharm":"self harm",
    "hapyy":"happy","hapy":"happy","sadd":"sad","saddd":"sad",
    "depresed":"depressed","depresseed":"depressed","depresion":"depression",
    "angrey":"angry","anggry":"angry","scarred":"scared","scard":"scared",
    "lonley":"lonely","lonliness":"loneliness","lonly":"lonely",
    "tierd":"tired","tird":"tired","exausted":"exhausted",
    "stresed":"stressed","stresd":"stressed","anxeity":"anxiety",
    "anxity":"anxiety","anexity":"anxiety","worrid":"worried",
    "nervus":"nervous","fustrated":"frustrated","frustated":"frustrated",
    "confussed":"confused","cofused":"confused","excitd":"excited",
    "gratefull":"grateful","greatful":"grateful","jelous":"jealous",
    "embrassed":"embarrassed","guilti":"guilty","gulity":"guilty",
    "disapointed":"disappointed","overwelmed":"overwhelmed",
    "headche":"headache","hedache":"headache","haedache":"headache",
    "headach":"headache","miagrane":"migraine","migrane":"migraine",
    "fevr":"fever","feer":"fever","panik":"panic","panick":"panic",
    "insomia":"insomnia","insomania":"insomnia","stomack":"stomach",
    "stomch":"stomach","nausia":"nausea","nausious":"nauseous",
    "dizzey":"dizzy","breathng":"breathing","breething":"breathing",
    "vomitte":"vomiting","vomite":"vomiting","vommiting":"vomiting",
    "vommit":"vomit","nausiated":"nauseous",
    "frend":"friend","freind":"friend","frnds":"friends",
    "famly":"family","familly":"family","skool":"school","scool":"school",
    "collage":"college","univercity":"university","wrk":"work","wrok":"work",
    "becuase":"because","becouse":"because","becasue":"because",
    "doesnt":"doesn't","didnt":"didn't","cant":"can't","wont":"won't",
    "isnt":"isn't","wasnt":"wasn't","iam":"i am","dont":"don't",
    "everythin":"everything","evrything":"everything",
    "somthing":"something","nothin":"nothing","poeple":"people",
    "beleive":"believe","belive":"believe","definately":"definitely",
    "untill":"until","tomarrow":"tomorrow","tommorow":"tomorrow",
    "yestarday":"yesterday","alot":"a lot","probaly":"probably",
    "seriosly":"seriously","actully":"actually","litterally":"literally",
    "decison":"decision","desicion":"decision","teh":"the","hte":"the",
    "adn":"and","nad":"and","usualy":"usually","usally":"usually",
    "u":"you","ur":"your","im":"i am","ive":"i have","id":"i would",
    "thats":"that's","whos":"who's","whats":"what's",
}

FUZZY_CRISIS_PATTERNS = [
    r"su[iy]?[cs][iy]?d",
    r"kil+\s*(my)?sel[fv]",
    r"(want|wanna)\s+to\s+d[iy]e",
    r"end\s+(my|this)\s+life",
    r"(harm|hurt)\s*(my)?sel[fv]",
    r"no\s+(reason|point)\s+(to\s+)?li[vf]e",
    r"(better|bettr)\s+off\s+dead",
    r"(cant|can'?t)\s+go\s+on",
    r"not\s+worth\s+li[vf]ing",
]

# High-risk phrases for risk scoring (Point 6)
HIGH_RISK_PHRASES = [
    "want to disappear", "feel useless", "no one would miss me",
    "feel worthless", "can't do this anymore", "given up",
    "nobody cares about me", "what's the point", "tired of living",
    "feel empty", "feel numb", "lost all hope", "see no way out",
    "burden to everyone", "better without me",
]
MEDIUM_RISK_PHRASES = [
    "feel hopeless", "nothing matters", "don't care anymore",
    "feel so alone", "can't cope", "falling apart",
    "losing my mind", "can't take this", "breaking down",
]


def normalize_text(text: str) -> str:
    t     = text.lower().strip()
    words = t.split()
    fixed = []
    for word in words:
        prefix, suffix, clean = "", "", word
        while clean and not clean[0].isalpha():
            prefix += clean[0]; clean = clean[1:]
        while clean and not clean[-1].isalpha():
            suffix = clean[-1] + suffix; clean = clean[:-1]
        fixed.append(prefix + SPELLING_FIXES.get(clean, clean) + suffix)
    return " ".join(fixed)

# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD-BASED POSITIVE SIGNAL OVERRIDE
#  The DistilBERT model can misclassify clearly positive text as sadness when
#  the user's history is predominantly negative (calibration bias + model bias).
#  This layer uses unambiguous positive vocabulary to hard-correct the label
#  BEFORE it reaches the UI or any downstream logic.
# ══════════════════════════════════════════════════════════════════════════════

# Tier 1 — single word is enough (unambiguously positive, can't be negative ctx)
STRONG_POSITIVE_WORDS = {
    "excited", "exciting", "excitement",
    "happy", "happiness", "happier", "happiest",
    "joy", "joyful", "joyous",
    "love", "loving", "loved",
    "wonderful", "amazing", "fantastic", "awesome", "brilliant",
    "great", "excellent", "superb", "magnificent",
    "beautiful", "gorgeous", "pretty", "stunning", "lovely",
    "proud", "pride",
    "grateful", "thankful", "blessed",
    "thrilled", "ecstatic", "elated", "overjoyed",
    "delighted", "glad", "pleased", "cheerful",
    "optimistic", "hopeful", "confident",
    "celebrating", "celebrate", "celebrated",
    "laughing", "laugh", "smiling", "smile",
    "winning", "won", "success", "succeeded",
    "accomplished", "achievement",
    "fun", "enjoy", "enjoyed", "enjoying",
    "good", "well", "fine", "okay",
}

# Tier 2 — phrases that together signal positivity
POSITIVE_PHRASES = [
    "looking good", "looking pretty", "looking beautiful", "looking great",
    "feeling good", "feeling happy", "feeling excited", "feeling amazing",
    "had a great", "had a good", "had a wonderful", "had a fantastic",
    "good haircut", "nice haircut", "great haircut", "new haircut",
    "so happy", "so excited", "so glad", "so proud", "so grateful",
    "really happy", "really excited", "really glad", "really proud",
    "cant wait", "can't wait", "looking forward",
    "good news", "great news", "excited to share", "excited to tell",
    "made me happy", "made me smile", "made my day",
    "i am happy", "i am excited", "i am glad", "i am proud",
    "i am beautiful", "i am pretty", "i am confident",
    "i feel good", "i feel happy", "i feel great", "i feel excited",
    "i feel beautiful", "i feel pretty", "i feel amazing",
    "going well", "went well", "turned out well",
]

# Emotion mapped per keyword signal
POSITIVE_KEYWORD_EMOTION = {
    "excited": "excitement", "exciting": "excitement", "excitement": "excitement",
    "thrilled": "excitement", "ecstatic": "excitement", "elated": "excitement",
    "happy": "joy", "happiness": "joy", "happier": "joy", "happiest": "joy",
    "joy": "joy", "joyful": "joy", "joyous": "joy",
    "glad": "joy", "pleased": "joy", "cheerful": "joy", "delighted": "joy",
    "overjoyed": "joy",
    "proud": "pride", "pride": "pride", "accomplished": "pride",
    "achievement": "pride", "winning": "pride", "won": "pride",
    "success": "pride", "succeeded": "pride",
    "grateful": "gratitude", "thankful": "gratitude", "blessed": "gratitude",
    "wonderful": "joy", "amazing": "joy", "fantastic": "joy",
    "awesome": "joy", "brilliant": "joy", "excellent": "joy",
    "beautiful": "joy", "pretty": "joy", "gorgeous": "joy",
    "stunning": "joy", "lovely": "joy",
    "great": "joy", "superb": "joy", "magnificent": "joy",
    "optimistic": "optimism", "hopeful": "optimism",
    "confident": "pride",
    "love": "love", "loving": "love", "loved": "love",
    "laughing": "amusement", "laugh": "amusement", "fun": "amusement",
    "smiling": "joy", "smile": "joy",
    "celebrating": "joy", "celebrate": "joy", "celebrated": "joy",
    "enjoy": "joy", "enjoyed": "joy", "enjoying": "joy",
}


def keyword_positive_override(normalized: str, model_emotion: str,
                               model_confidence: float):
    """
    Returns (corrected_emotion, corrected_confidence, was_overridden).

    Logic:
    - Check for Tier-2 phrases first (multi-word, strongest signal).
    - Then check Tier-1 words.
    - If a positive signal is found AND the model returned a negative emotion
      OR returned a positive with low confidence (< 0.45), override.
    - Never override if the model ALREADY returned a positive emotion with
      high confidence — trust the model in that case.
    """
    # Don't override if model is already confident and positive
    if model_emotion in POSITIVE_EMOTIONS and model_confidence >= 0.50:
        return model_emotion, model_confidence, False

    words = set(normalized.split())

    # Tier-2: phrase scan
    for phrase in POSITIVE_PHRASES:
        if phrase in normalized:
            # Pick the dominant positive word in the phrase for emotion mapping
            for w in phrase.split():
                if w in POSITIVE_KEYWORD_EMOTION:
                    em = POSITIVE_KEYWORD_EMOTION[w]
                    return em, max(model_confidence, 0.65), True
            return "joy", max(model_confidence, 0.65), True

    # Tier-1: individual word scan
    for word in words:
        if word in STRONG_POSITIVE_WORDS:
            em = POSITIVE_KEYWORD_EMOTION.get(word, "joy")
            return em, max(model_confidence, 0.60), True

    return model_emotion, model_confidence, False


#  risk_score = emotion_weight + keyword_score + history_penalty
#  Graduated: low → concern → alert → crisis
# ══════════════════════════════════════════════════════════════════════════════
CRISIS_KEYWORDS = [
    "want to die", "kill myself", "end my life", "take my life",
    "suicide", "suicidal", "harm myself", "hurt myself",
    "self harm", "self-harm", "no point living", "not worth living",
    "better off dead", "wish i was dead", "want to disappear forever",
    "cant go on", "can't go on", "end it all", "give up on life",
    "life is not worth", "don't want to be here anymore",
    "want to stop existing", "no reason to live", "done with life",
    "nothing to live for", "would be better if i was gone",
]

CRISIS_RESPONSES = [
    (
        "I hear you, and what you just shared has me genuinely worried about you. "
        "Please reach out to iCall right now — 9152987821 — they're available 24/7 "
        "and trained to listen without judgment. "
        "The part of you that typed this message still wants to be heard. "
        "Please let that part make one more call."
    ),
    (
        "I'm so glad you said something. What you're carrying sounds unbearable right now, "
        "and you deserve real human support immediately. "
        "Please call AASRA: 9820466627 — they are there for exactly this moment. "
        "You don't have to figure out what comes next. Just make that one call."
    ),
    (
        "Thank you for trusting me with this. Your safety matters more than anything right now. "
        "Please contact Vandrevala Foundation: 1860-2662-345, available 24 hours. "
        "You matter far beyond this moment of pain. Please reach out to them now."
    ),
]

ALERT_RESPONSES = [
    (
        "It sounds like you're going through something really difficult right now. "
        "What you're feeling is real and it deserves proper support. "
        "If things get darker, please don't hesitate to reach out to iCall: 9152987821. "
        "For now — would you like to tell me more about what's been happening?"
    ),
    (
        "I'm noticing that what you're sharing feels quite heavy. "
        "You don't have to carry this alone. "
        "Would you be open to talking to someone who specializes in this? "
        "iCall (9152987821) is free and confidential. "
        "In the meantime — I'm here. What's been going on?"
    ),
]

CONCERN_RESPONSES = [
    "What you're describing sounds like a lot to be carrying. I want to make sure you're okay. What's been weighing on you the most?",
    "I'm picking up on something that sounds heavier than the words alone. How are you really doing right now?",
    "That kind of feeling deserves real attention. Can you tell me more about how long you've been feeling this way?",
]


def compute_risk_score(normalized: str, emotion: str,
                       confidence: float, history: list) -> float:
    """
    Computes a graduated risk score 0.0 → 1.0
    risk = emotion_component + keyword_component + history_component
    """
    # Emotion component
    emotion_risk = RISK_WEIGHTS.get(emotion, 0.0) * confidence

    # Keyword component
    keyword_score = 0.0
    for phrase in HIGH_RISK_PHRASES:
        if phrase in normalized:
            keyword_score = max(keyword_score, 0.7)
    for phrase in MEDIUM_RISK_PHRASES:
        if phrase in normalized:
            keyword_score = max(keyword_score, 0.4)

    # History component — persistent negative emotions increase risk
    history_score = 0.0
    if len(history) >= 3:
        recent_negative = sum(1 for e in history[-5:] if e in NEGATIVE_EMOTIONS)
        history_score = recent_negative / 5 * 0.3

    raw = emotion_risk * 0.4 + keyword_score * 0.4 + history_score * 0.2
    return min(max(raw, 0.0), 1.0)


def is_crisis(original: str, normalized: str) -> bool:
    for phrase in CRISIS_KEYWORDS:
        if phrase in normalized or phrase in original:
            return True
    for pattern in FUZZY_CRISIS_PATTERNS:
        if re.search(pattern, original) or re.search(pattern, normalized):
            return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(TOKENIZER_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, NUM_LABELS)
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls)


def load_model(path: str) -> EmotionClassifier:
    model = EmotionClassifier()
    if os.path.exists(path):
        try:
            model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False))
            print(f"[Chatbot] Model loaded: {path}")
        except Exception as e:
            print(f"[Chatbot] WARNING: {e} — using untrained weights")
    else:
        print(f"[Chatbot] WARNING: No model at {path}")
    model.eval()
    return model

# ══════════════════════════════════════════════════════════════════════════════
#  POINT 2: EMOTION DRIFT DETECTION
#  Monitors trajectory, detects sudden drops, triggers proactive alerts
# ══════════════════════════════════════════════════════════════════════════════
class EmotionDriftDetector:
    """
    Tracks emotional trajectory over time.
    Detects: gradual decline, sudden crash, sustained negativity.
    Unlike basic trackers, this looks at the SHAPE of the curve.
    """
    def __init__(self, window: int = 8):
        self.emotions   = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)
        self.scores     = deque(maxlen=window)  # numeric score per emotion

    def _score(self, emotion: str, confidence: float) -> float:
        base = RISK_WEIGHTS.get(emotion, 0.0)
        return base * confidence

    def update(self, emotion: str, confidence: float):
        self.emotions.append(emotion)
        self.timestamps.append(datetime.datetime.now())
        self.scores.append(self._score(emotion, confidence))

    def detect_positive_shift(self, current_emotion: str, current_confidence: float) -> bool:
        """
        Returns True when the user's CURRENT message is clearly positive
        but their recent history was predominantly negative.

        Rule: history trend was negative/worsening AND current emotion is
        positive with confidence >= 0.45.  A single strong positive signal
        always wins over accumulated negative history.
        """
        if current_emotion not in POSITIVE_EMOTIONS:
            return False
        if current_confidence < 0.45:
            return False
        if len(self.emotions) < 2:
            return False
        # Count negative emotions in the window (exclude the very last one
        # which hasn't been appended yet — that's the current message)
        history_emotions = list(self.emotions)
        neg_count = sum(1 for e in history_emotions if e in NEGATIVE_EMOTIONS)
        total     = len(history_emotions)
        # Shift triggered when majority of recent history was negative
        return neg_count >= max(1, total // 2)

    def get_trend(self) -> str:
        if len(self.scores) < 2:
            return "stable"
        pos_count = sum(1 for e in self.emotions if e in POSITIVE_EMOTIONS)
        neg_count = sum(1 for e in self.emotions if e in NEGATIVE_EMOTIONS)
        half      = len(self.scores) // 2
        first     = sum(list(self.scores)[:half])
        second    = sum(list(self.scores)[half:])
        if second > first + 0.4:   return "worsening"
        if second < first - 0.4:   return "improving"
        if neg_count >= len(self.emotions) - 1: return "persistent_negative"
        if pos_count >= len(self.emotions) - 1: return "persistent_positive"
        return "mixed"

    def detect_drift(self) -> dict:
        """
        Returns drift info: type, severity, alert_message
        drift types: sudden_drop, gradual_decline, sustained_negative,
                     sudden_lift, stable
        """
        if len(self.scores) < 3:
            return {"type": "stable", "severity": 0, "alert": None}

        score_list = list(self.scores)
        emotion_list = list(self.emotions)

        # Sudden drop — last score much worse than average of previous
        avg_prev = sum(score_list[:-1]) / len(score_list[:-1])
        last     = score_list[-1]
        if last - avg_prev > 0.35:
            return {
                "type": "sudden_drop",
                "severity": min(last - avg_prev, 1.0),
                "alert": (
                    "I noticed a sudden shift in how you're feeling. "
                    "Something seems to have changed — would you like to talk about what happened?"
                )
            }

        # Gradual decline — monotonically worsening over last 4
        if len(score_list) >= 4:
            last4 = score_list[-4:]
            if all(last4[i] <= last4[i+1] for i in range(len(last4)-1)) \
               and last4[-1] - last4[0] > 0.2:
                return {
                    "type": "gradual_decline",
                    "severity": last4[-1] - last4[0],
                    "alert": (
                        "I've been noticing your mood has been gradually declining. "
                        "It seems like things have been getting harder over the course of our conversation. "
                        "Would you like to talk about what's been building up?"
                    )
                }

        # Sustained negativity — 5+ consecutive negative emotions
        if len(emotion_list) >= 5:
            last5 = emotion_list[-5:]
            if all(e in NEGATIVE_EMOTIONS for e in last5):
                return {
                    "type": "sustained_negative",
                    "severity": 0.7,
                    "alert": (
                        "I want to check in with you — you've been expressing difficult emotions "
                        "consistently throughout our conversation. "
                        "That's a lot to carry. How are you actually doing right now, beneath all of this?"
                    )
                }

        return {"type": "stable", "severity": 0, "alert": None}

    def get_previous_emotion(self):
        return list(self.emotions)[-2] if len(self.emotions) >= 2 else None

    def get_dominant(self) -> str:
        if not self.emotions: return "neutral"
        counts = {}
        for e in self.emotions:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def emotion_just_changed(self, current: str) -> bool:
        prev = self.get_previous_emotion()
        if not prev: return False
        return (prev in POSITIVE_EMOTIONS and current in NEGATIVE_EMOTIONS) or \
               (prev in NEGATIVE_EMOTIONS and current in POSITIVE_EMOTIONS)

# ══════════════════════════════════════════════════════════════════════════════
#  POINT 1: PERSONAL EMOTION CALIBRATION LAYER
#  Learns each user's emotional language patterns.
#  "I'm fine" → if user consistently showed sadness after saying "fine",
#  system applies a calibration to adjust the raw model output.
# ══════════════════════════════════════════════════════════════════════════════
class PersonalEmotionCalibrator:
    """
    Two-level architecture:
    Global Model output → Personal Calibration → Adjusted emotion

    Calibration dictionary maps:
    phrase_pattern → {emotion: adjustment_weight}

    Example after learning:
    "i am fine" → system saw sadness 4 times after this
    → calibration adds weight to sadness score when "fine" appears

    TIME DECAY (Fix 6):
    When user has happy sessions, the sadness bias for "i am fine" fades.
    So: Day 1 (sad) → "i am fine" = sadness detected
        Day 3 (happy sessions) → "i am fine" = positive detected (moved on)
    The calibrator learns the CURRENT emotional state, not just the past.
    """
    def __init__(self):
        # phrase → {emotion: count}
        self.phrase_emotion_map: dict = defaultdict(lambda: defaultdict(int))
        self.total_observations: int  = 0
        self.MIN_OBS = 3

    def record(self, text: str, emotion: str, confidence: float):
        """
        Record what emotion followed this text pattern.
        Fix 6: When recording a POSITIVE emotion with high confidence,
        decay (reduce) the negative emotion counts for the same phrase.
        This makes the calibrator forget past sadness when user is genuinely happy.
        """
        words = text.lower().split()
        for n in [1, 2, 3]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if len(phrase) > 2:
                    self.phrase_emotion_map[phrase][emotion] += 1

                    # Fix 6: TIME DECAY — if current emotion is POSITIVE with high
                    # confidence, reduce negative emotion counts for this phrase
                    # This means "i am fine" stops predicting sadness once user
                    # genuinely starts feeling better
                    if (emotion in POSITIVE_EMOTIONS and confidence >= 0.65
                            and phrase in self.phrase_emotion_map):
                        for neg_em in list(self.phrase_emotion_map[phrase].keys()):
                            if neg_em in NEGATIVE_EMOTIONS:
                                # Decay: reduce negative count by 1 (never below 0)
                                cur = self.phrase_emotion_map[phrase][neg_em]
                                if cur > 1:
                                    self.phrase_emotion_map[phrase][neg_em] = cur - 1
                                elif cur == 1:
                                    # Remove completely if count drops to 0
                                    del self.phrase_emotion_map[phrase][neg_em]

        self.total_observations += 1

    def calibrate(self, text: str, raw_probs: list) -> list:
        """
        Adjust raw model probabilities based on learned user patterns.
        Returns calibrated probability list.
        """
        if self.total_observations < self.MIN_OBS:
            return raw_probs

        calibrated = list(raw_probs)
        words = text.lower().split()

        for n in [1, 2, 3]:
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if phrase in self.phrase_emotion_map:
                    emotion_counts = self.phrase_emotion_map[phrase]
                    total = sum(emotion_counts.values())
                    if total >= self.MIN_OBS:
                        for emotion, count in emotion_counts.items():
                            if emotion in EMOTION_INDEX:
                                idx    = EMOTION_INDEX[emotion]
                                weight = (count / total) * 0.3
                                calibrated[idx] = min(calibrated[idx] + weight, 1.0)

        total_prob = sum(calibrated)
        if total_prob > 0:
            calibrated = [p / total_prob for p in calibrated]
        return calibrated

    def get_hidden_emotion_hint(self, text: str) -> str:
        """
        Check if phrase historically correlates with a NEGATIVE emotion.
        Fix 6: Only returns a hint if negative counts still dominate.
        If positive counts have grown (user moved on), no hint is returned.
        """
        words = text.lower().split()
        for phrase in [" ".join(words), " ".join(words[:3])]:
            if phrase in self.phrase_emotion_map:
                counts = self.phrase_emotion_map[phrase]
                total  = sum(counts.values())
                if total >= self.MIN_OBS:
                    dominant  = max(counts, key=counts.get)
                    dominance = counts[dominant] / total
                    # Only hint sadness if negative emotion is STILL dominant
                    # (not faded by positive sessions)
                    if dominance > 0.6 and dominant in NEGATIVE_EMOTIONS:
                        return dominant
        return None

    def to_dict(self) -> dict:
        return {
            "phrase_emotion_map":  {k: dict(v) for k, v in self.phrase_emotion_map.items()},
            "total_observations":  self.total_observations,
        }

    def from_dict(self, data: dict):
        self.total_observations = data.get("total_observations", 0)
        for k, v in data.get("phrase_emotion_map", {}).items():
            self.phrase_emotion_map[k] = defaultdict(int, v)

# ══════════════════════════════════════════════════════════════════════════════
#  POINT 3: PRIVACY-PRESERVING EMOTIONAL MEMORY
#  Stores emotion probability VECTORS, not raw text.
#  "I feel lonely today" → {sadness: 0.8, grief: 0.3, ...} + topic
#  The actual sentence is NEVER written to disk.
# ══════════════════════════════════════════════════════════════════════════════
class PrivacyPreservingMemory:
    """
    Instead of: "User said: I feel lonely today"
    Stores:     {emotion_vector: [0.8, 0.0, 0.0, ...], topic: "loneliness", timestamp: "..."}

    This is the core privacy-preserving innovation:
    - Cannot reconstruct original text from stored data
    - Still captures full emotional context
    - Enables personalization without text privacy risk
    """
    def __init__(self, user_id: str):
        self.user_id      = user_id
        self.path         = os.path.join(HISTORY_DIR, f"{user_id}_emotional_memory.json")
        self.data         = self._load()
        self.calibrator   = PersonalEmotionCalibrator()
        self._load_calibrator()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "user_id":          self.user_id,
            "sessions":         0,
            "total_messages":   0,
            "emotion_counts":   {},
            "daily_log":        {},
            # PRIVACY-PRESERVING: stores vectors not text
            "emotion_timeline": [],   # list of {vector, topic, timestamp, emotion}
            "knowledge_graph":  {},   # Point 7: emotion→topic relationships
            "first_seen":       str(datetime.date.today()),
            "last_seen":        str(datetime.date.today()),
            "calibrator_data":  {},
        }

    def _load_calibrator(self):
        if "calibrator_data" in self.data and self.data["calibrator_data"]:
            self.calibrator.from_dict(self.data["calibrator_data"])

    def save(self):
        """
        In federated mode the server does NOT persist user data to disk.
        Data is returned to the client via API responses and stored in
        the client's localStorage. This method is kept for compatibility
        but intentionally does not write to disk.
        """
        self.data["calibrator_data"] = self.calibrator.to_dict()
        # No file write — client owns the data.

    def start_session(self):
        self.data["sessions"]  += 1
        self.data["last_seen"]  = str(datetime.date.today())
        self.save()

    def record_emotion_vector(self, probs: list, emotion: str,
                               topic: str, user_text: str):
        """
        PRIVACY-PRESERVING STORAGE:
        Store the probability vector + topic + timestamp.
        The raw user_text is passed to the calibrator for learning
        but NOT stored to disk.
        """
        # Update calibrator with text pattern → emotion mapping
        self.calibrator.record(user_text, emotion, max(probs))

        # Store vector (not text) — PRIVACY PRESERVED
        timestamp = datetime.datetime.now().isoformat()
        today     = str(datetime.date.today())

        # Round to 2 decimal places to reduce storage size
        compact_vector = [round(p, 2) for p in probs]

        self.data["emotion_timeline"].append({
            "vector":    compact_vector,
            "emotion":   emotion,
            "topic":     topic,
            "timestamp": timestamp,
            "date":      today,
        })
        # Keep only last 200 entries
        if len(self.data["emotion_timeline"]) > 200:
            self.data["emotion_timeline"] = self.data["emotion_timeline"][-200:]

        # Emotion counts
        c = self.data["emotion_counts"]
        c[emotion] = c.get(emotion, 0) + 1
        self.data["total_messages"] += 1

        # Daily log
        dl = self.data.setdefault("daily_log", {})
        dl.setdefault(today, {})[emotion] = dl.get(today, {}).get(emotion, 0) + 1

        # Point 7: Update knowledge graph
        if topic:
            kg = self.data.setdefault("knowledge_graph", {})
            kg.setdefault(emotion, {})
            kg[emotion][topic] = kg[emotion].get(topic, 0) + 1

        self.save()

    def get_emotional_history(self, days: int = 7) -> list:
        """Returns recent emotion labels for drift analysis."""
        cutoff = datetime.date.today() - datetime.timedelta(days=days)
        return [
            entry["emotion"]
            for entry in self.data["emotion_timeline"]
            if entry.get("date", "2000-01-01") >= str(cutoff)
        ]

    def get_average_vector(self, days: int = 7) -> list:
        """Returns average emotion probability vector over past N days."""
        history = [
            entry["vector"]
            for entry in self.data["emotion_timeline"]
            if entry.get("date", "2000-01-01") >= str(
                datetime.date.today() - datetime.timedelta(days=days)
            )
        ]
        if not history:
            return [0.0] * NUM_LABELS
        avg = [sum(h[i] for h in history) / len(history) for i in range(NUM_LABELS)]
        return avg

    def dominant_emotion(self) -> str:
        c = self.data.get("emotion_counts", {})
        return max(c, key=c.get) if c else "neutral"

    def is_returning(self) -> bool:
        return self.data.get("sessions", 0) > 1

    def sessions(self) -> int:
        return self.data.get("sessions", 0)

    def total_messages(self) -> int:
        return self.data.get("total_messages", 0)

# ══════════════════════════════════════════════════════════════════════════════
#  POINT 7: FEDERATED EMOTIONAL KNOWLEDGE GRAPH
#  Builds emotion→topic relationship graphs locally per user.
#  Enables: "When this user is sad, it's usually about work/study/family"
# ══════════════════════════════════════════════════════════════════════════════
class EmotionalKnowledgeGraph:
    """
    Local knowledge graph mapping:
    emotion → {topic: frequency}

    Built from each user's conversation history.
    In a real FL system, these graphs would be federated.
    Here, each user's graph is local and private.

    Usage:
    "User is sad" → look up graph → "sad usually co-occurs with study"
    → personalize response to mention study context
    """
    def __init__(self, graph_data: dict):
        self.graph = graph_data  # {emotion: {topic: count}}

    def get_likely_topic(self, emotion: str) -> str:
        """Returns the most common topic for this emotion for this user."""
        if emotion not in self.graph:
            return None
        topics = self.graph[emotion]
        if not topics:
            return None
        return max(topics, key=topics.get)

    def get_emotion_topic_strength(self, emotion: str, topic: str) -> float:
        """Returns how strongly this emotion co-occurs with this topic (0-1)."""
        if emotion not in self.graph or not self.graph[emotion]:
            return 0.0
        total = sum(self.graph[emotion].values())
        return self.graph[emotion].get(topic, 0) / total if total > 0 else 0.0

    def get_graph_summary(self) -> dict:
        """Returns top emotion-topic pairs for display."""
        summary = {}
        for emotion, topics in self.graph.items():
            if topics:
                top = max(topics, key=topics.get)
                summary[emotion] = top
        return summary

# ══════════════════════════════════════════════════════════════════════════════
#  TOPIC & CONTEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
TOPIC_KEYWORDS = {
    "work":       ["work", "job", "boss", "office", "colleague", "meeting",
                   "deadline", "project", "career", "salary", "fired", "hired"],
    "family":     ["family", "mom", "dad", "sister", "brother", "parent",
                   "mother", "father", "home", "relative", "sibling"],
    "friends":    ["friend", "friends", "social", "party", "group", "peer",
                   "classmate", "teammate", "bestie", "buddy"],
    "study":      ["study", "exam", "college", "university", "class", "test",
                   "assignment", "marks", "grade", "professor", "school"],
    "love":       ["relationship", "partner", "girlfriend", "boyfriend",
                   "crush", "breakup", "dating", "love", "heart", "miss"],
    "future":     ["future", "goal", "dream", "plan", "career", "hope",
                   "worried about", "scared of", "uncertain", "what if"],
    "loneliness": ["alone", "lonely", "isolated", "no one", "nobody",
                   "nobody cares", "by myself", "left out", "ignored"],
    "health":     ["health", "sick", "ill", "pain", "hospital", "doctor",
                   "medicine", "tired", "sleep", "body"],
}

DURATION_PATTERNS = [
    (r"\b(\d+)\s*day", "days"), (r"\b(\d+)\s*week", "weeks"),
    (r"\b(\d+)\s*month", "months"), (r"\bjust\s+(now|started)", "recent"),
    (r"\btoday\b", "today"), (r"\byesterday\b", "yesterday"),
    (r"\b(long\s+time|ages|always)", "chronic"),
]

SEVERITY_WORDS_HIGH = ["severe", "terrible", "horrible", "unbearable",
                        "worst", "excruciating", "intense", "extreme",
                        "awful", "dreadful", "killing me", "so bad"]
SEVERITY_WORDS_LOW  = ["little", "slight", "mild", "bit", "kind of",
                        "minor", "small", "not too bad"]


def extract_topic(normalized: str):
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in normalized for kw in keywords):
            return topic
    return None


def extract_duration(normalized: str) -> str:
    for pattern, label in DURATION_PATTERNS:
        m = re.search(pattern, normalized)
        if m:
            if label == "days":
                days = int(m.group(1))
                return f"concerning:{days}days" if days >= 3 else f"short:{days}days"
            return f"duration:{label}"
    return None


def extract_severity(normalized: str) -> str:
    if any(w in normalized for w in SEVERITY_WORDS_HIGH): return "high"
    if any(w in normalized for w in SEVERITY_WORDS_LOW):  return "low"
    return "medium"

# ══════════════════════════════════════════════════════════════════════════════
#  MEDICAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════
MEDICAL_CONDITIONS = {
    "nausea": {
        "keywords": ["nausea", "nauseous", "vomit", "vomiting", "feel sick",
                     "want to throw up", "throwing up", "queasy", "stomach upset",
                     "stomach pain", "stomach ache", "indigestion"],
        "emergency_keywords": ["blood", "can't stop", "days", "severe"],
        "emergency_response": "Nausea or vomiting lasting several days, or with blood or severe pain, needs immediate medical care. Please see a doctor right away.",
        "fallback": [
            "I'm sorry you're feeling that way — nausea is really draining. Try sipping small amounts of cold water or ginger tea. Sit upright and avoid heavy foods. Plain crackers or rice if you can manage it. If it continues for more than a few hours, please see a doctor. What else are you feeling alongside this?",
            "That sounds really uncomfortable. Stay hydrated with small sips of water. Ginger — as tea, raw, or ginger ale — genuinely helps. Avoid strong smells and fatty foods. How long has this been going on?",
        ],
    },
    "headache": {
        "keywords": ["headache", "head pain", "head ache", "migraine",
                     "head is pounding", "head hurts", "throbbing head", "pressure in head"],
        "emergency_keywords": ["worst headache", "sudden severe", "with fever",
                               "with vomiting", "blurred vision", "stiff neck", "days"],
        "emergency_response": "A headache lasting several days or one with fever, vomiting, or vision changes needs medical attention. Please see a doctor today — don't wait this out.",
        "fallback": [
            "That's been going on long enough to need proper attention. Drink a large glass of water right now, step away from screens, and rest in a cool dark room. Three days is your body asking to be taken seriously — if it doesn't ease today, please see a doctor.",
            "A headache this persistent means your body needs real rest. Dehydration is a major trigger — have you been drinking enough water? Paracetamol can help if you're not allergic. If it keeps worsening or you have other symptoms, please get checked.",
        ],
    },
    "fever": {
        "keywords": ["fever", "temperature", "feeling hot", "body temperature",
                     "chills", "shivering", "burning up"],
        "emergency_keywords": ["103", "104", "105", "seizure", "unconscious", "rash"],
        "emergency_response": "A fever this high or with these symptoms needs immediate medical care. Please contact a doctor or go to emergency services right away.",
        "fallback": [
            "Fever means your immune system is working. Stay hydrated with water, ORS, or coconut water. Rest completely and use a cool damp cloth on your forehead. If it's above 102°F or has lasted more than 48 hours, please visit a doctor. How high is it?",
            "I'm sorry you're dealing with that — fevers really drain you. Keep sipping fluids even if you don't feel thirsty. Light food like rice or toast is easier right now. Is there anyone with you who can help keep an eye on you?",
        ],
    },
    "anxiety attack": {
        "keywords": ["anxiety attack", "panic attack", "can't breathe", "heart racing",
                     "heart pounding", "overwhelming anxiety", "sudden fear", "losing control"],
        "emergency_keywords": ["first time", "chest pain", "can't stop", "passing out"],
        "emergency_response": "If this is your first panic attack or you have chest pain, please get medical evaluation. You deserve proper support.",
        "fallback": [
            "You are safe right now. Place both feet flat on the floor. Try box breathing: in for 4, hold for 4, out for 4, hold for 4. Repeat three times. Name five things you can see around you right now. These feelings are intense but they will pass.",
            "I'm right here with you. Try 5-4-3-2-1 grounding: five things you see, four you can touch, three you hear, two you smell, one you taste. Your body is reacting as if there's danger, but you are physically safe. Stay with me.",
        ],
    },
    "stress": {
        "keywords": ["so stressed", "overwhelmed", "can't cope", "too much pressure",
                     "breaking point", "stressed out", "everything is too much", "can't handle"],
        "emergency_keywords": ["can't function", "can't eat", "thoughts of harming"],
        "emergency_response": "When stress reaches this level, professional support is essential. Please speak with a counselor or doctor — this is not something to carry alone.",
        "fallback": [
            "That level of stress is real and your body is feeling it. Right now: separate what is genuinely urgent from what just feels urgent. Write down every stressor, then circle only the ones within your control today. What feels most pressing right now?",
            "When everything feels like too much, do just one small thing — not the whole list, just one. It breaks the paralysis. Have you eaten and had water today? Stress depletes both faster than we realize.",
        ],
    },
    "sleep": {
        "keywords": ["can't sleep", "insomnia", "not sleeping", "sleep problems",
                     "awake all night", "can't fall asleep", "exhausted", "no sleep"],
        "emergency_keywords": ["weeks", "months", "hallucinating", "can't function"],
        "emergency_response": "Extended sleep deprivation needs professional attention. Please consult a doctor — this level of disruption requires proper evaluation.",
        "fallback": [
            "Sleep and emotional wellbeing are deeply connected. For tonight: cool dark room, no screens 30 minutes before bed, and instead of trying to sleep — just tell yourself to rest. Remove the pressure of 'must sleep now.' How long has this been going on?",
            "That's exhausting. Try keeping a consistent sleep time even on weekends, avoid caffeine after 2pm, and if your mind races, write everything down before bed — it's called a brain dump and it genuinely helps. What's been on your mind at night?",
        ],
    },
}


def detect_medical_condition(normalized: str):
    for condition, data in MEDICAL_CONDITIONS.items():
        if any(kw in normalized for kw in data["keywords"]):
            is_emergency = any(ek in normalized for ek in data["emergency_keywords"])
            return condition, is_emergency
    return None, False


def get_groq_response(prompt: str) -> str:
    """
    Uses Ollama — local LLM running entirely on this machine.
    No internet needed. Text never leaves the device.
    Perfectly aligns with the project's privacy-preserving claim.

    Requirements:
      1. Install Ollama: https://ollama.com/download
      2. Pull model:     ollama pull llama3
      3. pip install requests
    """
    try:
        import requests
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,
                "num_predict": 150,
            }
        }
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        return result if result else None
    except requests.exceptions.ConnectionError:
        print("[Ollama] Not running — start with 'ollama serve' in a terminal")
        return None
    except Exception as e:
        print(f"[Ollama] Unavailable: {e} — using local response")
        return None


def get_medical_response(condition: str, is_emergency: bool,
                          user_text: str, duration: str = None) -> str:
    data = MEDICAL_CONDITIONS[condition]
    if is_emergency or (duration and "concerning" in str(duration)):
        return data["emergency_response"]
    # Try Ollama for dynamic medical response
    prompt = (
        f"You are MindBridge, a caring wellness chatbot. "
        f"User: '{user_text}'. Concern: {condition}. "
        f"3 sentences: acknowledge personally, give 2 specific immediate actions, "
        f"advise when to see doctor. Caring friend tone, not medical manual. Under 80 words."
    )
    r = get_groq_response(prompt)
    if r: return r
    templates = data["fallback"]
    return templates[len(user_text) % len(templates)]

# ══════════════════════════════════════════════════════════════════════════════
#  POINT 8: EMOTION-ADAPTIVE RESPONSE STYLE
#  Response tone is conditioned on emotion category
# ══════════════════════════════════════════════════════════════════════════════
RESPONSE_STYLES = {
    "grief_sadness": {
        "pace": "slow",
        "openers": [
            "That's a really heavy thing to be carrying.",
            "I'm sorry you're going through that.",
            "What you're feeling makes complete sense.",
            "That kind of sadness is real, and it deserves space.",
            "That sounds genuinely painful to sit with.",
            "I hear how difficult this has been.",
            "Something like that would weigh on anyone.",
            "That pain is real — I don't want to rush past it.",
        ],
    },
    "anger_annoyance": {
        "pace": "calm",
        "openers": [
            "That frustration is coming through clearly, and it makes sense.",
            "Something clearly crossed a line for you.",
            "I can hear how upset you are — that reaction sounds justified.",
            "That would make a lot of people angry in your position.",
            "When things feel that unfair, anger is the natural response.",
            "Something isn't sitting right, and I understand why you're fired up.",
            "I hear you — that sounds genuinely infuriating.",
            "That kind of thing is hard to let go of.",
        ],
    },
    "fear_nervousness": {
        "pace": "grounding",
        "openers": [
            "That kind of fear is exhausting to carry.",
            "It makes sense that this is frightening.",
            "Being that scared about something takes a real toll.",
            "Fear like that doesn't just disappear — it needs to be heard.",
            "The uncertainty you're feeling is completely valid.",
            "That anxious feeling is real — your body is responding to something it cares about.",
            "Nervous energy like that can be really draining.",
            "Being keyed up about this makes complete sense.",
        ],
    },
    "joy_excitement": {
        "pace": "energetic",
        "openers": [
            "That's genuinely wonderful to hear.",
            "Something clearly went right — I'd love to hear more.",
            "That kind of happiness is worth holding onto.",
            "It really sounds like things are clicking for you right now.",
            "That excitement is coming through clearly — what's happening?",
            "Something has you energised in a really good way.",
            "I can feel the positive energy in what you wrote.",
            "That kind of joy is worth a proper moment.",
        ],
    },
    "default": {
        "pace": "neutral",
        "openers": [
            "I'm listening — thank you for sharing that.",
            "That means something, and I don't want to rush past it.",
            "I hear you.",
            "That's worth talking about.",
            "I'm here with you.",
            "Go on — I'm fully paying attention.",
            "That came through clearly. Tell me more.",
            "I'm glad you said something.",
        ],
    },
}

# Context bridges — injected based on what user mentioned
CONTEXT_BRIDGES = {
    "crying":       ["Crying is the body's way of releasing what words can't hold.",
                     "There's nothing weak about tears — they're honest.",
                     "Letting yourself cry is one of the most human things you can do."],
    "loneliness":   ["Loneliness has a way of making temporary isolation feel permanent.",
                     "Feeling unseen by others is one of the quieter forms of pain.",
                     "The ache of not feeling truly connected is very real."],
    "exam":         ["Academic pressure can be relentless, especially when you care about the outcome.",
                     "The weight of exams goes beyond the grade — it's about identity and future.",
                     "That kind of pressure doesn't just affect the mind — it affects everything."],
    "work":         ["Work-related stress has a way of bleeding into the rest of life.",
                     "When the job feels overwhelming, it's hard to switch off.",
                     "Workplace pressure can feel invisible to others but very real to you."],
    "family":       ["Family situations carry a specific weight because the stakes feel personal.",
                     "When things are difficult at home, there's no real escape from it.",
                     "Family dynamics are complicated — love and difficulty can coexist."],
    "relationship": ["Relationship pain has a way of touching everything else.",
                     "The heart doesn't operate on logic — what you're feeling makes sense.",
                     "When something is wrong in a relationship, it's hard to concentrate on anything else."],
    "future":       ["Uncertainty about the future is one of the most common sources of anxiety.",
                     "Not knowing what comes next can make the present feel unstable.",
                     "The future is genuinely uncertain — and sitting with that is hard."],
    "nausea":       ["Physical discomfort and emotional stress are deeply connected.",
                     "When the body struggles, the mind often follows.",],
    "pain":         ["Physical pain is exhausting — it drains you emotionally too.",
                     "Dealing with pain on top of everything else takes real strength."],
}

KEYWORD_EXTRACTOR = {
    "crying": ["cry", "crying", "tears", "sobbing", "weeping"],
    "loneliness": ["alone", "lonely", "isolated", "no one", "nobody"],
    "exam": ["exam", "test", "grade", "marks", "assignment", "homework"],
    "work": ["work", "job", "boss", "office", "deadline", "colleague"],
    "family": ["family", "mom", "dad", "sister", "brother", "parent"],
    "relationship": ["relationship", "partner", "girlfriend", "boyfriend",
                     "breakup", "crush", "miss", "ex"],
    "future": ["future", "tomorrow", "goal", "dream", "uncertain", "plan"],
    "pain": ["pain", "aching", "hurt", "hurting", "sore"],
}


def get_style(emotion: str) -> dict:
    if emotion in {"sadness", "grief", "remorse", "disappointment"}:
        return RESPONSE_STYLES["grief_sadness"]
    if emotion in {"anger", "annoyance", "disapproval", "disgust"}:
        return RESPONSE_STYLES["anger_annoyance"]
    if emotion in {"fear", "nervousness", "embarrassment"}:
        return RESPONSE_STYLES["fear_nervousness"]
    if emotion in {"joy", "excitement", "amusement", "pride"}:
        return RESPONSE_STYLES["joy_excitement"]
    return RESPONSE_STYLES["default"]


def extract_context_keywords(normalized: str) -> list:
    found = []
    for label, words in KEYWORD_EXTRACTOR.items():
        if any(w in normalized for w in words):
            found.append(label)
    return found


# Trend-aware middle sentences
TREND_LINES = {
    "worsening": [
        "I want you to know I'm noticing that things seem to be getting heavier for you.",
        "This seems to be building rather than easing — I'm here.",
        "I'm paying attention to how this has been accumulating.",
    ],
    "persistent_negative": [
        "This has been sitting with you consistently — that's a lot to carry.",
        "Something that stays this consistent deserves real attention.",
    ],
    "improving": [
        "There's a thread of something lighter in what you're sharing now.",
        "It seems like something has shifted — even slightly — and I notice that.",
    ],
    # ── POSITIVE SHIFT: user moved from negative history → clearly positive now ──
    # These lines celebrate WITHOUT dragging in past sadness.
    "shifting_positive": ["", ""],
    "mixed": ["Your feelings have been moving around — that's very human.", ""],
    "stable": ["", ""],
    "persistent_positive": ["", ""],
}

# Openers specifically for when user has SHIFTED from negative history to positive now.
# They celebrate the present without referencing past sadness.
POSITIVE_SHIFT_OPENERS = [
    "That's really lovely to hear — you deserve that kind of moment.",
    "It's great when something genuinely lifts your spirits like that.",
    "That kind of positive energy is worth holding onto!",
    "I love hearing that — sounds like a good day.",
    "That's the kind of thing that makes a real difference to how you feel.",
    "Something clearly went right today — I'm glad you shared it.",
    "That sounds genuinely uplifting — good for you!",
]

POSITIVE_SHIFT_CLOSERS = [
    "What's been the best part of it?",
    "How are you feeling overall right now?",
    "What else has been going well?",
    "Is there anything else on your mind today?",
    "What made it happen — anything special?",
]

# Knowledge-graph-aware lines — injected when graph reveals pattern
KG_PATTERN_LINES = {
    "study": "Based on what I know about you, academic stress tends to be a big one for you.",
    "work": "I've noticed work tends to come up quite a bit when things feel hard for you.",
    "family": "Family situations seem to affect you deeply — that makes sense given how much you care.",
    "love": "Relationship feelings seem to have a particular weight for you.",
    "loneliness": "Feeling disconnected seems to be something that hits you especially hard.",
}

CLOSERS = {
    "work":       ["What's been the most draining part at work lately?",
                   "Is this something that's been building, or did something specific happen?",
                   "What would feel like even a small relief in that situation?",
                   "How long has this been going on at work?"],
    "family":     ["What's weighing on you most when it comes to family right now?",
                   "Has something specific happened recently, or has this been building?",
                   "How are you holding up in the middle of all of that?"],
    "study":      ["Is the pressure from studies what's been affecting you the most?",
                   "What does the workload feel like right now — manageable or overwhelming?",
                   "Is there a specific exam or deadline that's sitting over you?"],
    "love":       ["How long has this been weighing on you?",
                   "What's the hardest part of this situation to sit with?",
                   "How are you doing on the inside, honestly?"],
    "loneliness": ["How long have you been feeling this disconnected?",
                   "Is there anyone in your life right now who you feel even slightly understood by?",
                   "What would connection look like for you right now?"],
    "future":     ["What feels most uncertain about the future right now?",
                   "What would make you feel even slightly more grounded?",
                   "What does your gut tell you, beneath all the worry?"],
    "friends":    ["What's been going on with your social world lately?",
                   "Is there something specific that happened?"],
    "default":    ["What's been sitting with you the most?",
                   "How long have you been carrying this?",
                   "What would feel like even a small step forward?",
                   "Is there something specific that brought this on today?",
                   "What would you most want someone to understand about what you're going through?",
                   "What's the part of this that's hardest to put into words?",
                   "If you had to name the one thing affecting you most — what would it be?",
                   "What does this feel like in your body right now?"],
}

TRANSITION_OPENERS = {
    "neg_to_pos": ["Something seems to have shifted since earlier — I'm genuinely glad to hear it.",
                   "That's a meaningful turn from where things were.",
                   "I notice things feel a bit different now. What changed?"],
    "pos_to_neg": ["Something seems heavier now than it was — what happened?",
                   "I notice things feel different from before — I'm here.",
                   "That shift is noticeable. What's going on?"],
}

GRATITUDE_RESPONSES = [
    "I'm really glad I could be here for that. How are you feeling now overall?",
    "That means a lot — it's what this space is for. What else is on your mind?",
    "I'm glad something helped. Is there anything else you want to talk through?",
    "You don't need to thank me — just having you here matters. What's next for you?",
]

QUESTION_RESPONSES = {
    "confusion": ["That's a real question — not an easy one. What's making it feel so unclear?",
                  "It's okay not to have the answer yet. What would clarity even look like for you?"],
    "curiosity": ["I love that you're thinking about this. Where is that curiosity pulling you?",
                  "Genuine curiosity usually means you're close to something. What are you circling?"],
    "default":   ["That's worth sitting with. What does your gut say, even just as a feeling?",
                  "You might have more of the answer than you realize. What comes to mind first?",
                  "What would it feel like to trust yourself on this one?"],
}


def build_response(normalized: str, emotion: str, confidence: float,
                   trend: str, topic, drift: dict, knowledge_graph,
                   memory, temporal, msg_count: int,
                   keywords: list, duration: str, severity: str) -> str:
    """
    Point 8: Emotion-adaptive response building.
    Style (pace, tone) is determined by emotion category.
    Content is shaped by: topic, drift, knowledge graph, duration, severity.

    POSITIVE SHIFT OVERRIDE:
    When trend == 'shifting_positive', we skip all negative-history references
    and respond purely to the user's current positive state.
    """
    parts = []
    style = get_style(emotion)

    # ── POSITIVE SHIFT FAST PATH ──────────────────────────────────────────────
    # User was previously sad/negative but is now clearly positive.
    # Do NOT drag in past sadness. Celebrate the present moment only.
    if trend == "shifting_positive":
        parts.append(random.choice(POSITIVE_SHIFT_OPENERS))
        parts.append(random.choice(POSITIVE_SHIFT_CLOSERS))
        return " ".join(p for p in parts if p)

    # ── NORMAL PATH ───────────────────────────────────────────────────────────

    # Returning user opener
    if memory.is_returning() and msg_count == 1:
        dominant = memory.dominant_emotion()
        if dominant in NEGATIVE_EMOTIONS:
            parts.append("Welcome back. I know things have been tough — I'm fully here.")
        else:
            parts.append(random.choice(["Welcome back.", "Good to see you again.",
                                         "I remember you — glad you're here."]))

    # Drift alert — if detected, lead with that (Point 2)
    if drift["alert"]:
        parts.append(drift["alert"])
    elif temporal.emotion_just_changed(emotion):
        prev = temporal.get_previous_emotion()
        if prev and prev in NEGATIVE_EMOTIONS and emotion in POSITIVE_EMOTIONS:
            parts.append(random.choice(TRANSITION_OPENERS["neg_to_pos"]))
        elif prev and prev in POSITIVE_EMOTIONS and emotion in NEGATIVE_EMOTIONS:
            parts.append(random.choice(TRANSITION_OPENERS["pos_to_neg"]))

    # Emotion-adaptive opener (Point 8)
    parts.append(random.choice(style["openers"]))

    # Context bridge — specific to what they mentioned
    for kw in keywords:
        if kw in CONTEXT_BRIDGES:
            parts.append(random.choice(CONTEXT_BRIDGES[kw]))
            break

    # Knowledge graph insight (Point 7) — if strong pattern exists
    if knowledge_graph and topic:
        strength = knowledge_graph.get_emotion_topic_strength(emotion, topic)
        likely_topic = knowledge_graph.get_likely_topic(emotion)
        if strength > 0.5 and likely_topic and likely_topic in KG_PATTERN_LINES:
            parts.append(KG_PATTERN_LINES[likely_topic])

    # Duration awareness
    if duration and "concerning" in str(duration):
        days = re.search(r"\d+", str(duration))
        if days:
            parts.append(
                f"Something going on for {days.group()} days is your body asking "
                f"to be taken seriously — please don't push through it alone."
            )

    # Severity note
    if severity == "high" and emotion in NEGATIVE_EMOTIONS:
        parts.append(random.choice([
            "The intensity of what you're feeling right now is real — please don't dismiss it.",
            "What you're describing sounds genuinely overwhelming.",
            "That level of intensity deserves real care.",
        ]))

    # Trend line (Point 2)
    trend_pool = TREND_LINES.get(trend, TREND_LINES["stable"])
    trend_line = random.choice(trend_pool)
    if trend_line:
        parts.append(trend_line)

    # Topic-aware closer
    closer_pool = CLOSERS.get(topic, CLOSERS["default"])
    parts.append(random.choice(closer_pool))

    # Night check-in
    if datetime.datetime.now().hour >= 21 and emotion in NEGATIVE_EMOTIONS:
        parts.append(random.choice([
            "And please try to get some rest tonight — your mind needs recovery time too.",
            "Make sure you're not pushing through another sleepless night with this.",
        ]))

    return " ".join(p for p in parts if p)


def special_case_check(normalized: str, emotion: str):
    gratitude = ["thank you", "thanks", "that helped", "feeling better",
                 "appreciate", "helpful", "you helped", "better now"]
    if any(p in normalized for p in gratitude) and emotion in {"gratitude", "relief", "joy"}:
        return random.choice(GRATITUDE_RESPONSES)
    if normalized.strip().endswith("?") or \
       normalized.startswith(("what", "why", "how", "when", "who", "where",
                              "should", "could", "would", "can", "is it")):
        pool = QUESTION_RESPONSES.get(emotion, QUESTION_RESPONSES["default"])
        return random.choice(pool)
    return None

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION ANALYSIS GRAPH
# ══════════════════════════════════════════════════════════════════════════════
def generate_analysis_graph(user_data: dict, user_id: str):
    emotion_counts = user_data.get("emotion_counts", {})
    daily_log      = user_data.get("daily_log", {})
    if not emotion_counts:
        return None

    sorted_e = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    emotions = [e[0] for e in sorted_e]
    counts   = [e[1] for e in sorted_e]
    colors   = [EMOTION_COLORS.get(e, "#94A3B8") for e in emotions]

    has_tl = len(daily_log) >= 2
    ncols  = 3 if has_tl else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 2:
        axes = list(axes)
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(f"Your Emotion Analysis — {user_id}",
                 fontsize=13, color="white", fontweight="bold", y=1.02)

    ax1 = axes[0]
    ax1.set_facecolor("#1E293B")
    bars = ax1.barh(emotions[::-1], counts[::-1], color=colors[::-1], height=0.6)
    ax1.set_title("Most Frequent Emotions", color="white", fontsize=11)
    ax1.set_xlabel("Times expressed", color="#94A3B8")
    ax1.tick_params(colors="white", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#374151")
    for bar, count in zip(bars, counts[::-1]):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(count), va="center", color="white", fontsize=8)

    ax2 = axes[1]
    ax2.set_facecolor("#1E293B")
    wedges, texts, autotexts = ax2.pie(
        counts, labels=emotions, colors=colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75, wedgeprops=dict(width=0.6))
    for t in texts:
        t.set_color("white"); t.set_fontsize(8)
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(7)
    total    = user_data.get("total_messages", 0)
    sessions = user_data.get("sessions", 1)
    ax2.text(0, 0.1, str(total), ha="center", va="center",
             fontsize=16, color="white", fontweight="bold")
    ax2.text(0, -0.2, "messages", ha="center", va="center",
             fontsize=8, color="#94A3B8")
    ax2.set_title("Emotion Distribution", color="white", fontsize=11)

    if has_tl:
        ax3 = axes[2]
        ax3.set_facecolor("#1E293B")
        dates = sorted(daily_log.keys())
        scores = []
        for date in dates:
            s = sum(cnt if em in POSITIVE_EMOTIONS else -cnt if em in NEGATIVE_EMOTIONS else 0
                    for em, cnt in daily_log[date].items())
            scores.append(s)
        ax3.plot(range(len(dates)), scores, marker="o", color="#3B82F6",
                 linewidth=2, markersize=6)
        ax3.fill_between(range(len(dates)), scores, alpha=0.2, color="#3B82F6")
        ax3.axhline(y=0, color="#EF4444", linestyle="--", linewidth=1, alpha=0.7, label="Neutral")
        ax3.set_xticks(range(len(dates)))
        ax3.set_xticklabels([d[5:] for d in dates], rotation=45, color="white", fontsize=7)
        ax3.tick_params(axis="y", colors="white")
        ax3.set_title("Wellbeing Timeline", color="white", fontsize=11)
        ax3.set_ylabel("Mood Score", color="#94A3B8", fontsize=9)
        ax3.legend(fontsize=8, labelcolor="white", facecolor="#1E293B", edgecolor="#374151")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#374151")

    first_seen = user_data.get("first_seen", "today")
    last_seen  = user_data.get("last_seen", "today")
    dominant   = emotions[0] if emotions else "neutral"
    fig.text(0.5, -0.02,
             f"Total: {total}  |  Sessions: {sessions}  |  Dominant: {dominant}  |  "
             f"Since: {first_seen}  |  Last: {last_seen}",
             ha="center", fontsize=8, color="#94A3B8",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E293B",
                       alpha=0.8, edgecolor="#374151"))
    plt.tight_layout()
    save_path = os.path.join(HISTORY_DIR, f"{user_id}_analysis.png")
    plt.savefig(save_path, dpi=120, facecolor="#0F1117", bbox_inches="tight")
    plt.close()
    return save_path

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHATBOT CLASS
# ══════════════════════════════════════════════════════════════════════════════
class MindBridge:
    def __init__(self, user_id: str = "guest", model_path: str = MODEL_PATH):
        self.user_id    = user_id
        self.tokenizer  = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
        self.model      = load_model(model_path)
        self.drift      = EmotionDriftDetector(window=8)     # Point 2
        self.memory     = PrivacyPreservingMemory(user_id)   # Points 1 & 3
        self.memory.start_session()
        self.msg_count  = 0
        self._used      = set()

        # Point 7: Knowledge graph from saved data
        self.knowledge_graph = EmotionalKnowledgeGraph(
            self.memory.data.get("knowledge_graph", {})
        )

        print(f"\n[MindBridge] Ready for: {user_id}")
        print(f"[MindBridge] Sessions: {self.memory.sessions()} | "
              f"Messages: {self.memory.total_messages()}")
        print(f"[MindBridge] Privacy memory: emotion vectors only (no text stored)")
        print(f"[MindBridge] Calibrator observations: "
              f"{self.memory.calibrator.total_observations}")
        # Check if Ollama is running
        try:
            import requests as _req
            _req.get("http://localhost:11434", timeout=2)
            print("[MindBridge] Ollama: ACTIVE — local LLM, fully private, no internet needed")
        except Exception:
            print("[MindBridge] Ollama: not running — start with 'ollama serve' | using local templates")

    def detect_emotion(self, text: str):
        """
        Point 1: Two-level detection:
        1. Raw DistilBERT probabilities
        2. Personal calibration adjustment
        3. Keyword positive override — fixes model misclassifying happy text as sad
        Returns calibrated emotion + confidence
        """
        enc = self.tokenizer(text, max_length=MAX_LEN, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])

        raw_probs = torch.softmax(logits, dim=1)[0].tolist()

        # Apply personal calibration layer (Point 1)
        calibrated_probs = self.memory.calibrator.calibrate(text, raw_probs)

        idx        = calibrated_probs.index(max(calibrated_probs))
        confidence = calibrated_probs[idx]
        emotion    = EMOTION_LABELS[idx]

        # ── Keyword positive override (layer 3) ──────────────────────────────
        # If the model returned a negative/neutral emotion but the text clearly
        # contains positive vocabulary, correct the label so the topbar, tag,
        # memory, and all downstream logic show the right emotion.
        emotion, confidence, overridden = keyword_positive_override(
            text, emotion, confidence)
        if overridden:
            # Rebuild probs so memory vector reflects the correction
            if emotion in EMOTION_INDEX:
                calibrated_probs[EMOTION_INDEX[emotion]] = confidence

        return emotion, round(confidence, 3), calibrated_probs

    def respond(self, user_text: str) -> dict:
        user_text       = user_text.strip()
        normalized      = normalize_text(user_text)
        self.msg_count += 1

        if not user_text:
            return {"response": "I'm here and listening whenever you're ready.",
                    "emotion": "neutral", "confidence": 1.0, "trend": "stable",
                    "topic": None, "is_crisis": False, "is_medical": False,
                    "risk_score": 0.0}

        # ── 1. Safety layer ───────────────────────────────────────────────────
        if is_crisis(user_text, normalized):
            emotion, confidence, probs = self.detect_emotion(normalized)
            self.drift.update(emotion, confidence)
            self.memory.record_emotion_vector(probs, emotion, None, user_text)
            return {"response": random.choice(CRISIS_RESPONSES),
                    "emotion": "crisis", "confidence": 1.0,
                    "trend": self.drift.get_trend(), "topic": None,
                    "is_crisis": True, "is_medical": False, "risk_score": 1.0}

        # ── 2. Medical detection ──────────────────────────────────────────────
        medical_condition, is_emergency = detect_medical_condition(normalized)
        duration = extract_duration(normalized)
        if medical_condition:
            med_response = get_medical_response(
                medical_condition, is_emergency, user_text, duration)
            emotion, confidence, probs = self.detect_emotion(normalized)
            self.drift.update(emotion, confidence)
            self.memory.record_emotion_vector(probs, emotion, medical_condition, user_text)
            return {"response": med_response, "emotion": emotion,
                    "confidence": confidence, "trend": self.drift.get_trend(),
                    "topic": medical_condition, "is_crisis": False,
                    "is_medical": True, "risk_score": 0.0}

        # ── 3. Emotion detection with calibration (Point 1) ───────────────────
        emotion, confidence, probs = self.detect_emotion(normalized)

        # ── 3a. POSITIVE SHIFT DETECTION ─────────────────────────────────────
        # Check BEFORE applying the hidden-hint calibrator.
        # If the user's current message is clearly positive but their history
        # was predominantly negative, we override the trend and suppress any
        # sadness-hint injection — the present emotion wins.
        positive_shift = self.drift.detect_positive_shift(emotion, confidence)

        # Check for hidden emotion hint (calibration insight)
        # SKIP this override entirely when a positive shift is detected —
        # we never want to re-inject sadness on top of a happy message.
        if not positive_shift:
            hidden_hint = self.memory.calibrator.get_hidden_emotion_hint(normalized)
            if hidden_hint and hidden_hint != emotion:
                emotion    = hidden_hint
                confidence = max(confidence, 0.55)

        # ── 4. Context signals ────────────────────────────────────────────────
        # Use 'shifting_positive' as the trend when a shift is detected so that
        # all downstream logic (build_response, Ollama prompt, UI) knows to
        # celebrate rather than probe for hidden sadness.
        if positive_shift:
            trend = "shifting_positive"
        else:
            trend = self.drift.get_trend()

        topic    = extract_topic(normalized)
        keywords = extract_context_keywords(normalized)
        severity = extract_severity(normalized)
        history  = self.memory.get_emotional_history(days=1)

        # ── 5. Point 6: Risk scoring ──────────────────────────────────────────
        # Positive-shift messages carry no meaningful risk — don't inflate score.
        if positive_shift:
            risk_score = 0.0
        else:
            risk_score = compute_risk_score(normalized, emotion, confidence, history)

        # ── 6. Point 2: Drift detection ───────────────────────────────────────
        # Suppress drift alerts during a positive shift — no "I noticed things
        # have been getting heavier" when user is clearly happy.
        if positive_shift:
            drift_info = {"type": "stable", "severity": 0, "alert": None}
        else:
            drift_info = self.drift.detect_drift()

        # ── 7. Special cases ──────────────────────────────────────────────────
        response = special_case_check(normalized, emotion)

        # ── 8. Risk-level response override (Point 6) ─────────────────────────
        # Never override with concern/alert when user is shifting positive.
        if response is None and not positive_shift:
            if risk_score >= 0.75:
                response = random.choice(ALERT_RESPONSES)
            elif risk_score >= 0.5:
                response = random.choice(CONCERN_RESPONSES)

        # ── 9. OLLAMA — PRIMARY RESPONSE GENERATOR ─────────────────────────
        if response is None:
            if positive_shift:
                # Celebrate the positive moment. Explicitly forbid sadness references.
                prompt = (
                    f"You are MindBridge, a warm private emotional support companion. "
                    f"The user just said: '{user_text}'. "
                    f"They are feeling {emotion} right now. "
                    f"IMPORTANT: Even though previous conversations may have involved "
                    f"sadness or difficult emotions, the user's CURRENT message is "
                    f"clearly positive and happy. "
                    f"DO NOT mention, reference, or allude to any past sadness. "
                    f"DO NOT ask what is weighing on their heart. "
                    f"DO NOT probe for hidden negative feelings. "
                    f"Simply celebrate their current positive moment warmly and naturally. "
                    f"Ask one light, upbeat follow-up question about what they shared. "
                    f"2-3 sentences max. Under 60 words. Sound like a happy friend."
                )
            else:
                trend_desc = {
                    "worsening":           "Their mood has been declining.",
                    "improving":           "Their mood has been improving.",
                    "persistent_negative": "They've been consistently negative.",
                }.get(trend, "")
                kg_insight = ""
                if topic:
                    strength = self.knowledge_graph.get_emotion_topic_strength(emotion, topic)
                    if strength > 0.5:
                        kg_insight = f"For this user, {emotion} often relates to {topic}."
                prompt = (
                    f"You are MindBridge, a warm private emotional support companion. "
                    f"User: '{user_text}'. Emotion: {emotion} ({confidence:.0%}). "
                    f"Topic: {topic or 'general'}. {trend_desc} {kg_insight} "
                    f"2-3 sentences: acknowledge specifically, ask one thoughtful question. "
                    f"Sound like a caring friend. No bullet points. Under 70 words. "
                    f"Never start with 'I'm sorry'. Vary your opening each time."
                )
            response = get_groq_response(prompt)

        # ── 10. Local response builder (Point 8) ──────────────────────────────
        if response is None:
            response = build_response(
                normalized, emotion, confidence, trend, topic,
                drift_info, self.knowledge_graph, self.memory,
                self.drift, self.msg_count, keywords, duration, severity,
            )

        # Avoid exact repeat
        attempts = 0
        while response in self._used and attempts < 3:
            response = build_response(
                normalized, emotion, confidence, trend, topic,
                drift_info, self.knowledge_graph, self.memory,
                self.drift, self.msg_count, keywords, duration, severity,
            )
            attempts += 1
        self._used.add(response)

        # ── 11. Update all memory systems ─────────────────────────────────────
        self.drift.update(emotion, confidence)
        # Point 3: Store vector, not text
        self.memory.record_emotion_vector(probs, emotion, topic, user_text)
        # Update knowledge graph (Point 7)
        self.knowledge_graph.graph = self.memory.data.get("knowledge_graph", {})
        # Update calibrator with this observation (Point 1)
        self.memory.calibrator.record(normalized, emotion, confidence)

        return {"response": response, "emotion": emotion, "confidence": confidence,
                "trend": trend, "topic": topic, "is_crisis": False,
                "is_medical": False, "risk_score": round(risk_score, 2)}

    def get_stats(self) -> dict:
        return {
            "user_id":          self.user_id,
            "sessions":         self.memory.sessions(),
            "total_messages":   self.memory.total_messages(),
            "emotion_counts":   self.memory.data["emotion_counts"],
            "session_trend":    self.drift.get_trend(),
            "dominant_today":   self.drift.get_dominant(),
            "dominant_ever":    self.memory.dominant_emotion(),
            "calibrator_obs":   self.memory.calibrator.total_observations,
            "knowledge_graph":  self.knowledge_graph.get_graph_summary(),
        }

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════════════════════
    #  RETRAIN COORDINATION HELPERS
    #  Communicate with retrain.py via a shared lock file and log file.
    #  retrain.py writes  → saved_model/retrain_lock.json  while running
    #  retrain.py writes  → saved_model/retrain_log.json   when done
    #  chatbot.py reads both files to detect and react to retraining
    # ══════════════════════════════════════════════════════════════════════════
    LOCK_FILE = "saved_model/retrain_lock.json"
    LOG_FILE  = "saved_model/retrain_log.json"

    def is_retrain_running() -> bool:
        """
        Returns True if retrain.py is currently running.
        retrain.py creates this lock file when it starts
        and deletes it when it finishes.
        """
        if not os.path.exists(LOCK_FILE):
            return False
        try:
            with open(LOCK_FILE) as f:
                data = json.load(f)
            # Lock file has a timestamp — if older than 10 minutes,
            # retrain probably crashed, ignore it
            started = datetime.datetime.fromisoformat(
                data.get("started", "2000-01-01T00:00:00")
            )
            age_minutes = (datetime.datetime.now() - started).seconds / 60
            return age_minutes < 10
        except Exception:
            return False

    def get_last_retrain_info() -> dict:
        """
        Reads the retrain log to find what happened last night.
        Returns dict with date, accuracy, clients, samples etc.
        Returns None if no log exists.
        """
        if not os.path.exists(LOG_FILE):
            return None
        try:
            with open(LOG_FILE) as f:
                data = json.load(f)
            rounds = data.get("rounds", [])
            if not rounds:
                return None
            # Return the most recent successful round
            for r in reversed(rounds):
                if r.get("result") == "success":
                    return r
        except Exception:
            pass
        return None

    def reload_model_from_disk(bot_ref) -> bool:
        """
        Reload the updated model_final.pt into the running chatbot session.
        Called after retrain completes — user gets improved model immediately
        without restarting the terminal.
        """
        try:
            bot_ref.model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            )
            bot_ref.model.eval()
            return True
        except Exception as e:
            print(f"  Warning: Could not reload model: {e}")
            return False

    def wait_for_retrain_to_finish(bot_ref):
        """
        Called when user types something and retrain is in progress.
        Shows freeze message, polls until retrain finishes,
        then reloads the updated model and unfreezes chat.
        """
        import time as _time

        started_at = datetime.datetime.now()

        print(f"\n  {'█'*54}")
        print(f"  ⏳  FL RETRAINING IN PROGRESS — PLEASE WAIT")
        print(f"  {'█'*54}")
        print(f"\n  MindBridge is currently updating its global model")
        print(f"  using today's emotional data from all users.")
        print(f"\n  This happens automatically every night at midnight.")
        print(f"  Duration: usually 30 seconds to 2 minutes.")
        print(f"\n  Please do not close this window.")
        print(f"  You will be notified the moment it is complete.\n")
        print(f"  Started at : {started_at.strftime('%I:%M:%S %p')}")
        print(f"  {'─'*54}")

        # Poll every 3 seconds until lock file disappears
        dots = 0
        while is_retrain_running():
            elapsed = int((datetime.datetime.now() - started_at).seconds)
            print(f"  Aggregating{'.' * (dots % 4):<4}  elapsed: {elapsed}s",
                  end="\r", flush=True)
            dots += 1
            _time.sleep(3)

        elapsed = int((datetime.datetime.now() - started_at).seconds)
        print(f"\n\n  {'█'*54}")
        print(f"  ✅  MODEL UPDATE COMPLETE — CHAT UNFROZEN")
        print(f"  {'█'*54}\n")
        print(f"  Completed at  : {datetime.datetime.now().strftime('%I:%M:%S %p')}")
        print(f"  Time taken    : {elapsed} seconds\n")

        # Show what improved
        info = get_last_retrain_info()
        if info:
            print(f"  Clients who contributed today : {info.get('clients', '?')}")
            print(f"  Training samples used         : {info.get('total_samples', '?')}")
            print(f"  Updated model accuracy        : "
                  f"{info.get('avg_accuracy', 0):.2%}")
            print(f"  All users receive this update on next session")

        # Reload updated model into running session
        print(f"\n  Loading updated model into your session...")
        if reload_model_from_disk(bot_ref):
            print(f"  ✅ Updated model loaded — you are now chatting "
                  f"with the improved model!\n")
        else:
            print(f"  ⚠️  Could not reload live — restart chatbot to get update\n")

        print(f"MindBridge: I'm back! The model just got smarter "
              f"from today's conversations across all users. "
              f"How are you feeling?\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  STARTUP
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MindBridge — Privacy-Preserving Emotion-Aware Assistant")
    print("  Emotion vectors stored. No raw text saved to disk.")
    print("  Daily FL retraining at midnight improves the model.")
    print("  Type 'help' for all commands.")
    print("=" * 60)

    uid = input("\nYour name (Enter for guest): ").strip() or "guest"
    bot = MindBridge(user_id=uid, model_path=MODEL_PATH)

    hour = datetime.datetime.now().hour
    tg   = ("Good morning" if 5 <= hour < 12 else "Good afternoon"
            if 12 <= hour < 17 else "Good evening" if 17 <= hour < 21 else "Hi")

    if bot.memory.is_returning():
        dominant = bot.memory.dominant_emotion()
        opening  = (f"{tg}, {uid}. I'm glad you came back. How are you feeling today?"
                    if dominant in NEGATIVE_EMOTIONS else
                    f"{tg}, {uid}! Good to see you again. How are you doing today?")
    else:
        opening = (
            f"{tg}! I'm MindBridge — your private emotional support companion. "
            f"Everything you share stays here. I learn your patterns over time "
            f"to understand you better. How are you feeling today?\n"
            f"  (Type 'help' for commands)"
        )

    print(f"\nMindBridge: {opening}\n")

    # ── Point 3: Show retrain update on startup ───────────────────────────────
    # Checks if retraining happened since last session and tells user
    last_retrain = get_last_retrain_info()
    if last_retrain:
        retrain_date = last_retrain.get("date", "")
        today_str    = str(datetime.date.today())
        yesterday    = str(datetime.date.today() - datetime.timedelta(days=1))

        # Show notification if retrain happened today or yesterday
        if retrain_date in [today_str, yesterday]:
            acc      = last_retrain.get("avg_accuracy", 0)
            clients  = last_retrain.get("clients", 0)
            samples  = last_retrain.get("total_samples", 0)
            end_time = last_retrain.get("end_time", "midnight")

            print(f"  ╔══════════════════════════════════════════════════╗")
            if retrain_date == today_str:
                print(f"  ║  🔄  MODEL UPDATED TONIGHT (retraining complete) ║")
            else:
                print(f"  ║  🔄  MODEL UPDATED LAST NIGHT                    ║")
            print(f"  ╠══════════════════════════════════════════════════╣")
            print(f"  ║  Completed at  : {end_time:<32}║")
            print(f"  ║  Users learned : {clients:<32}║")
            print(f"  ║  Samples used  : {samples:<32}║")
            print(f"  ║  New accuracy  : {f'{acc:.2%}':<32}║")
            print(f"  ║  You are now chatting with the improved model.   ║")
            print(f"  ╚══════════════════════════════════════════════════╝\n")

    # ── Main conversation loop ────────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nMindBridge: Take good care of yourself. "
                  "I'm here whenever you need.")
            break

        # ── Point 2: Check if retrain is running right now ────────────────────
        if is_retrain_running():
            wait_for_retrain_to_finish(bot)
            continue

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
            print(f"\nMindBridge: {random.choice([f'Take care, {uid}. I will be here whenever you need.', f'It was good talking with you, {uid}. Come back anytime.', 'Goodbye. You do not have to carry things alone.'])}")
            break

        if user_input.lower() == "help":
            print(f"\n{'─'*48}")
            print(f"  MindBridge Commands")
            print(f"{'─'*48}")
            print(f"  analyze            → Show emotion analysis graph")
            print(f"  history today      → Show today's conversations")
            print(f"  history YYYY-MM-DD → Show specific day's chat")
            print(f"  stats              → Show profile + knowledge graph")
            print(f"  help               → Show this menu")
            print(f"  quit               → Exit MindBridge")
            print(f"{'─'*48}\n")
            continue

        if user_input.lower() == "analyze":
            print("\nMindBridge: Generating your personal emotion analysis...\n")
            graph_path = generate_analysis_graph(bot.memory.data, uid)
            if graph_path:
                print(f"  ╔══════════════════════════════════════════╗")
                print(f"  ║      EMOTION ANALYSIS READY              ║")
                print(f"  ║  Saved → {graph_path:<32}║")
                print(f"  ╚══════════════════════════════════════════╝\n")
                try:
                    subprocess.Popen(["start", graph_path], shell=True)
                except Exception:
                    pass
                ec    = bot.memory.data.get("emotion_counts", {})
                total = bot.memory.data.get("total_messages", 0)
                sess  = bot.memory.data.get("sessions", 1)
                trend = bot.drift.get_trend()
                if ec:
                    dominant  = max(ec, key=ec.get)
                    pos_pct   = round(sum(v for k, v in ec.items()
                                         if k in POSITIVE_EMOTIONS) / total * 100) if total else 0
                    neg_pct   = round(sum(v for k, v in ec.items()
                                         if k in NEGATIVE_EMOTIONS) / total * 100) if total else 0
                    print(f"MindBridge: Your emotional patterns summary:\n")
                    print(f"  📊 Total messages     : {total}")
                    print(f"  🔄 Sessions           : {sess}")
                    print(f"  💡 Dominant emotion   : {dominant}")
                    print(f"  📈 Positive moments   : {pos_pct}%")
                    print(f"  📉 Challenging moments: {neg_pct}%")
                    print(f"  🌊 Current trend      : {trend}")
                    print(f"  🔒 Privacy: emotion vectors stored (no raw text)")
                    print(f"\n  Top emotions expressed:")
                    for em, cnt in sorted(ec.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {em:<15} {'█' * min(cnt, 20)} ({cnt})")
                    kg = bot.knowledge_graph.get_graph_summary()
                    if kg:
                        print(f"\n  Your Emotional Knowledge Graph:")
                        for emotion, topic in list(kg.items())[:5]:
                            print(f"    {emotion:<15} → {topic}")
                    print()
            else:
                print("\nMindBridge: Keep chatting and I'll build your profile!\n")
            continue

        if user_input.lower().startswith("history"):
            parts    = user_input.lower().split()
            date_str = parts[1] if len(parts) > 1 else str(datetime.date.today())
            if date_str == "today":
                date_str = str(datetime.date.today())
            # Note: only emotion metadata stored, not full text (Point 3)
            timeline = [
                e for e in bot.memory.data.get("emotion_timeline", [])
                if e.get("date") == date_str
            ]
            if not timeline:
                print(f"\nMindBridge: No emotion records found for {date_str}.")
                dates = sorted(set(e.get("date","") for e in
                                   bot.memory.data.get("emotion_timeline", [])))
                if dates:
                    print(f"  Available dates: {', '.join(dates[-10:])}")
                print()
            else:
                print(f"\n{'─'*50}")
                print(f"  Emotion History — {date_str}  (privacy: vectors only)")
                print(f"{'─'*50}")
                for entry in timeline:
                    ts    = entry.get("timestamp", "")[:16].replace("T", " ")
                    em    = entry.get("emotion", "?")
                    topic = entry.get("topic") or "general"
                    print(f"  [{ts}]  {em:<15}  topic: {topic}")
                print(f"{'─'*50}\n")
            continue

        if user_input.lower() == "stats":
            s = bot.get_stats()
            # Next retrain countdown
            now      = datetime.datetime.now()
            midnight = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                        + datetime.timedelta(days=1))
            mins_left = int((midnight - now).total_seconds() / 60)
            hrs  = mins_left // 60
            mins = mins_left % 60

            print(f"\n{'─'*48}")
            print(f"  Your MindBridge Profile")
            print(f"{'─'*48}")
            print(f"  Sessions            : {s['sessions']}")
            print(f"  Total messages      : {s['total_messages']}")
            print(f"  Session trend       : {s['session_trend']}")
            print(f"  Today dominant      : {s['dominant_today']}")
            print(f"  All-time dominant   : {s['dominant_ever']}")
            print(f"  Calibrator obs.     : {s['calibrator_obs']} (learning your patterns)")
            print(f"  Training samples    : {s.get('training_samples', 0)} collected today")
            print(f"  Next FL retrain     : in {hrs}h {mins}m (midnight)")
            print(f"  Privacy mode        : emotion vectors only (no text stored)")

            # Show last retrain info
            info = get_last_retrain_info()
            if info:
                print(f"\n  Last Retrain:")
                print(f"    Date      : {info.get('date')}")
                print(f"    Time      : {info.get('time')} → {info.get('end_time')}")
                print(f"    Clients   : {info.get('clients')}")
                print(f"    Accuracy  : {info.get('avg_accuracy', 0):.2%}")
                print(f"    Duration  : {info.get('elapsed_sec')}s")

            if s["knowledge_graph"]:
                print(f"\n  Your Emotional Knowledge Graph:")
                for emotion, topic in list(s["knowledge_graph"].items())[:6]:
                    print(f"    {emotion:<15} → {topic}")
            print(f"{'─'*48}\n")
            continue

        result = bot.respond(user_input)

        medical_tag = " 🏥 HEALTH"    if result.get("is_medical") else ""
        crisis_tag  = " ⚠️  CRISIS"   if result["is_crisis"]       else ""
        risk        = result.get("risk_score", 0)
        risk_tag    = f" 🔴 RISK:{risk:.2f}" if risk >= 0.5 and not result["is_crisis"] else ""

        print(f"\n  [{result['emotion']} | "
              f"{result['confidence']:.0%} | "
              f"trend: {result['trend']}"
              f"{medical_tag}{crisis_tag}{risk_tag}]")
        print(f"\nMindBridge: {result['response']}\n")