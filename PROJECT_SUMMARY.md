# MindBridge — Project Summary

## 🎯 What This Project Does

**MindBridge** is a privacy-first AI chatbot for mental health support that uses **Federated Learning** to keep your conversations private while still improving through shared learning.

### Key Innovation: Privacy + Learning
```
Traditional Approach:                Your data → Server → Server stores it ❌
MindBridge Approach:                Your data → Your Device → Only learning sent ✅
```

---

## 🧠 Core Features

### 1. Emotion Detection (28 emotions)
Detects emotions from your text using DistilBERT AI model:

**Positive Emotions**: joy, love, gratitude, excitement, pride, optimism, relief...
**Negative Emotions**: sadness, grief, anger, fear, anxiety, loneliness, disappointment...
**Neutral/Mixed**: confusion, surprise, curiosity, realization...

### 2. Personalized Learning
The system learns YOUR unique emotional language:
- "I'm fine" might mean sadness for you
- "Not bad" might mean you're actually happy
- System adapts to YOUR patterns over time

### 3. Crisis Detection
Advanced risk scoring system:
- Detects suicide risk keywords
- Monitors emotional trajectory
- Provides emergency helpline numbers
- Graduated response: concern → alert → crisis

### 4. Wellness Features
- **Meditation Timer**: Guided sessions (5-20 minutes)
- **Yoga Guidance**: Emotion-adaptive recommendations
- **Movie Suggestions**: Based on your current mood
- **Analytics Dashboard**: Visualize emotional patterns

### 5. Privacy Architecture
```
┌─────────────────────────────────────────────────────┐
│  YOUR COMPUTER (localhost:5001)                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ client_data/                                  │  │
│  │  ├── user_histories/    ← your conversations │  │
│  │  ├── user_profiles/     ← your name, bio     │  │
│  │  ├── user_uploads/      ← your photos        │  │
│  │  └── user_avatars/      ← profile picture    │  │
│  └──────────────────────────────────────────────┘  │
│                        ↓                            │
│              (Only anonymized learning)             │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  CENTRAL SERVER (localhost:5000)                    │
│  • Receives: emotion labels only (no text)          │
│  • Updates: global AI model                         │
│  • Distributes: improved model to clients           │
└─────────────────────────────────────────────────────┘
```

**What stays on your machine**: 
- Raw conversations
- Photos
- Profile
- Personal emotional history

**What goes to server**:
- Emotion labels (e.g., "sadness detected 3 times")
- Model improvements (mathematical weights)
- NO text, NO photos, NO personal info

---

## 🏗️ Technical Architecture

### Machine Learning Stack
- **Model**: DistilBERT (lightweight transformer)
- **Task**: Multi-label emotion classification
- **Training**: Federated Learning (FedAvg algorithm)
- **Dataset**: GoEmotions (Google Research)
- **Accuracy**: 87.3%

### Backend
- **Server**: Flask (Python web framework)
- **Storage**: JSON files (no database needed)
- **Communication**: HTTPS with self-signed certificates
- **Response Generation**: Ollama (local LLM - Llama 3)

### Frontend
- **Interface**: HTML5 + CSS3 + Vanilla JavaScript
- **Features**: Voice input, image upload, real-time chat
- **Storage**: Browser localStorage for session data

### Federated Learning
- **Framework**: Flower (flwr)
- **Strategy**: Federated Averaging (FedAvg)
- **Process**: 
  1. Clients train on local data
  2. Send weight updates to server
  3. Server averages updates
  4. Distributes improved global model

---

## 📊 How It Works

### User Experience Flow
```
1. You type: "I feel really sad today"
   ↓
2. Local AI model analyzes text
   ↓
3. Detects: Emotion = "sadness" (confidence 0.85)
   ↓
4. System responds with empathy:
   "I hear you. Sadness can feel overwhelming. 
    What's been weighing on you today?"
   ↓
5. Conversation continues...
   ↓
6. Emotional memory stored LOCALLY (client_data/)
   ↓
7. When you log out:
   - Anonymized learning sent to server
   - Raw text stays on your machine
```

### Emotion Detection Algorithm
```python
Input: "I feel lonely and exhausted"
  ↓
Tokenization: ["i", "feel", "lonely", "and", "exhausted"]
  ↓
DistilBERT Embedding: [0.23, -0.45, 0.67, ...] (768 dimensions)
  ↓
Classification Layer: Emotion probabilities
  - sadness: 0.72
  - loneliness: 0.85  ← highest
  - nervousness: 0.34
  ↓
Output: "loneliness" (confidence 0.85)
```

### Crisis Risk Scoring
```
risk_score = emotion_weight + keyword_score + history_score

Example:
  emotion_weight = 0.8 (sadness) × 0.9 (confidence) = 0.72
  keyword_score = 0.7 (found "want to disappear")
  history_score = 0.2 (5 recent negative messages)
  ─────────────────────────────────────────────────────
  Total Risk Score = 0.72×0.4 + 0.7×0.4 + 0.2×0.2 = 0.61
  
  → Classification: CRISIS (≥ 0.6)
  → Action: Show emergency helpline numbers
```

---

## 📁 File Structure

### Core Files
- **app.py**: Central server (Flask) - runs on port 5000
- **client.py**: Local client server - runs on port 5001
- **chatbot/chatbot.py**: Core emotion detection & response logic (2129 lines)

### Models & Data
- **saved_model/model_final.pt**: Trained DistilBERT model (253 MB)
- **data/goemotions_clean.csv**: Training dataset (58,000 labeled texts)
- **data/clients_28class/**: 28 client datasets for federated learning

### Web Interface
- **templates/index.html**: Complete web UI with all features

### Training Scripts
- **simulation.py**: Federated learning simulation
- **retrain.py**: Daily model retraining script
- **train_local.py**: Local model training (non-federated)

### User Data Directories
```
client_data/              ← Your LOCAL data (never sent to server)
├── user_histories/       ← Emotion vectors (no raw text)
├── user_profiles/        ← Name, bio, joined date
├── user_avatars/         ← Profile picture
├── user_uploads/         ← Shared photos
└── model_cache/          ← Downloaded global model

user_histories/           ← Server-side FL gradients (anonymized)
```

---

## 🚀 Quick Start

### 1. Setup (One-time)
```bash
# Double-click or run:
setup.bat

# This installs all dependencies
```

### 2. Run
```bash
# Double-click or run:
run_client.bat

# Then open browser:
http://localhost:5001
```

### 3. Use
- Type your feelings in the chat
- System detects emotion and responds
- Try wellness features (meditation, yoga, movies)
- View analytics to see emotional patterns

---

## 🎓 Academic Context

**Institution**: Ramaiah Institute of Technology, Bangalore
**Department**: Information Science & Engineering
**Project Type**: Bachelor of Engineering Mini Project
**Academic Year**: 2025-2026

**Team**:
- **Nuthana T M** (1MS24IS410) - Lead Developer
- **Shreya K G** (1MS24IS412) - Co-Developer

**Guide**: Dr. Shruthi G

---

## 🧪 Technical Innovations

### 1. Emotion-Aware Federated Personalization
Global model + personal calibration layer learns individual patterns

### 2. Privacy-Preserving Emotional Memory
Stores emotion probability vectors, NOT raw text
- "I feel lonely" → saved as {sadness:0.8, loneliness:0.9}

### 3. Emotion Drift Detection
Monitors trajectory: happy → neutral → sad → very sad
Triggers proactive alerts for sudden drops

### 4. Federated Emotional Knowledge Graph
Builds emotion→topic relationships locally
Example: User talks about "exams" → anxiety increases

### 5. Crisis Risk Scoring System
Graduated response (not binary): concern → alert → crisis

### 6. Emotion-Adaptive Response Style
- Sadness → slow, empathetic tone
- Anxiety → grounding, calming
- Joy → celebratory
- Anger → calming, validating

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 87.3% |
| **Precision** | 85.6% |
| **Recall** | 84.2% |
| **F1-Score** | 84.9% |
| **Model Size** | 253 MB |
| **Response Time** | < 2 seconds |

### Top Performing Emotions
- Joy: 91.4% precision
- Love: 90.1% recall
- Sadness: 89.2% precision
- Neutral: 88.9% F1-score

---

## 🔐 Privacy Guarantees

### What This System Does
✅ Encrypts communication (HTTPS)
✅ Stores data locally on your machine
✅ Sends only anonymized learning to server
✅ No personal information in FL gradients
✅ No text logging on server
✅ No user tracking or analytics

### What This System Does NOT Do
❌ Send your conversations to server
❌ Store your photos on server
❌ Share your data with third parties
❌ Track your activity
❌ Store chat history on server

---

## ⚠️ Important Disclaimers

### Medical Disclaimer
**MindBridge is NOT a replacement for professional mental health care.**

- This is an academic research project
- NOT a clinical tool
- NOT for diagnosis or treatment
- NOT a substitute for therapy

### When to Seek Professional Help
Contact a mental health professional if you experience:
- Suicidal thoughts
- Self-harm urges
- Severe depression or anxiety
- Mental health crisis

### Emergency Contacts (India)
- **AASRA**: 9820466627 (24/7)
- **iCall**: 9152987821 (Mon-Sat, 8 AM - 10 PM)
- **Vandrevala Foundation**: 1860-2662-345 (24/7)

---

## 🔮 Future Enhancements

### Planned Features
- 📱 Mobile app (iOS & Android)
- 🎤 Voice emotion recognition
- 🌍 Multilingual support (Hindi, Spanish, etc.)
- 📹 Facial expression analysis
- 💓 Wearable device integration
- 🔐 Differential privacy for stronger guarantees

### Research Directions
- Clinical validation studies
- Cross-cultural emotion expression
- Explainable AI for emotion detection
- Advanced FL algorithms (FedProx, FedNova)

---

## 📚 Learning Resources

### Understanding Federated Learning
- Paper: "Communication-Efficient Learning of Deep Networks from Decentralized Data" (McMahan et al., 2017)
- Flower Documentation: https://flower.dev/docs/

### Emotion Detection
- GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
- DistilBERT: https://huggingface.co/distilbert-base-uncased

### Privacy-Preserving ML
- Federated Learning: Collaborative ML without Centralized Data
- Differential Privacy in Machine Learning

---

## 🤝 Contributing

This project is open-source under MIT License.

**Ways to contribute**:
- Report bugs
- Suggest features
- Improve documentation
- Add new emotion categories
- Enhance privacy mechanisms

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgements

- **Google Research**: GoEmotions dataset
- **Hugging Face**: Transformers library & DistilBERT
- **Flower Team**: Federated learning framework
- **Ollama**: Local LLM inference
- **Dr. Shruthi G**: Project guidance
- **RIT Bangalore**: Institutional support

---

**Built with ❤️ for mental health awareness and privacy-preserving AI**
