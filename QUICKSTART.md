# MindBridge — Quick Start Guide

## 🚀 Setup Steps

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# On Windows
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This will take 5-10 minutes as it downloads PyTorch, Transformers, etc.

### 4. Check if Model Exists
```bash
ls saved_model/model_final.pt
```

- **If exists**: Skip to step 6
- **If missing**: You need to train the model (see step 5)

### 5. Train Model (Optional — if model_final.pt missing)
```bash
# Quick local training (30-60 minutes on CPU)
python train_local.py

# OR federated learning simulation (1-2 hours)
python simulation.py
```

### 6. Install Ollama (Optional but Recommended)
Ollama provides better conversational responses.

1. Download from https://ollama.com/download
2. Install it
3. Run these commands:
```bash
ollama pull llama3
ollama serve
```

Keep the `ollama serve` terminal running in the background.

---

## 🎮 Running the Application

### **Option A: Local Client Mode (Recommended)**

This is the privacy-first mode — your data stays on YOUR machine.

```bash
python client.py
```

Then open in your browser:
```
http://localhost:5001
```

**Your personal data is stored in**: `client_data/` folder (on your machine only)

---

### **Option B: Central Server Mode**

Run the central server (for administrators or testing):

```bash
python app.py
```

Then open:
```
http://localhost:5000
```

---

## 🔧 Project Structure

```
MindBridge/
├── app.py                    # Central server (Flask)
├── client.py                 # Local client server (runs on your machine)
├── chatbot/chatbot.py        # Core emotion detection & chat logic
├── saved_model/
│   └── model_final.pt        # Trained DistilBERT model (256 MB)
├── data/
│   ├── goemotions_clean.csv  # Training dataset
│   └── clients_28class/      # Federated learning datasets
├── templates/index.html      # Web interface
├── client_data/              # YOUR LOCAL DATA (never sent to server)
│   ├── user_histories/       # Emotion memory
│   ├── user_profiles/        # Name, bio
│   ├── user_uploads/         # Photos you share
│   └── user_avatars/         # Profile picture
└── user_histories/           # Server-side FL gradients (anonymized)
```

---

## 🎭 Features

### Core Features
- **28-emotion detection**: sadness, joy, anger, fear, anxiety, loneliness, excitement, etc.
- **Crisis detection**: Detects suicide risk and provides helpline numbers
- **Personalized learning**: System learns YOUR unique emotional patterns
- **Temporal memory**: Tracks emotional changes over time

### Wellness Features
- 🧘 **Meditation Timer**: 5, 10, 15, 20 minute guided sessions
- 🧘‍♀️ **Yoga Guidance**: Emotion-adaptive yoga recommendations
- 🎬 **Movie Suggestions**: Mood-based recommendations
- 📊 **Analytics Dashboard**: Visualize your emotional journey

### Privacy Features
- All conversations stored locally in `client_data/`
- Only anonymized emotion vectors sent to server
- No raw text ever leaves your device
- HTTPS encryption for server communication

---

## ⚙️ Common Issues

### Issue: "No module named 'transformers'"
**Solution**: Make sure virtual environment is activated
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Model file not found"
**Solution**: Train the model first
```bash
python train_local.py
```

### Issue: Ollama not responding
**Solution**: Make sure Ollama is running
```bash
ollama serve
```
Keep this terminal open while using MindBridge.

### Issue: Port 5001 already in use
**Solution**: Use a different port
```bash
python client.py --port 5002
```

---

## 🧪 Testing

### Check if everything works:
```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Run client
python client.py

# 3. Open browser → http://localhost:5001

# 4. Try typing: "I feel sad today"
#    → Should detect 'sadness' emotion
```

---

## 📚 Technical Details

### Emotion Detection Model
- **Architecture**: DistilBERT-base-uncased + Linear Classifier
- **Input**: Text (max 32 tokens)
- **Output**: 28 emotion probabilities
- **Training**: Federated learning with FedAvg algorithm

### Federated Learning
- **Framework**: Flower (flwr)
- **Strategy**: FedAvg (Federated Averaging)
- **Privacy**: Only model weight updates shared, not data
- **Retraining**: Scheduled daily at midnight

### Crisis Detection
- **Algorithm**: Risk score = emotion_weight + keyword_score + history_score
- **Thresholds**: 
  - Low (0.0-0.3): Concern
  - Medium (0.3-0.6): Alert
  - High (0.6-1.0): Crisis
- **Resources**: Provides emergency helpline numbers

---

## 🆘 Emergency Contacts (India)

If you're in crisis:
- **AASRA**: 9820466627 (24/7)
- **iCall**: 9152987821 (Mon-Sat, 8 AM - 10 PM)
- **Vandrevala Foundation**: 1860-2662-345 (24/7)

---

## 🎓 Learning Resources

### Understanding the Code
- `chatbot/chatbot.py`: Core emotion detection logic
- `app.py`: Central Flask server
- `client.py`: Local client implementation
- `simulation.py`: Federated learning training

### Key Algorithms
1. **Emotion Calibration**: Learns your personal emotional language
2. **Drift Detection**: Monitors emotional trajectory changes
3. **Knowledge Graph**: Maps emotion-topic relationships
4. **Risk Scoring**: Graduated crisis risk assessment

---

## 🤝 Contributing

This is an academic research project by:
- **Nuthana T M** (1MS24IS410)
- **Shreya K G** (1MS24IS412)

Ramaiah Institute of Technology, Bangalore

---

## 📝 Next Steps

1. ✅ Install dependencies
2. ✅ Train or download model
3. ✅ Install Ollama (optional)
4. ✅ Run `python client.py`
5. ✅ Open http://localhost:5001
6. ✅ Start chatting!

**Need help?** Check the main README.md for detailed documentation.
