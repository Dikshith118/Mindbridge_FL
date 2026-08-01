# How MindBridge Works — System Architecture & Flow

## Overview
MindBridge is a **privacy-first emotional support chatbot** that uses **Federated Learning** to personalize emotion detection while keeping your data completely private on your device.

---

## 🏗️ Architecture

### Two Operating Modes

#### **1. Local Client Mode (Recommended)**
```
User Device (Port 5001)
├── Flask Client Server (client.py)
├── MindBridge Chatbot Engine
├── Local Model (model_final.pt)
└── Private Data Storage (client_data/)
    ├── user_histories/      ← Emotion memory
    ├── user_profiles/       ← Name, bio
    ├── user_avatars/        ← Profile pictures
    └── user_uploads/        ← Shared photos
```

**Data Flow:**
- Everything stays on YOUR machine
- Only anonymized emotion vectors sent to server
- No raw conversation text ever leaves your device

#### **2. Central Server Mode**
```
Server (Port 5000)
├── Flask Server (app.py)
├── Model Training Pipeline
└── Federated Learning Aggregator
```

---

## 🔄 How It Works When Running

### **Step 1: Startup**

When you run `python client.py`:

1. **Virtual Environment Check**
   - Activates Python environment
   - Loads dependencies (transformers, torch, flask)

2. **Model Synchronization**
   - Checks if `client_data/model_cache/model_final.pt` exists locally
   - If missing, downloads from central server
   - Falls back to `saved_model/model_final.pt` if server unavailable

3. **Chatbot Initialization**
   - Loads DistilBERT tokenizer
   - Initializes 28-emotion classifier
   - Prepares local storage directories

4. **Server Starts**
   - Flask server binds to `0.0.0.0:5001`
   - Opens web interface at `http://localhost:5001`

---

### **Step 2: User Login**

When you enter your name and click "Start":

1. **Session Creation**
   ```
   POST /start
   ├── Creates MindBridge instance for your user ID
   ├── Restores emotion memory from client_data/user_histories/
   ├── Loads personal calibration data
   └── Returns personalized greeting
   ```

2. **Memory Restoration**
   - Loads your previous conversations (emotion vectors only, not text)
   - Restores emotion counts, daily logs
   - Rebuilds personal emotion calibrator
   - Reconstructs knowledge graph

3. **Profile Loading**
   - Reads `{name}_profile.json` from `client_data/user_profiles/`
   - Checks for avatar in `client_data/user_avatars/`

---

### **Step 3: Chat Message Flow**

When you send a message like *"I feel sad today"*:

#### **A. Text Preprocessing**
```python
# Normalize and spell-correct
"i feel sadd today" → "i feel sad today"
```

#### **B. Emotion Detection** (3-layer system)

**Layer 1: Global DistilBERT Model**
```
Input: "i feel sad today"
↓
Tokenization (max 32 tokens)
↓
DistilBERT Encoder (768-dim embedding)
↓
Classifier Head (28 emotions)
↓
Raw Probabilities: [sadness: 0.78, grief: 0.12, ...]
```

**Layer 2: Personal Calibration**
```
System learns YOUR emotional language:
- If you say "I'm fine" but model detected sadness 3+ times before
- Calibrator adjusts: "fine" → adds weight to sadness
- Adapts to how YOU express emotions uniquely
```

**Layer 3: Keyword Override**
```
Positive signal detection:
- Checks for unambiguous positive words ("excited", "happy", "proud")
- Overrides model if it misclassified positive as negative
- Prevents false negatives when you're genuinely happy
```

#### **C. Crisis Detection**
```
Risk Score = Emotion Weight + Keyword Score + History Score

Emotion Weight:
- grief: 0.9, sadness: 0.8, fear: 0.7
- joy: -0.3, love: -0.3

Keyword Score:
- "want to die", "suicide" → +0.7
- "feel hopeless" → +0.4

History Score:
- 5 consecutive negative emotions → +0.3

Final Score:
- 0.0-0.3: Concern (gentle check-in)
- 0.3-0.6: Alert (suggests helpline)
- 0.6-1.0: Crisis (immediate helpline numbers)
```

#### **D. Emotion Drift Detection**
```
Tracks last 8 emotions:
- happy → happy → neutral → sad → very sad
- Detects: "sudden_drop" (alert: something changed?)
- Detects: "gradual_decline" (pattern over session)
- Detects: "sustained_negative" (5+ consecutive)
```

#### **E. Response Generation**

**Two-stage system:**

1. **MindBridge Core Response**
   - Emotion-specific templates
   - Sadness → empathetic, slow pacing
   - Anxiety → grounding techniques
   - Joy → celebratory, energetic

2. **Ollama LLM Enhancement** (if installed)
   ```
   Prompt: "User expressed sadness with message 'I feel sad today'.
           Respond as MindBridge — warm, empathetic, 2-3 sentences."
   ↓
   Ollama (llama3)
   ↓
   Enhanced response (more natural, contextual)
   ```

#### **F. Memory Storage**
```json
Stored locally in client_data/user_histories/{name}_emotional_memory.json:
{
  "emotion_timeline": [
    {
      "emotion": "sadness",
      "confidence": 0.78,
      "vector": [0.02, 0.01, 0.03, ..., 0.78, ...],  // 28 probs
      "date": "2026-08-01",
      "timestamp": "2026-08-01T14:32:17",
      "topic": "mental_health"
    }
  ],
  "emotion_counts": {"sadness": 5, "joy": 3, ...},
  "daily_log": {"2026-08-01": {"sadness": 2, "joy": 1}},
  "calibrator_data": {...}  // personal emotion language patterns
}
```

**Key Privacy Feature:** Raw text is NEVER stored — only emotion vectors.

---

### **Step 4: Federated Learning (Background)**

Every 5 messages, anonymized data is sent to server:

#### **What Gets Sent:**
```json
{
  "user_id": "client_05550398",  // anonymized hash
  "training_samples": [
    {
      "text": "user expressed sadness",  // pseudo-text, not real
      "label": "sadness",
      "confidence": 0.78,
      "date": "2026-08-01"
    }
  ],
  "emotion_counts": {"sadness": 5, "joy": 3},
  "daily_log": {"2026-08-01": {"sadness": 2}}
}
```

#### **What NEVER Gets Sent:**
- ❌ Your actual typed text
- ❌ Your name
- ❌ Your photos
- ❌ Your calibrator data (personal patterns)
- ❌ Your profile

#### **Server Retraining (Daily at Midnight)**
```
1. Collects gradients from ALL clients
2. FedAvg: Averages emotion patterns
3. Trains updated global model
4. Clients download new model (improved accuracy)
```

---

### **Step 5: Advanced Features**

#### **Knowledge Graph**
```
Tracks emotion-topic relationships:
"work" → [stress: 12, anxiety: 8, pride: 3]
"family" → [joy: 15, love: 10, annoyance: 2]

Used to understand: "When you talk about work, I notice stress often comes up."
```

#### **Drift Alerts**
```
Session timeline: joy → joy → neutral → sad → grief
                  ↓
Drift Detector: "I noticed a sudden shift in how you're feeling.
                 Something seems to have changed — would you like
                 to talk about what happened?"
```

#### **Wellness Features**
- **Meditation Timer**: 5/10/15/20 min guided sessions
- **Yoga Recommendations**: Emotion-adaptive poses
- **Movie Suggestions**: Mood-based recommendations
- **Analytics Dashboard**: Emotion trends over time

---

## 🔐 Privacy Architecture

### **Three Privacy Layers:**

1. **Local Storage**
   - All personal data in `client_data/` (never leaves your device)
   - Conversation text never persisted (only emotion vectors)

2. **Anonymized FL Gradient**
   - Only emotion labels + pseudo-text sent to server
   - User ID is a one-way hash (can't trace back)

3. **HTTPS Encryption** (when server configured)
   - FL gradient encrypted during transmission
   - Server never decrypts personal data (doesn't receive any)

---

## 🧠 AI Model Architecture

### **DistilBERT Emotion Classifier**
```
Input Text (max 32 tokens)
↓
DistilBERT-base-uncased (66M params)
├── Frozen (no training on client)
└── 768-dimensional embeddings
↓
Dropout (0.3)
↓
Linear Classifier (768 → 28 emotions)
↓
Softmax → Probability Distribution
```

### **28 Emotion Labels (GoEmotions Dataset)**
```
Positive: joy, love, gratitude, excitement, admiration, amusement, 
          approval, caring, optimism, pride, relief, surprise

Negative: sadness, grief, anger, annoyance, fear, disappointment,
          disapproval, disgust, embarrassment, nervousness, remorse

Ambiguous: confusion, curiosity, desire, realization, neutral
```

---

## 🚀 Startup Sequence Summary

```
1. python client.py
2. ├── Load virtual environment
3. ├── Initialize Flask (port 5001)
4. ├── Check model cache
5. │   ├── Exists locally? ✓ Load
6. │   └── Missing? Download from server
7. ├── Load chatbot/chatbot.py
8. ├── Initialize DistilBERT
9. ├── Restore user memories (if returning)
10. └── Start web server → http://localhost:5001
11. 
12. User opens browser → enters name → clicks Start
13. ├── POST /start
14. ├── Session created
15. ├── Memory restored from client_data/
16. └── Greeting displayed
17. 
18. User types message
19. ├── POST /chat
20. ├── Text preprocessing
21. ├── Emotion detection (3 layers)
22. ├── Crisis check
23. ├── Drift detection
24. ├── Response generation (MindBridge + Ollama)
25. ├── Memory updated (local only)
26. └── Response displayed
27. 
28. Every 5 messages:
29. ├── Background thread starts
30. ├── Anonymized gradient prepared
31. ├── POST /receive_gradient → server
32. └── Server saves for FL retraining
33. 
34. User clicks Logout
35. ├── POST /logout
36. ├── Final gradient sent to server
37. ├── Memory saved to client_data/
38. └── Session cleared
```

---

## 📊 Example Session Flow

```
User: "I'm so excited about my new job!"
↓
[Emotion Detection]
- DistilBERT: excitement (0.72), joy (0.18)
- Keyword Override: "excited" found → excitement (0.75) ✓
- Crisis Check: PASS
- Drift: stable → improving (previous was neutral)
↓
[Response]
- Template: "That's wonderful news! Starting something new is such an exciting chapter."
- Ollama: "I can feel your excitement! What are you most looking forward to about it?"
↓
[Memory Storage]
- emotion_timeline += {emotion: "excitement", confidence: 0.75, vector: [...]}
- emotion_counts["excitement"] += 1
- calibrator.record("excited about job", "excitement", 0.75)
- knowledge_graph.link("work", "excitement")
↓
[FL Gradient]
- training_sample: {text: "user expressed excitement", label: "excitement"}
```

---

## 🛠️ Tech Stack

**Backend:**
- Flask (web server)
- PyTorch (deep learning)
- Transformers (DistilBERT)
- Ollama (LLM responses)

**Frontend:**
- Vanilla JavaScript
- HTML5 + CSS3
- Chart.js (analytics)

**Model:**
- DistilBERT-base-uncased (66M params)
- Custom 28-class emotion classifier
- Trained on GoEmotions dataset (58k labeled examples)

**Privacy:**
- Client-side storage (localStorage + server files)
- Federated Learning (FedAvg)
- Differential privacy (anonymized gradients)

---

## 🔄 Retraining Pipeline

When server runs `python retrain.py`:

```
1. Collect gradients from user_histories/
2. Load global model (saved_model/model_final.pt)
3. For each client gradient:
   ├── Simulate local training (3 epochs)
   ├── Compute weight updates
   └── Store client weights
4. FedAvg: Average all client weights
5. Update global model
6. Save → saved_model/model_final.pt
7. Clients download updated model on next login
```

---

## 💡 Key Innovations

1. **Privacy-Preserving Memory**: Stores emotion vectors, not text
2. **Personal Calibration**: Learns YOUR emotional language
3. **Emotion Drift Detection**: Proactive mental health monitoring
4. **Graduated Crisis Response**: Not binary, understands degree of risk
5. **Federated Learning**: Global improvement, local privacy
6. **Keyword Override**: Prevents false negatives on positive emotions
7. **Knowledge Graph**: Context-aware emotional understanding

---

## 🆘 Emergency Contacts (India)

When crisis detected, system provides:
- **AASRA**: 9820466627 (24/7)
- **iCall**: 9152987821 (Mon-Sat, 8 AM - 10 PM)
- **Vandrevala Foundation**: 1860-2662-345 (24/7)

---

## 📝 Verification

To verify your data stays local:

```bash
# Check local storage location
curl http://localhost:5001/client_info

# Returns:
{
  "storage_location": "/path/to/MindBridge/client_data",
  "server_url": "https://192.168.1.63:5000",
  "model_cached": true,
  "local_files": {
    "user_histories": ["YourName_emotional_memory.json"],
    "user_profiles": ["YourName_profile.json"],
    "user_avatars": ["YourName_avatar.jpg"],
    "user_uploads": []
  }
}
```

Your actual conversation text is NEVER in these files — only emotion vectors and metadata.

---

## 🎯 Design Philosophy

**Privacy-First:**
- Your data = YOUR property
- Server = computation helper, not data collector
- You can verify: check `client_data/` folder — no raw text stored

**Federated Learning:**
- Train together, learn separately
- Your personal patterns stay personal
- Global model improves for everyone

**Human-Centered:**
- Not a therapist replacement
- Complements professional help
- Always points to real helplines in crisis

---

**Made with care by the MindBridge team**  
Ramaiah Institute of Technology, Bangalore
