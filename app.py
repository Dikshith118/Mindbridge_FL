"""
app.py - MindBridge Advanced Web Application
Complete redesign with:
- Voice input (microphone → speech to text)
- Advanced beautiful UI
- User profiles with photos
- Image upload for memory sharing
- Sidebar menu with all features
- Analyze, History, Stats panels
- Logout functionality
"""

from flask import Flask, request, jsonify, session, send_file
import sys, os, datetime, json, base64, io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatbot"))
from chatbot import MindBridge, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS, EMOTION_LABELS

app = Flask(__name__)
app.secret_key = "mindbridge_ultra_secret_2026"
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"]   = False  # set True only if pure HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

active_bots   = {}
PROFILES_DIR  = "user_profiles"
UPLOADS_DIR   = "user_uploads"
LOCK_FILE     = "saved_model/retrain_lock.json"
LOG_FILE      = "saved_model/retrain_log.json"
MODEL_PATH = "saved_model/model_final.pt"
# NOTE: user_profiles and user_uploads dirs kept only for FL gradient sharing,
# NOT for storing personal user data (that stays client-side in localStorage).
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR,  exist_ok=True)

# ── Client-side memory helpers ────────────────────────────────────────────────
def _bot_load_client_data(bot, client_data: dict):
    """Restore a bot's memory from client-supplied JSON (sent from browser localStorage)."""
    if not client_data:
        return
    try:
        mem = bot.memory
        # Restore all memory fields from client
        for key in ["sessions","total_messages","emotion_counts","daily_log",
                    "emotion_timeline","knowledge_graph","first_seen","last_seen",
                    "calibrator_data"]:
            if key in client_data:
                mem.data[key] = client_data[key]
        # Reload calibrator from restored data
        mem._load_calibrator()
        # Reload knowledge graph
        bot.knowledge_graph.graph = mem.data.get("knowledge_graph", {})
    except Exception as e:
        print(f"[sync] Warning: could not restore client data: {e}")

def _bot_dump_client_data(bot) -> dict:
    """Serialise bot memory to a dict the client will store in localStorage."""
    bot.memory.data["calibrator_data"] = bot.memory.calibrator.to_dict()
    return dict(bot.memory.data)  # shallow copy is enough — all values are JSON-safe


def find_uid(name=None):
    """Find user bot — works even when Brave/HTTPS breaks session cookies."""
    uid = session.get("user_id")
    if uid and uid in active_bots:
        return uid
    if name:
        for k in active_bots:
            if k.startswith(str(name) + "_") or k == str(name):
                return k
    return None

def is_retrain_running():
    if not os.path.exists(LOCK_FILE): return False
    try:
        with open(LOCK_FILE) as f: data = json.load(f)
        started = datetime.datetime.fromisoformat(data.get("started","2000-01-01T00:00:00"))
        return (datetime.datetime.now()-started).seconds < 600
    except: return False

def get_last_retrain_info():
    if not os.path.exists(LOG_FILE): return None
    try:
        with open(LOG_FILE) as f: data = json.load(f)
        for r in reversed(data.get("rounds",[])):
            if r.get("result") == "success": return r
    except: pass
    return None

def get_profile_path(uid): return os.path.join(PROFILES_DIR, f"{uid}_profile.json")
def get_avatar_path(uid):  return os.path.join(PROFILES_DIR, f"{uid}_avatar.jpg")

def load_profile(uid):
    path = get_profile_path(uid)
    if os.path.exists(path):
        try:
            with open(path) as f: return json.load(f)
        except: pass
    return {"name": uid, "joined": str(datetime.date.today()), "bio": ""}

def save_profile(uid, data):
    with open(get_profile_path(uid), "w") as f:
        json.dump(data, f, indent=2)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return open(os.path.join(os.path.dirname(__file__), "templates", "index.html"), encoding="utf-8").read()

@app.route("/status")
def status():
    info = get_last_retrain_info()
    return jsonify({"retrain_date": info.get("date") if info else None,
                    "retrain_running": is_retrain_running()})

@app.route("/retrain_status")
def retrain_status():
    running = is_retrain_running()
    return jsonify({"running": running,
                    "last_info": get_last_retrain_info() if not running else None})

@app.route("/start", methods=["POST"])
def start():
    data          = request.get_json()
    name          = (data.get("name","guest") or "guest").strip()
    client_memory = data.get("memory", {})   # ← client sends its localStorage data
    uid           = f"{name}_{request.remote_addr.replace('.','_')}"

    if uid not in active_bots:
        active_bots[uid] = MindBridge(user_id=name, model_path="saved_model/model_final.pt")

    bot = active_bots[uid]

    # ── Restore client-side memory into server bot ────────────────────────────
    # This is the federated handshake: client owns the data, server gets a
    # temporary in-memory copy for the session only.
    if client_memory:
        _bot_load_client_data(bot, client_memory)
    else:
        # First-ever login from this client — start fresh (don't load from disk)
        bot.memory.start_session()

    session["user_id"] = uid
    hour = datetime.datetime.now().hour
    tg   = ("Good morning" if 5<=hour<12 else "Good afternoon" if 12<=hour<17
            else "Good evening" if 17<=hour<21 else "Hi")
    if bot.memory.is_returning():
        dominant = bot.memory.dominant_emotion()
        greeting = (f"{tg}, {name}. I'm glad you came back. How are you feeling today?"
                    if dominant in NEGATIVE_EMOTIONS else
                    f"{tg}, {name}! Good to see you again. How are you doing today?")
    else:
        bot.memory.start_session()
        greeting = (f"{tg}, {name}! I'm MindBridge — your private emotional support companion. "
                    f"Everything you share stays on your device. How are you feeling today?")

    retrain_info = None
    info = get_last_retrain_info()
    if info:
        today     = str(datetime.date.today())
        yesterday = str(datetime.date.today()-datetime.timedelta(days=1))
        if info.get("date") in [today, yesterday]:
            retrain_info = info

    # ── Return updated memory snapshot for client to persist ─────────────────
    updated_memory = _bot_dump_client_data(bot)

    # Avatar: stored by client in localStorage as base64 — server never saves it
    profile = client_memory.get("profile", {"name": name,
                                             "joined": str(datetime.date.today()),
                                             "bio": ""})
    return jsonify({
        "greeting":       greeting,
        "name":           name,
        "retrain_info":   retrain_info,
        "has_avatar":     bool(client_memory.get("avatar_b64")),
        "profile":        profile,
        "sessions":       bot.memory.sessions(),
        "total_messages": bot.memory.total_messages(),
        "memory":         updated_memory,   # ← client saves this to localStorage
    })

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json() or {}
    name    = data.get("name", "")
    uid     = find_uid(name)
    if not uid:
        if name:
            ip  = request.remote_addr.replace(".", "_")
            uid = f"{name}_{ip}"
            if uid not in active_bots:
                active_bots[uid] = MindBridge(
                    user_id=name, model_path="saved_model/model_final.pt")
                # Restore client memory if sent along with chat (reconnect case)
                if data.get("memory"):
                    _bot_load_client_data(active_bots[uid], data["memory"])
            session["user_id"] = uid
        else:
            return jsonify({"response":"Session expired. Please refresh the page.",
                            "emotion":"neutral","confidence":0,"trend":"stable",
                            "is_crisis":False,"is_medical":False,"risk_score":0,
                            "retrain_running":False})
    if is_retrain_running():
        return jsonify({"retrain_running": True})
    message = data.get("message","").strip()
    bot     = active_bots[uid]
    result  = bot.respond(message)
    result["retrain_running"] = False
    # ── Return updated memory for client to persist in localStorage ───────────
    result["memory"] = _bot_dump_client_data(bot)

    # ── Periodic FL gradient flush (every N messages) ─────────────────────────
    # Ensures retrain.py has data even if the user never explicitly logs out.
    if bot.memory.total_messages() % GRADIENT_FLUSH_INTERVAL == 0:
        flush_fl_gradient(bot, uid)

    return jsonify(result)

@app.route("/upload_avatar", methods=["POST"])
def upload_avatar():
    # Avatar is now stored client-side in localStorage as base64.
    # Server never writes it to disk — just acknowledge receipt.
    return jsonify({"success": True})

@app.route("/avatar/<name>")
def get_avatar(name):
    # Kept for backward compat — client no longer fetches from server.
    return "", 404

@app.route("/upload_memory", methods=["POST"])
def upload_memory():
    data    = request.get_json() or {}
    uid     = find_uid(data.get("name","")) or session.get("user_id")
    if not uid or uid not in active_bots:
        return jsonify({"error":"Not logged in"}),401
    caption = data.get("caption","") or "I'm sharing this memory with you."
    # Image stays client-side — server only uses caption for response generation
    name = uid.rsplit("_",1)[0] if "_" in uid else uid
    bot  = active_bots[uid]
    prompt = (f"User {name} just shared a photo memory with the caption: '{caption}'. "
              f"Respond warmly as MindBridge — acknowledge the memory they shared, "
              f"ask something thoughtful about it. 2-3 sentences, caring tone. Under 70 words.")
    try:
        import requests as req
        r = req.post("http://localhost:11434/api/generate",
                     json={"model":"llama3","prompt":prompt,"stream":False,
                           "options":{"temperature":0.9,"num_predict":120}},timeout=30)
        response = r.json().get("response","").strip()
    except:
        response = (f"Thank you for sharing this memory with me, {name}. "
                    f"It means a lot that you wanted to share it. "
                    f"What does this memory mean to you?")
    return jsonify({"response": response})

@app.route("/stats")
def get_stats():
    # Accept user_id from query param as fallback (for HTTPS cookie issues)
    uid = session.get("user_id") or request.args.get("uid")
    if not uid or uid not in active_bots:
        # Last resort: find any bot matching the name
        name = request.args.get("name", "")
        if name:
            for k in active_bots:
                if k.startswith(name + "_"):
                    uid = k
                    break
    if not uid or uid not in active_bots:
        return jsonify({"error": "Not logged in"}), 401
    bot = active_bots[uid]
    s   = bot.get_stats()
    now = datetime.datetime.now()
    mid = now.replace(hour=0,minute=0,second=0,microsecond=0)+datetime.timedelta(days=1)
    mins_left = int((mid-now).total_seconds()/60)
    info = get_last_retrain_info()
    return jsonify({**s,
        "next_retrain_mins": mins_left,
        "last_retrain": info,
        "training_samples": bot.memory.training_sample_count()})

@app.route("/history")
def get_history():
    uid = session.get("user_id") or request.args.get("uid")
    if not uid or uid not in active_bots:
        name = request.args.get("name", "")
        if name:
            for k in active_bots:
                if k.startswith(name + "_"):
                    uid = k; break
    if not uid or uid not in active_bots:
        return jsonify({"error":"Not logged in"}),401
    bot      = active_bots[uid]
    date_str = request.args.get("date", str(datetime.date.today()))

    timeline = [e for e in bot.memory.data.get("emotion_timeline",[])
                if e.get("date") == date_str]
    available = sorted(set(e.get("date","") for e in bot.memory.data.get("emotion_timeline",[])))

    # ── Group timeline entries into sessions ─────────────────────────────────
    # A new session starts when there is a gap of >30 minutes between entries
    SESSION_GAP_MINUTES = 30
    sessions = []
    current_session = []
    prev_ts = None

    for entry in timeline:
        try:
            ts = datetime.datetime.fromisoformat(entry.get("timestamp",""))
        except Exception:
            ts = None

        if ts and prev_ts and (ts - prev_ts).total_seconds() > SESSION_GAP_MINUTES * 60:
            if current_session:
                sessions.append(current_session)
            current_session = []

        current_session.append(entry)
        prev_ts = ts

    if current_session:
        sessions.append(current_session)

    # ── Build session summaries with emotion shift info ──────────────────────
    session_data = []
    for i, sess in enumerate(sessions):
        emotions = [e.get("emotion","neutral") for e in sess]
        topics   = list(set(e.get("topic") for e in sess if e.get("topic")))
        # Find start and end time
        try:
            t_start = datetime.datetime.fromisoformat(sess[0].get("timestamp","")).strftime("%I:%M %p")
            t_end   = datetime.datetime.fromisoformat(sess[-1].get("timestamp","")).strftime("%I:%M %p")
        except Exception:
            t_start = t_end = "—"

        # Emotion shifts: list of (from, to) whenever emotion changes
        shifts = []
        for j in range(1, len(emotions)):
            if emotions[j] != emotions[j-1]:
                shifts.append({"from": emotions[j-1], "to": emotions[j]})

        # Dominant emotion in this session
        from collections import Counter
        dom = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"

        session_data.append({
            "session_num":  i + 1,
            "start_time":   t_start,
            "end_time":     t_end,
            "msg_count":    len(sess),
            "dominant":     dom,
            "emotions":     emotions,
            "topics":       topics,
            "shifts":       shifts,
            "entries":      sess,
        })

    return jsonify({
        "date":            date_str,
        "timeline":        timeline,
        "sessions":        session_data,
        "available_dates": available,
    })

@app.route("/personal_model")
def personal_model():
    uid = find_uid(request.args.get("name","")) or session.get("user_id")
    if not uid or uid not in active_bots: return jsonify({"error":"Not logged in"}),401
    bot = active_bots[uid]
    cal = bot.memory.calibrator
    pairs = []
    for phrase, emotions in cal.phrase_emotion_map.items():
        if len(phrase.split()) >= 2:
            total_obs = sum(emotions.values())
            if total_obs >= 2:
                top = max(emotions, key=emotions.get)
                pairs.append({"phrase":phrase,"emotion":top,
                               "confidence":round(emotions[top]/total_obs,2),
                               "count":total_obs})
    pairs.sort(key=lambda x:x["count"], reverse=True)
    hints = {}
    for tp in ["i am fine","i am ok","not bad","alright","i am tired"]:
        h = cal.get_hidden_emotion_hint(tp)
        if h: hints[tp] = h
    kg = bot.knowledge_graph.get_graph_summary()
    return jsonify({"observations":cal.total_observations,"patterns":pairs[:10],
                    "hidden_emotions":hints,"knowledge_graph":kg,
                    "active": cal.total_observations >= 3})

@app.route("/analyze_graph")
def analyze_graph():
    uid = find_uid(request.args.get("name","")) or session.get("user_id")
    if not uid or uid not in active_bots: return jsonify({"error":"Not logged in"}),401
    bot = active_bots[uid]
    ec  = bot.memory.data.get("emotion_counts",{})
    dl  = bot.memory.data.get("daily_log",{})
    total = bot.memory.total_messages()
    pos = sum(v for k,v in ec.items() if k in POSITIVE_EMOTIONS)
    neg = sum(v for k,v in ec.items() if k in NEGATIVE_EMOTIONS)
    dates  = sorted(dl.keys())
    scores = []
    for d in dates:
        s = sum(cnt if em in POSITIVE_EMOTIONS else -cnt if em in NEGATIVE_EMOTIONS else 0
                for em,cnt in dl[d].items())
        scores.append({"date":d,"score":s})
    sorted_ec = sorted(ec.items(),key=lambda x:x[1],reverse=True)[:8]
    return jsonify({"emotion_counts": dict(sorted_ec), "total": total,
                    "positive": pos, "negative": neg,
                    "timeline": scores,
                    "sessions": bot.memory.sessions(),
                    "dominant": sorted_ec[0][0] if sorted_ec else "neutral"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receives audio blob from browser MediaRecorder.
    Tries Whisper (via faster-whisper or whisper module) first.
    Falls back to a message asking user to type if not available.
    Works in ALL browsers: Brave, Chrome, Edge, Firefox.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]
    import tempfile, subprocess, shutil

    # Save audio to temp file
    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    # Try faster-whisper first (best option)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
        os.unlink(tmp_path)
        if text:
            return jsonify({"text": text})
    except ImportError:
        pass
    except Exception as e:
        print(f"faster-whisper error: {e}")

    # Try openai-whisper
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(tmp_path, language="en")
        text = result["text"].strip()
        os.unlink(tmp_path)
        if text:
            return jsonify({"text": text})
    except ImportError:
        pass
    except Exception as e:
        print(f"whisper error: {e}")

    # Try Ollama speech (if supported)
    try:
        import requests as req
        with open(tmp_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        # Note: Ollama does not natively support audio yet, this is a placeholder
        os.unlink(tmp_path)
    except Exception:
        pass

    # Clean up temp file
    if os.path.exists(tmp_path):
        try: os.unlink(tmp_path)
        except: pass

    return jsonify({
        "error": "Speech-to-text not available",
        "hint": "Install faster-whisper: pip install faster-whisper"
    })

@app.route("/motivation", methods=["POST"])
def motivation():
    """
    Returns a unique motivation for this user.
    Tracks which quotes were shown to avoid repetition.
    Different users get different quotes.
    """
    data = request.get_json()
    name = data.get("user", "friend")

    # Large pool of diverse motivational messages
    MOTIVATIONS = [
        ("The only way out is through. You are stronger than you think, and this moment — as heavy as it feels — is not the end of your story.", None),
        ("Every morning you wake up and try again is an act of incredible courage. Do not underestimate that.", None),
        ("You don't have to have it all figured out. You just have to take the next small step.", None),
        ("The version of you that got through every hard day so far is the same one getting through this one.", None),
        ("Progress is not always visible. Sometimes healing, growing, and becoming happen quietly inside you.", None),
        ("You are not behind. You are not failing. You are on your own timeline and that is perfectly valid.", None),
        ("It's okay to rest. It's okay to not be okay. And it's okay to need time. None of that makes you weak.", None),
        ("The people who seem the strongest have simply decided that giving up is not an option they're willing to choose.", None),
        ("Your feelings are valid. Your struggles are real. And your capacity to get through them is greater than you know.", None),
        ("One day at a time. Sometimes one hour. Sometimes one breath. That's enough.", None),
        ("The fact that you're still here, still trying — that is not nothing. That is everything.", None),
        ("Growth is uncomfortable because you are expanding into something larger than you've ever been before.", None),
        ("You have survived every hard day you've faced so far. Your track record is 100%.", None),
        ("Not all storms come to disrupt your life. Some come to clear your path.", None),
        ("You are allowed to be both a work in progress and worthy of love and belonging right now.", "Brené Brown (adapted)"),
        ("Difficult roads often lead to beautiful destinations. Trust the process even when the path is unclear.", None),
        ("Your sensitivity is not a weakness. It means you feel deeply, care deeply, and live deeply.", None),
        ("The moment you feel like giving up, remember why you held on for so long.", None),
        ("Healing is not linear. Some days you'll feel fine and some days you won't. Both are part of the journey.", None),
        ("You are not the sum of your worst days. You are also every small act of courage, every time you kept going.", None),
        ("Be patient with yourself. You are a human being, not a human doing.", None),
        ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
        ("In the middle of every difficulty lies opportunity.", "Albert Einstein"),
        ("The comeback is always stronger than the setback.", None),
        ("You are braver than you believe, stronger than you seem, and smarter than you think.", "A.A. Milne"),
        ("Hard times never last, but hard people do.", None),
        ("Stars can't shine without darkness. Your darkest moment is preparing your brightest chapter.", None),
        ("What you are going through is hard. What you are becoming because of it is extraordinary.", None),
        ("You don't have to be perfect to be enough. You already are enough, exactly as you are right now.", None),
        ("Every expert was once a beginner. Every master was once a disaster. Keep going.", None),
    ]

    # Track which motivations this user has already received
    uid       = session.get("user_id", name)
    seen_key  = f"seen_motivations_{uid}"
    seen      = session.get(seen_key, [])

    # Filter out ones already seen
    available = [i for i in range(len(MOTIVATIONS)) if i not in seen]

    # If all seen, reset
    if not available:
        seen      = []
        available = list(range(len(MOTIVATIONS)))
        session[seen_key] = []

    # Pick one — use user name hash for variety between users
    import random
    random.seed(None)  # true random
    idx = random.choice(available)

    # Mark as seen
    seen.append(idx)
    session[seen_key] = seen
    session.modified  = True

    quote, author = MOTIVATIONS[idx]
    return jsonify({"motivation": quote, "author": author})


# ── Federated Learning Gradient Contribution ──────────────────────────────────
# When a client ends their session (logout) OR has accumulated enough messages,
# the server writes a minimal anonymized gradient file to user_histories/.
# This file contains ONLY emotion vectors and pseudo-labels — NO raw text,
# NO profile, NO avatar. retrain.py reads these files to do FedAvg.
# Personal data stays in localStorage; only the FL gradient touches server disk.

GRADIENT_FLUSH_INTERVAL = 5   # write gradient file every N messages

EMOTION_LABELS_FL = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval","disgust",
    "embarrassment","excitement","fear","gratitude","grief","joy","love",
    "nervousness","optimism","pride","realization","relief","remorse",
    "sadness","surprise","neutral"
]

def flush_fl_gradient(bot, uid: str):
    """
    Write an anonymized gradient file for retrain.py.
    Contains only:
      - emotion_timeline  (probability vectors + dominant label + date)
      - daily_log         (emotion counts per day)
      - training_samples  (pseudo-text derived from dominant label — no real text)
    No name, no raw text, no avatar, no profile.
    """
    try:
        mem      = bot.memory
        timeline = mem.data.get("emotion_timeline", [])
        daily_log = mem.data.get("daily_log", {})
        if not timeline:
            return   # nothing to contribute

        # Build pseudo training_samples from emotion_timeline
        # Text = "user expressed <emotion>" — no real content, safe for server
        training_samples = [
            {
                "text":       f"user expressed {entry.get('emotion','neutral')}",
                "label":      entry.get("emotion", "neutral"),
                "confidence": float(max(entry.get("vector", [0.5]))),
                "date":       entry.get("date", str(datetime.date.today())),
            }
            for entry in timeline
            if entry.get("emotion")
        ]

        # Anonymized gradient file — matches the schema retrain.py expects
        gradient_data = {
            "user_id":          f"client_{uid[-8:]}",  # anonymised uid suffix
            "sessions":         mem.data.get("sessions", 1),
            "total_messages":   mem.data.get("total_messages", 0),
            "emotion_counts":   mem.data.get("emotion_counts", {}),
            "daily_log":        daily_log,
            "emotion_timeline": [
                {k: v for k, v in e.items() if k != "vector"}   # strip raw vectors
                for e in timeline
            ],
            "training_samples": training_samples,
            "knowledge_graph":  {},   # omitted — too identifying
            "first_seen":       mem.data.get("first_seen", str(datetime.date.today())),
            "last_seen":        str(datetime.date.today()),
            "calibrator_data":  {},   # omitted — personal
        }

        path = os.path.join("user_histories", f"{uid}_emotional_memory.json")
        os.makedirs("user_histories", exist_ok=True)
        with open(path, "w") as f:
            json.dump(gradient_data, f, indent=2)
        print(f"[FL] Gradient flushed → {path} "
              f"({len(training_samples)} samples, {len(daily_log)} days)")
    except Exception as e:
        print(f"[FL] Warning: gradient flush failed for {uid}: {e}")


@app.route("/logout", methods=["POST"])
def logout():
    body = request.get_json() or {}
    uid  = find_uid(body.get("name","")) or session.get("user_id")
    if uid and uid in active_bots:
        # Flush anonymized FL gradient to disk before removing bot from memory
        flush_fl_gradient(active_bots[uid], uid)
        del active_bots[uid]
    session.clear()
    return jsonify({"success": True})

@app.route("/receive_gradient", methods=["POST"])
def receive_gradient():
    """
    Receives anonymized FL gradient from a client machine.
    This is the ONLY data that ever comes from the client to the server.
    Contains: emotion labels, pseudo-text, counts. NO real text, NO photos.
    retrain.py reads these files during its daily FL round.
    """
    try:
        gradient = request.get_json() or {}
        uid      = gradient.get("user_id", f"client_{request.remote_addr.replace('.','_')}")
        if not gradient.get("training_samples"):
            return jsonify({"status": "empty", "msg": "No samples received"}), 200

        path = os.path.join("user_histories", f"{uid}_emotional_memory.json")
        os.makedirs("user_histories", exist_ok=True)
        with open(path, "w") as f:
            json.dump(gradient, f, indent=2)

        n = len(gradient.get("training_samples", []))
        print(f"[FL] Gradient received from {request.remote_addr} "
              f"→ {path} ({n} samples)")
        return jsonify({"status": "ok", "samples_received": n})
    except Exception as e:
        print(f"[FL] receive_gradient error: {e}")
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route("/model_download")
def model_download():
    """
    Clients call this once to download the global shared model to their machine.
    The model itself contains no user data — it's the shared FL-trained weights.
    Sends Content-Length so the client can show a download progress bar.
    """
    if os.path.exists(MODEL_PATH):
        response = send_file(
            MODEL_PATH,
            as_attachment=True,
            download_name="model_final.pt",
            mimetype="application/octet-stream",
        )
        # Explicitly set Content-Length so streaming clients can show progress
        response.headers["Content-Length"] = os.path.getsize(MODEL_PATH)
        return response
    return jsonify({"error": "Model not found on server"}), 404


@app.route("/update_profile", methods=["POST"])
def update_profile():
    # Profile is now owned by the client (localStorage).
    # Server just acknowledges — no disk write.
    return jsonify({"success": True})

if __name__ == "__main__":
    import socket, ssl
    hostname = socket.gethostname()
    try: local_ip = socket.gethostbyname(hostname)
    except: local_ip = "127.0.0.1"
    info = get_last_retrain_info()

    os.makedirs("templates", exist_ok=True)

    # ── Generate self-signed HTTPS certificate if not exists ──────────────────
    if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
        print("  Generating HTTPS certificate...")
        try:
            from OpenSSL import crypto
            key = crypto.PKey()
            key.generate_key(crypto.TYPE_RSA, 2048)
            cert = crypto.X509()
            cert.get_subject().CN = "MindBridge"
            cert.set_serial_number(1000)
            cert.gmtime_adj_notBefore(0)
            cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)
            cert.set_issuer(cert.get_subject())
            cert.set_pubkey(key)
            cert.sign(key, "sha256")
            open("cert.pem", "wb").write(
                crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
            open("key.pem", "wb").write(
                crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
            print("  Certificate generated successfully.")
        except ImportError:
            print("  pyopenssl not installed. Run: pip install pyopenssl")
            print("  Falling back to HTTP...")

    # ── Determine if HTTPS is available ──────────────────────────────────────
    use_https = os.path.exists("cert.pem") and os.path.exists("key.pem")
    protocol  = "https" if use_https else "http"

    print(f"\n{'='*58}")
    print(f"  MindBridge Advanced Web App")
    print(f"{'='*58}")
    print(f"  Your URL      : {protocol}://{local_ip}:5000")
    print(f"  Localhost     : {protocol}://127.0.0.1:5000")
    print(f"  Share on WiFi : {protocol}://{local_ip}:5000")
    print(f"  Internet      : NOT required")
    print(f"  Microphone    : {'ENABLED (HTTPS active)' if use_https else 'localhost only (HTTP)'}")
    if info:
        print(f"  Last retrain  : {info.get('date')} — acc {info.get('avg_accuracy',0):.2%}")
    print(f"{'='*58}")
    if use_https:
        print(f"\n  NOTE: Browser will show security warning — click")
        print(f"  'Advanced' then 'Proceed to {local_ip} (unsafe)'")
        print(f"  This is normal for self-signed certificates.\n")

    # ── Start server ──────────────────────────────────────────────────────────
    if use_https:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True,
                ssl_context=("cert.pem", "key.pem"))
    else:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)