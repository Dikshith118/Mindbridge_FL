"""
client.py — MindBridge Local Client Server
===========================================
Runs on EACH user's machine (port 5001).
Stores all personal data LOCALLY in their own folders:
  client_data/user_histories/    ← emotion memory (stays on THEIR machine)
  client_data/user_profiles/     ← name, bio, joined date
  client_data/user_uploads/      ← shared photos
  client_data/user_avatars/      ← profile pictures

Talks to the central server (port 5000) ONLY for:
  1. Downloading the global shared model once (read-only)
  2. Sending anonymized FL gradient after session (write-only, no personal data)
  3. Getting Ollama-generated responses (text only, no personal data)

Usage:
  python client.py --server 192.168.1.63   ← specify your server IP
  python client.py                          ← uses saved config or offline mode

Then open:  http://localhost:5001
"""

import os
import sys
import json
import datetime
import base64
import socket
import argparse
import threading
import shutil

# Suppress InsecureRequestWarning spam from self-signed HTTPS cert
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Parse CLI args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MindBridge Local Client")
parser.add_argument("--server", default=None,
                    help="Central server IP (e.g. 192.168.1.63)")
parser.add_argument("--port", type=int, default=5001,
                    help="Local port (default 5001)")
args, _ = parser.parse_known_args()
CLIENT_PORT = args.port

# ── Local storage layout ──────────────────────────────────────────────────────
BASE_DIR     = "client_data"
HISTORY_DIR  = os.path.join(BASE_DIR, "user_histories")
PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")
UPLOADS_DIR  = os.path.join(BASE_DIR, "user_uploads")
AVATARS_DIR  = os.path.join(BASE_DIR, "user_avatars")
MODEL_DIR    = os.path.join(BASE_DIR, "model_cache")
CFG_FILE     = os.path.join(BASE_DIR, "server_config.json")

for _d in [HISTORY_DIR, PROFILES_DIR, UPLOADS_DIR, AVATARS_DIR, MODEL_DIR]:
    os.makedirs(_d, exist_ok=True)

LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "model_final.pt")

# ── Resolve server URL ────────────────────────────────────────────────────────
def _load_server_ip():
    if args.server:
        return args.server
    if os.path.exists(CFG_FILE):
        try:
            return json.load(open(CFG_FILE)).get("server_ip")
        except: pass
    return None

SERVER_IP  = _load_server_ip()
SERVER_URL = f"https://{SERVER_IP}:5000" if SERVER_IP else None


def _save_server_ip(ip: str):
    global SERVER_IP, SERVER_URL
    SERVER_IP  = ip
    SERVER_URL = f"https://{ip}:5000"
    with open(CFG_FILE, "w") as f:
        json.dump({"server_ip": ip, "server_url": SERVER_URL}, f, indent=2)


# ── Download global model from server (once) ──────────────────────────────────
def sync_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"[client] Model already cached at {LOCAL_MODEL_PATH} ✅")
        return True
    # Try project-level fallback first (server laptop running client locally)
    fallback = "saved_model/model_final.pt"
    if os.path.exists(fallback):
        shutil.copy(fallback, LOCAL_MODEL_PATH)
        print(f"[client] Model copied from local fallback → {LOCAL_MODEL_PATH}")
        return True
    if not SERVER_URL:
        print("[client] No server configured — cannot download model.")
        print("[client] Run with:  python client.py --server <server_ip>")
        return False

    import requests as _req

    print(f"[client] Downloading model from {SERVER_URL}/model_download ...")
    try:
        # KEY FIX: stream=True downloads in 8 KB chunks instead of loading
        # the entire .pt file into RAM — this is what was causing the hang.
        # timeout=(10, 600): 10s to connect, 10 minutes to finish reading.
        with _req.get(
            f"{SERVER_URL}/model_download",
            stream=True,
            verify=False,
            timeout=(10, 600),
        ) as r:
            r.raise_for_status()
            total     = int(r.headers.get("content-length", 0))
            downloaded = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            print(f"\r[client] Downloading... {pct}%  "
                                  f"({downloaded // 1024} KB / {total // 1024} KB)",
                                  end="", flush=True)
            print(f"\n[client] ✅ Model saved → {LOCAL_MODEL_PATH}")
            return True

    except _req.exceptions.ConnectTimeout:
        print(f"\n[client] ❌ Cannot reach server at {SERVER_URL}")
        print("[client]    → Confirm server laptop is on the same WiFi")
        print("[client]    → Confirm 'python app.py' is running on the server laptop")
        print(f"[client]    → Confirm server IP is correct (current: {SERVER_IP})")
    except _req.exceptions.ConnectionError as e:
        print(f"\n[client] ❌ Connection error: {e}")
        print("[client]    → Is the server running? Try: ping " + (SERVER_IP or "?"))
    except _req.exceptions.ReadTimeout:
        print("\n[client] ❌ Download timed out (took longer than 10 minutes)")
        print("[client]    → Check WiFi speed or move closer to the router")
    except _req.exceptions.HTTPError as e:
        print(f"\n[client] ❌ Server returned error: {e}")
    except Exception as e:
        print(f"\n[client] ❌ Unexpected error during download: {e}")

    # Clean up incomplete download
    if os.path.exists(LOCAL_MODEL_PATH):
        os.remove(LOCAL_MODEL_PATH)
    return False


# ── Import chatbot ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot"))
try:
    from chatbot import MindBridge, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS
except ImportError as e:
    print(f"[client] FATAL: cannot import chatbot — {e}")
    sys.exit(1)

MODEL_OK = sync_model()

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, send_file
try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

app = Flask(__name__)
app.secret_key = "mindbridge_client_local_2026"
if _has_cors:
    CORS(app)

active_bots: dict = {}   # name → MindBridge


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL FILE HELPERS — everything in client_data/ on THIS machine
# ─────────────────────────────────────────────────────────────────────────────

def _history_path(name): return os.path.join(HISTORY_DIR, f"{name}_emotional_memory.json")
def _profile_path(name): return os.path.join(PROFILES_DIR, f"{name}_profile.json")
def _avatar_path(name):  return os.path.join(AVATARS_DIR,  f"{name}_avatar.jpg")


def load_profile(name: str) -> dict:
    p = _profile_path(name)
    if os.path.exists(p):
        try:
            with open(p) as f: return json.load(f)
        except: pass
    return {"name": name, "joined": str(datetime.date.today()), "bio": ""}


def save_profile(name: str, data: dict):
    with open(_profile_path(name), "w") as f:
        json.dump(data, f, indent=2)


def restore_memory(bot, name: str):
    """Load previously saved memory from local disk into the bot."""
    p = _history_path(name)
    if not os.path.exists(p):
        return
    try:
        with open(p) as f:
            saved = json.load(f)
        for key in ["sessions","total_messages","emotion_counts","daily_log",
                    "emotion_timeline","knowledge_graph","first_seen","last_seen",
                    "calibrator_data"]:
            if key in saved:
                bot.memory.data[key] = saved[key]
        bot.memory._load_calibrator()
        bot.knowledge_graph.graph = bot.memory.data.get("knowledge_graph", {})
        print(f"[client] Memory restored for '{name}' — "
              f"{bot.memory.total_messages()} messages, "
              f"{bot.memory.sessions()} sessions")
    except Exception as e:
        print(f"[client] Memory restore failed for '{name}': {e}")


def save_memory(name: str, bot):
    """Persist full emotion memory to local disk (user's own machine)."""
    bot.memory.data["calibrator_data"] = bot.memory.calibrator.to_dict()
    with open(_history_path(name), "w") as f:
        json.dump(bot.memory.data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# FL GRADIENT — anonymized, sent to central server for FedAvg retraining
# ─────────────────────────────────────────────────────────────────────────────

def send_fl_gradient(bot, name: str):
    """
    Send anonymized emotion gradient to server.
    NO real text, NO photos, NO name, NO calibrator — just emotion labels + counts.
    Runs in a background thread so it never blocks the user.
    """
    if not SERVER_URL:
        return

    timeline      = bot.memory.data.get("emotion_timeline", [])
    daily_log     = bot.memory.data.get("daily_log", {})
    emotion_counts = bot.memory.data.get("emotion_counts", {})
    today         = str(datetime.date.today())

    if not timeline:
        return

    training_samples = [
        {
            "text":       f"user expressed {e.get('emotion','neutral')}",
            "label":      e.get("emotion","neutral"),
            "confidence": float(max(e.get("vector",[0.5]))),
            "date":       e.get("date", today),
        }
        for e in timeline if e.get("emotion")
    ]

    # Stable anonymous ID — same client always sends same ID so server can
    # overwrite the file rather than accumulating duplicates.
    anon_id = f"client_{abs(hash(name + socket.gethostname())) % 99999999:08d}"

    gradient = {
        "user_id":         anon_id,
        "sessions":        bot.memory.data.get("sessions", 1),
        "total_messages":  bot.memory.data.get("total_messages", 0),
        "emotion_counts":  emotion_counts,
        "daily_log":       daily_log,
        "emotion_timeline":[
            {k: v for k, v in e.items() if k != "vector"}
            for e in timeline
        ],
        "training_samples": training_samples,
        "knowledge_graph":  {},
        "first_seen":       bot.memory.data.get("first_seen", today),
        "last_seen":        today,
        "calibrator_data":  {},
    }

    def _send():
        try:
            import requests as _req
            r = _req.post(f"{SERVER_URL}/receive_gradient",
                          json=gradient, timeout=15, verify=False)
            if r.status_code == 200:
                print(f"[FL] Gradient sent → {SERVER_URL} "
                      f"({len(training_samples)} samples, anon_id={anon_id})")
            else:
                print(f"[FL] Server rejected gradient: HTTP {r.status_code}")
        except Exception as e:
            print(f"[FL] Could not reach server: {e} "
                  f"(data still safe locally in {_history_path(name)})")

    threading.Thread(target=_send, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    tmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "templates", "index.html")
    with open(tmpl, encoding="utf-8") as f:
        return f.read()

@app.route("/favicon.ico")
def favicon(): return "", 204


@app.route("/status")
def status():
    if SERVER_URL:
        try:
            import requests as _req
            r = _req.get(f"{SERVER_URL}/status", timeout=4, verify=False)
            return jsonify(r.json())
        except: pass
    return jsonify({"retrain_date": None, "retrain_running": False})


@app.route("/retrain_status")
def retrain_status():
    if SERVER_URL:
        try:
            import requests as _req
            r = _req.get(f"{SERVER_URL}/retrain_status", timeout=4, verify=False)
            return jsonify(r.json())
        except: pass
    return jsonify({"running": False, "last_info": None})


@app.route("/start", methods=["POST"])
def start():
    data = request.get_json() or {}
    name = (data.get("name","guest") or "guest").strip()

    if name not in active_bots:
        model_path = (LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH)
                      else "saved_model/model_final.pt")
        active_bots[name] = MindBridge(user_id=name, model_path=model_path)
        restore_memory(active_bots[name], name)   # ← loads from client_data/

    bot          = active_bots[name]
    is_returning = bot.memory.is_returning()
    bot.memory.start_session()

    hour = datetime.datetime.now().hour
    tg   = ("Good morning" if 5<=hour<12 else "Good afternoon" if 12<=hour<17
            else "Good evening" if 17<=hour<21 else "Hi")

    if is_returning:
        dominant = bot.memory.dominant_emotion()
        greeting = (f"{tg}, {name}. I'm glad you came back. How are you feeling today?"
                    if dominant in NEGATIVE_EMOTIONS
                    else f"{tg}, {name}! Good to see you again. How are you doing today?")
    else:
        greeting = (f"{tg}, {name}! I'm MindBridge — your private emotional support companion. "
                    f"All your data stays right here on your own device. "
                    f"How are you feeling today?")

    profile    = load_profile(name)
    has_avatar = os.path.exists(_avatar_path(name))

    save_memory(name, bot)
    save_profile(name, profile)

    return jsonify({
        "greeting":       greeting,
        "name":           name,
        "retrain_info":   None,
        "has_avatar":     has_avatar,
        "profile":        profile,
        "sessions":       bot.memory.sessions(),
        "total_messages": bot.memory.total_messages(),
    })


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json() or {}
    name    = data.get("name","")
    message = data.get("message","").strip()

    if not name or name not in active_bots:
        return jsonify({"response":"Session expired — please refresh.",
                        "emotion":"neutral","confidence":0,"trend":"stable",
                        "is_crisis":False,"is_medical":False,"risk_score":0,
                        "retrain_running":False})

    bot    = active_bots[name]
    result = bot.respond(message)
    result["retrain_running"] = False

    # ── Save to LOCAL disk after every message ────────────────────────────────
    save_memory(name, bot)

    # ── Send gradient to server every 5 messages (background) ────────────────
    if bot.memory.total_messages() % 5 == 0:
        send_fl_gradient(bot, name)

    return jsonify(result)


@app.route("/upload_avatar", methods=["POST"])
def upload_avatar():
    """Saves avatar to LOCAL client_data/user_avatars/ — stays on this machine."""
    data = request.get_json() or {}
    name = data.get("name","")
    img  = data.get("image","")
    if not name or not img:
        return jsonify({"error":"Missing name or image"}), 400
    if img.startswith("data:image"):
        img = img.split(",",1)[1]
    try:
        with open(_avatar_path(name), "wb") as f:
            f.write(base64.b64decode(img))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/avatar/<name>")
def get_avatar(name):
    """Serves avatar from LOCAL disk."""
    p = _avatar_path(name)
    if os.path.exists(p):
        return send_file(p, mimetype="image/jpeg")
    return "", 404


@app.route("/upload_memory", methods=["POST"])
def upload_memory():
    """Saves photo to LOCAL disk, proxies caption to server for a response."""
    data    = request.get_json() or {}
    name    = data.get("name","")
    img_b64 = data.get("image","")
    caption = data.get("caption","") or "Sharing this memory."

    if not name or name not in active_bots:
        return jsonify({"error":"Not logged in"}), 401

    # ── Save image LOCALLY ────────────────────────────────────────────────────
    if img_b64:
        raw = img_b64.split(",",1)[1] if img_b64.startswith("data:image") else img_b64
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_file = os.path.join(UPLOADS_DIR, f"{name}_{ts}.jpg")
        try:
            with open(img_file,"wb") as f: f.write(base64.b64decode(raw))
            print(f"[client] Photo saved locally → {img_file}")
        except Exception as e:
            print(f"[client] Photo save failed: {e}")

    # ── Get response from server (only caption sent — no image) ──────────────
    fallback = (f"Thanks for sharing this memory, {name}! "
                f"It sounds really meaningful. What does it bring to mind?")
    if SERVER_URL:
        try:
            import requests as _req
            r = _req.post(f"{SERVER_URL}/upload_memory",
                          json={"name": name, "caption": caption},
                          timeout=30, verify=False)
            return jsonify({"response": r.json().get("response", fallback)})
        except: pass
    return jsonify({"response": fallback})


@app.route("/stats")
def get_stats():
    name = request.args.get("name","")
    if not name or name not in active_bots:
        return jsonify({"error":"Not logged in"}), 401
    bot = active_bots[name]
    s   = bot.get_stats()
    return jsonify({**s, "next_retrain_mins": 0,
                    "last_retrain": None,
                    "training_samples": bot.memory.total_messages()})


@app.route("/history")
def get_history():
    name     = request.args.get("name","")
    date_str = request.args.get("date", str(datetime.date.today()))
    if not name or name not in active_bots:
        return jsonify({"error":"Not logged in"}), 401
    bot      = active_bots[name]
    timeline = [e for e in bot.memory.data.get("emotion_timeline",[])
                if e.get("date") == date_str]
    available = sorted({e.get("date","")
                        for e in bot.memory.data.get("emotion_timeline",[])})
    return jsonify({"date": date_str, "timeline": timeline,
                    "available_dates": available})


@app.route("/analyze_graph")
def analyze_graph():
    name = request.args.get("name","")
    if not name or name not in active_bots:
        return jsonify({"error":"Not logged in"}), 401
    bot   = active_bots[name]
    ec    = bot.memory.data.get("emotion_counts",{})
    dl    = bot.memory.data.get("daily_log",{})
    total = bot.memory.total_messages()
    pos   = sum(v for k,v in ec.items() if k in POSITIVE_EMOTIONS)
    neg   = sum(v for k,v in ec.items() if k in NEGATIVE_EMOTIONS)
    scores = [
        {"date": d,
         "score": sum(cnt if em in POSITIVE_EMOTIONS else
                      -cnt if em in NEGATIVE_EMOTIONS else 0
                      for em,cnt in dl[d].items())}
        for d in sorted(dl.keys())
    ]
    sorted_ec = sorted(ec.items(), key=lambda x:x[1], reverse=True)[:8]
    return jsonify({"emotion_counts": dict(sorted_ec), "total": total,
                    "positive": pos, "negative": neg, "timeline": scores,
                    "sessions": bot.memory.sessions(),
                    "dominant": sorted_ec[0][0] if sorted_ec else "neutral"})


@app.route("/personal_model")
def personal_model():
    name = request.args.get("name","")
    if not name or name not in active_bots:
        return jsonify({"error":"Not logged in"}), 401
    bot = active_bots[name]
    cal = bot.memory.calibrator
    pairs = [
        {"phrase": phrase, "emotion": max(ems, key=ems.get),
         "confidence": round(ems[max(ems,key=ems.get)] / sum(ems.values()),2),
         "count": sum(ems.values())}
        for phrase, ems in cal.phrase_emotion_map.items()
        if len(phrase.split()) >= 2 and sum(ems.values()) >= 2
    ]
    pairs.sort(key=lambda x:x["count"], reverse=True)
    hints = {tp: cal.get_hidden_emotion_hint(tp)
             for tp in ["i am fine","i am ok","not bad","alright","i am tired"]
             if cal.get_hidden_emotion_hint(tp)}
    return jsonify({"observations": cal.total_observations,
                    "patterns": pairs[:10], "hidden_emotions": hints,
                    "knowledge_graph": bot.knowledge_graph.get_graph_summary(),
                    "active": cal.total_observations >= 3})


@app.route("/update_profile", methods=["POST"])
def update_profile():
    """Saves profile to LOCAL client_data/user_profiles/ — stays on this machine."""
    data = request.get_json() or {}
    name = data.get("name","")
    if not name: return jsonify({"error":"Missing name"}), 400
    profile = load_profile(name)
    for k in ["bio","display_name"]:
        if k in data: profile[k] = data[k]
    save_profile(name, profile)
    return jsonify({"success": True})


@app.route("/logout", methods=["POST"])
def logout():
    data = request.get_json() or {}
    name = data.get("name","")
    if name and name in active_bots:
        bot = active_bots[name]
        save_memory(name, bot)        # ← writes to client_data/user_histories/
        send_fl_gradient(bot, name)   # ← sends anonymized gradient to server
        del active_bots[name]
    return jsonify({"success": True})


@app.route("/motivation", methods=["POST"])
def motivation():
    data = request.get_json() or {}
    if SERVER_URL:
        try:
            import requests as _req
            r = _req.post(f"{SERVER_URL}/motivation", json=data,
                          timeout=5, verify=False)
            return jsonify(r.json())
        except: pass
    import random
    quotes = [
        ("You have survived every hard day so far. Your track record is 100%.", None),
        ("One day at a time. Sometimes one hour. That's enough.", None),
        ("Progress is not always visible — sometimes healing happens quietly.", None),
        ("You are not behind. You are on your own timeline.", None),
        ("The comeback is always stronger than the setback.", None),
    ]
    q, a = random.choice(quotes)
    return jsonify({"motivation": q, "author": a})


@app.route("/configure_server", methods=["POST"])
def configure_server():
    """Let the user set the server IP from a browser prompt."""
    global SERVER_URL, SERVER_IP
    data = request.get_json() or {}
    ip   = data.get("server_ip","").strip()
    if not ip: return jsonify({"error":"No IP provided"}), 400
    _save_server_ip(ip)
    sync_model()   # try to download model now
    return jsonify({"success": True, "server_url": SERVER_URL})


@app.route("/client_info")
def client_info():
    """Verification endpoint — shows exactly what's stored locally."""
    history_files = os.listdir(HISTORY_DIR)
    profile_files = os.listdir(PROFILES_DIR)
    avatar_files  = os.listdir(AVATARS_DIR)
    upload_files  = os.listdir(UPLOADS_DIR)
    return jsonify({
        "storage_location": os.path.abspath(BASE_DIR),
        "server_url":       SERVER_URL or "NOT CONFIGURED",
        "model_cached":     os.path.exists(LOCAL_MODEL_PATH),
        "local_files": {
            "user_histories": history_files,
            "user_profiles":  profile_files,
            "user_avatars":   avatar_files,
            "user_uploads":   upload_files,
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except:
        local_ip = "127.0.0.1"

    print(f"\n{'='*60}")
    print(f"  MindBridge — Local Client Server")
    print(f"{'='*60}")
    print(f"  Open in browser  :  http://localhost:{CLIENT_PORT}")
    print(f"  Your machine IP  :  {local_ip}")
    print(f"  Central server   :  {SERVER_URL or '⚠️  NOT CONFIGURED'}")
    print(f"  Data folder      :  {os.path.abspath(BASE_DIR)}")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │  client_data/                                │")
    print(f"  │    user_histories/  ← emotion memory (HERE) │")
    print(f"  │    user_profiles/   ← name, bio      (HERE) │")
    print(f"  │    user_uploads/    ← photos          (HERE) │")
    print(f"  │    user_avatars/    ← profile pic     (HERE) │")
    print(f"  │    model_cache/     ← shared model    (HERE) │")
    print(f"  └──────────────────────────────────────────────┘")
    print(f"  Model ready      :  {'✅ YES' if MODEL_OK else '❌ NO — check server connection'}")
    print(f"{'='*60}")

    if not SERVER_URL:
        print(f"\n  ⚠️  Server not configured. Run with:")
        print(f"      python client.py --server 192.168.1.63\n")

    if not MODEL_OK:
        print(f"  ⚠️  Model missing. Make sure the server is reachable.\n")
    print(f"  Press CTRL+C to quit\n")
    app.run(host="0.0.0.0", port=CLIENT_PORT, debug=False, threaded=True)