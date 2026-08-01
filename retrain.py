"""
retrain.py
==========
MindBridge — Daily Federated Learning Retraining Script

Run this independently from chatbot.py.
Schedule via Windows Task Scheduler to run at midnight automatically.
Chatbot does NOT need to be open for this to work.

Usage:
  python retrain.py           <- run retraining
  python retrain.py check     <- check if retraining happened today
"""

import os
import sys
import json
import glob
import datetime
import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel

# ── Config — must match chatbot.py ────────────────────────────────────────────
MODEL_PATH     = "saved_model/model_final.pt"
BACKUP_PATH    = "saved_model/model_backup_{}.pt"
LOG_PATH       = "saved_model/retrain_log.json"
HISTORY_DIR    = "user_histories"
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LEN        = 32
NUM_LABELS     = 28
MIN_SAMPLES    = 1       # even 1 message qualifies — no minimum limit
LEARNING_RATE  = 2e-4
LOCAL_EPOCHS   = 2
BATCH_SIZE     = 8

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]
LABEL_MAP = {e: i for i, e in enumerate(EMOTION_LABELS)}


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples   = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item  = self.samples[idx]
        label = LABEL_MAP.get(item.get("label", "neutral"), 27)
        enc   = self.tokenizer(
            item.get("text", ""), max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── Local Training ────────────────────────────────────────────────────────────
def local_train(samples, tokenizer):
    model = EmotionClassifier()
    try:
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        )
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        return None, None, 0.0

    loader    = DataLoader(EmotionDataset(samples, tokenizer),
                           batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    model.train()

    for _ in range(LOCAL_EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            out  = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(out, batch["label"])
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == batch["label"]).sum().item()
            total   += batch["label"].size(0)

    acc = correct / total if total > 0 else 0.0
    return (model.classifier.weight.data.cpu().numpy(),
            model.classifier.bias.data.cpu().numpy(), acc)


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg(weights, biases, counts):
    total = sum(counts)
    avg_w = sum(w * (n / total) for w, n in zip(weights, counts))
    avg_b = sum(b * (n / total) for b, n in zip(biases,  counts))
    return avg_w, avg_b


# ── Log helpers ───────────────────────────────────────────────────────────────
LOCK_FILE = "saved_model/retrain_lock.json"   # chatbot watches this file
LOG_FILE  = "saved_model/retrain_log.json"    # LOG_PATH alias for clarity
LOG_PATH  = LOG_FILE


def load_log():
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"rounds": []}


def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


# ── Main retrain ──────────────────────────────────────────────────────────────
def retrain():
    start = datetime.datetime.now()
    today = str(datetime.date.today())

    # 24-hour window: from yesterday midnight to tonight midnight
    # This means: any user who chatted in the last 24 hours contributes
    yesterday = str(datetime.date.today() - datetime.timedelta(days=1))
    window_dates = {today, yesterday}   # accept both today AND yesterday

    print("\n" + "=" * 58)
    print(f"  MindBridge — Daily FL Retraining")
    print(f"  Date   : {today}")
    print(f"  Window : {yesterday} 00:00 → {today} 23:59 (last 24 hours)")
    print(f"  Started: {start.strftime('%H:%M:%S')}")
    print("=" * 58)

    # ── Write lock file so chatbot knows retrain is running ───────────────────
    os.makedirs("saved_model", exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({"started": start.isoformat(), "date": today}, f)
    print(f"  Lock file created → {LOCK_FILE}")

    # Check if already ran today
    log = load_log()
    if any(r.get("date") == today and r.get("result") == "success"
           for r in log.get("rounds", [])):
        print(f"\n  Already completed today. Model is up to date.")
        print(f"  Run 'python retrain.py check' to see history.")
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        print("=" * 58)
        return

    # Scan user history files
    files = glob.glob(os.path.join(HISTORY_DIR, "*_emotional_memory.json"))
    if not files:
        print(f"\n  No user history files found. Users need to chat first.")
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        print("=" * 58)
        return

    print(f"\n  Scanning {len(files)} user file(s)...\n")
    clients = []

    for filepath in sorted(files):
        try:
            with open(filepath) as f:
                data = json.load(f)
            uid       = data.get("user_id", os.path.basename(filepath))
            daily_log = data.get("daily_log", {})

            # Check if user chatted in the last 24 hours (today OR yesterday)
            active_dates = [d for d in window_dates if d in daily_log]
            if not active_dates:
                print(f"  ⚪ {uid:<20} no activity in last 24 hours — skipped")
                continue

            # Collect ALL training samples from the last 24 hours
            all_samples  = data.get("training_samples", [])
            window_samples = [s for s in all_samples
                              if s.get("date") in window_dates]

            # If no training_samples collected (record_training_sample not called)
            # fall back to using emotion_timeline as pseudo-labels
            if len(window_samples) == 0:
                timeline = data.get("emotion_timeline", [])
                window_samples = [
                    {"text": f"user expressed {e.get('emotion','neutral')}",
                     "label": e.get("emotion", "neutral"),
                     "confidence": 0.65,
                     "date": e.get("date", today)}
                    for e in timeline
                    if e.get("date") in window_dates and e.get("emotion")
                ]
                if window_samples:
                    print(f"  ℹ️  {uid:<20} using emotion timeline as fallback "
                          f"({len(window_samples)} entries)")

            # Sanitise: ensure every sample has a non-empty text string
            # (gradient files use pseudo-text like "user expressed joy")
            sanitised = []
            for s in window_samples:
                text = (s.get("text") or "").strip()
                if not text:
                    text = f"user expressed {s.get('label','neutral')}"
                sanitised.append({**s, "text": text})
            window_samples = sanitised

            # ANY amount of activity qualifies — no minimum
            if len(window_samples) == 0:
                print(f"  ⚠️  {uid:<20} active but no samples collected — skipped")
                continue

            clients.append({"user_id": uid, "samples": window_samples,
                            "count": len(window_samples)})
            print(f"  ✅ {uid:<20} {len(window_samples)} samples — participating")

        except Exception as e:
            print(f"  ❌ Error reading {filepath}: {e}")

    print()

    if not clients:
        print("  No eligible clients today. Model unchanged.")
        log["rounds"].append({
            "date": today, "time": start.strftime("%H:%M:%S"),
            "result": "skipped", "reason": "no eligible clients",
            "clients": 0, "avg_accuracy": None,
        })
        save_log(log)
        # Remove lock file — retrain finished (skipped)
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        print("=" * 58)
        return

    print(f"  {len(clients)} client(s) participating in FL round")
    print(f"{'─'*58}")

    # Backup current model
    backup = BACKUP_PATH.format(today)
    try:
        shutil.copy(MODEL_PATH, backup)
        print(f"  Backup → {backup}")
    except Exception as e:
        print(f"  Warning — backup failed: {e}")

    # Local training
    print(f"\n  Loading tokenizer...")
    tok = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    ws, bs, ns, accs = [], [], [], []

    for i, client in enumerate(clients):
        print(f"\n  [{i+1}/{len(clients)}] {client['user_id']}")
        print(f"        Samples : {client['count']}")
        w, b, acc = local_train(client["samples"], tok)
        if w is not None:
            ws.append(w); bs.append(b)
            ns.append(client["count"]); accs.append(acc)
            print(f"        Accuracy: {acc:.2%}  ✓")
        else:
            print(f"        FAILED — skipped")

    if not ws:
        print("\n  No successful updates. Model unchanged.")
        print("=" * 58)
        return

    # FedAvg + update model
    print(f"\n{'─'*58}")
    print(f"  Running FedAvg across {len(ws)} client(s)...")
    avg_w, avg_b = fedavg(ws, bs, ns)

    gm = EmotionClassifier()
    gm.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
    with torch.no_grad():
        gm.classifier.weight.copy_(torch.tensor(avg_w))
        gm.classifier.bias.copy_(torch.tensor(avg_b))
    torch.save(gm.state_dict(), MODEL_PATH)

    # ── Remove lock file — retrain is complete, chat can unfreeze ─────────────
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    print(f"  Lock file removed → chatbot will unfreeze automatically")

    # Save log + summary
    end     = datetime.datetime.now()
    elapsed = (end - start).seconds
    avg_acc = sum(accs) / len(accs) if accs else 0

    log["rounds"].append({
        "date":          today,
        "time":          start.strftime("%H:%M:%S"),
        "end_time":      end.strftime("%H:%M:%S"),
        "result":        "success",
        "clients":       len(ws),
        "total_samples": sum(ns),
        "avg_accuracy":  round(avg_acc, 4),
        "elapsed_sec":   elapsed,
        "backup":        backup,
    })
    save_log(log)

    print(f"\n{'='*58}")
    print(f"  RETRAINING COMPLETE")
    print(f"{'='*58}")
    print(f"  Clients           : {len(ws)}")
    print(f"  Total samples     : {sum(ns)}")
    print(f"  Avg accuracy      : {avg_acc:.2%}")
    print(f"  Time taken        : {elapsed} seconds")
    print(f"  Finished at       : {end.strftime('%H:%M:%S')}")
    print(f"  Model updated     : {MODEL_PATH}")
    print(f"  Backup saved      : {backup}")
    print(f"  Log               : {LOG_PATH}")
    print(f"\n  All users get the improved model next session.")
    print(f"  Check anytime: python retrain.py check")
    print(f"{'='*58}\n")


# ── Check status ──────────────────────────────────────────────────────────────
def check_status():
    log   = load_log()
    today = str(datetime.date.today())
    rounds = log.get("rounds", [])

    print("\n" + "=" * 58)
    print("  MindBridge — Retraining History")
    print("=" * 58)

    if not rounds:
        print("\n  No retraining rounds yet.")
        print("  Run: python retrain.py")
        print("=" * 58)
        return

    today_r = [r for r in rounds if r.get("date") == today]
    if today_r and today_r[-1].get("result") == "success":
        r = today_r[-1]
        print(f"\n  TODAY ({today}):  RETRAINING DONE")
        print(f"  Time        : {r.get('time')} to {r.get('end_time')}")
        print(f"  Clients     : {r.get('clients')}")
        print(f"  Samples     : {r.get('total_samples')}")
        print(f"  Accuracy    : {r.get('avg_accuracy', 0):.2%}")
        print(f"  Duration    : {r.get('elapsed_sec')}s")
        print(f"  Model is UP TO DATE")
    else:
        print(f"\n  TODAY ({today}):  NOT YET RUN")
        print(f"  Run: python retrain.py")

    print(f"\n  {'─'*54}")
    print(f"  History (last {min(10, len(rounds))} rounds):")
    for r in reversed(rounds[-10:]):
        icon = "OK" if r.get("result") == "success" else "--"
        acc  = r.get("avg_accuracy", 0) or 0
        print(f"  [{icon}] {r.get('date')}  {r.get('time')}  "
              f"clients={r.get('clients', 0)}  "
              f"acc={acc:.2%}  "
              f"{r.get('elapsed_sec', 0)}s")
    print("=" * 58 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_status()
    else:
        retrain()