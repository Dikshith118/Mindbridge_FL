"""
Federated Emotion Classifier — Two-Phase Training
Fixes applied:
  - Tokenizer owned per-dataset (no thread conflict)
  - Server in main thread (no signal error)
  - BERT frozen (saves GPU memory)
  - Only 2 clients at a time (fixes Windows socket buffer overflow)
  - Model saved after Phase 1 AND Phase 2
  - Phase 2 loads Phase 1 weights (no overwrite)
  - Reduced batch size and sequence length for GTX 1650 4GB
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # use cached weights, no downloads

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple
import pandas as pd
import json, time, threading
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NUM_CLIENTS    = 2        # KEY FIX: 2 prevents Windows socket buffer overflow
NUM_ROUNDS_P1  = 7
NUM_ROUNDS_P2  = 7
LOCAL_EPOCHS   = 1
BATCH_SIZE     = 4
MAX_LEN        = 32
NUM_LABELS     = 28
SERVER_ADDR_1  = "127.0.0.1:9191"
SERVER_ADDR_2  = "127.0.0.1:9292"
CLIENT_DIR     = "data/clients_28class"
TOKENIZER_NAME = "distilbert-base-uncased"

SAVE_DIR       = "saved_model"
MODEL_PHASE1   = os.path.join(SAVE_DIR, "model_phase1.pt")
MODEL_FINAL    = os.path.join(SAVE_DIR, "model_final.pt")
WEIGHTS_P1     = os.path.join(SAVE_DIR, "phase1_weights.npy")

os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
class EmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = DistilBertModel.from_pretrained(TOKENIZER_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, NUM_LABELS)
        # Freeze BERT — only train classifier head
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(cls)


def get_weights(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_weights(model, weights):
    state = {k: torch.tensor(v)
             for k, v in zip(model.state_dict().keys(), weights)}
    model.load_state_dict(state, strict=True)


def save_model(weights, path):
    model = EmotionClassifier()
    set_weights(model, weights)
    torch.save(model.state_dict(), path)
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Model saved → {path}  ({mb:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET — each instance owns its tokenizer
# ══════════════════════════════════════════════════════════════════════════════
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(self.labels[idx]),
                                           dtype=torch.long)
        }

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_all_data(data_dir):
    all_texts, all_labels = [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    print(f"\nLoading {len(files)} CSV files...")
    for fname in files:
        df = pd.read_csv(os.path.join(data_dir, fname)).dropna(
            subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        df = df[df["label"].between(0, NUM_LABELS - 1)]
        all_texts.extend(df["text"].tolist())
        all_labels.extend(df["label"].tolist())
        print(f"  {fname}: {len(df)} rows")

    idx        = np.random.permutation(len(all_texts))
    all_texts  = [all_texts[i]  for i in idx]
    all_labels = [all_labels[i] for i in idx]
    mid        = len(all_texts) // 2

    print(f"\nTotal: {len(all_texts)} | Phase1: {mid} | Phase2: {len(all_texts)-mid}\n")
    return (all_texts[:mid], all_labels[:mid]), \
           (all_texts[mid:], all_labels[mid:])


def make_partitions(texts, labels, n):
    size  = len(texts) // n
    parts = []
    for i in range(n):
        parts.append((texts[i*size:(i+1)*size],
                      labels[i*size:(i+1)*size]))
        print(f"  Client {i}: {size} samples")
    return parts

# ══════════════════════════════════════════════════════════════════════════════
#  FLOWER CLIENT
# ══════════════════════════════════════════════════════════════════════════════
class EmotionClient(fl.client.NumPyClient):
    def __init__(self, cid, partitions):
        texts, labels = partitions[int(cid)]
        split         = int(0.85 * len(texts))
        train_ds = EmotionDataset(texts[:split], labels[:split])
        val_ds   = EmotionDataset(texts[split:],  labels[split:])
        self.train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, pin_memory=False)
        self.val_loader   = DataLoader(
            val_ds, batch_size=BATCH_SIZE,
            num_workers=0, pin_memory=False)
        self.model     = EmotionClassifier().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.cid       = int(cid)
        print(f"[C{self.cid}] train={len(train_ds)} val={len(val_ds)}")

    def get_parameters(self, config=None):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-4)
        self.model.train()
        rnd   = (config or {}).get("server_round", "?")
        phase = (config or {}).get("phase",        "?")

        for epoch in range(LOCAL_EPOCHS):
            total, steps = 0.0, 0
            for batch in self.train_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                loss = self.criterion(self.model(ids, mask), lbls)
                loss.backward()
                optimizer.step()
                total += loss.item()
                steps += 1
                if steps % 50 == 0 and DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            print(f"[P{phase}][C{self.cid}] Rnd {rnd} | "
                  f"Ep {epoch+1} | Loss: {total/steps:.4f}")

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                ids  = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                lbls = batch["label"].to(DEVICE)
                logits  = self.model(ids, mask)
                loss   += self.criterion(logits, lbls).item() * ids.size(0)
                correct += (logits.argmax(1) == lbls).sum().item()
                total   += lbls.size(0)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        acc = correct / total if total else 0.0
        print(f"[C{self.cid}] Eval Acc: {acc:.4f}")
        return loss/total if total else 0.0, \
               len(self.val_loader.dataset), {"accuracy": float(acc)}

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY
# ══════════════════════════════════════════════════════════════════════════════
accuracy_log = []


def build_strategy(phase, initial_params=None):

    class PhaseStrategy(FedAvg):

        def aggregate_fit(self, server_round, results, failures):
            agg = super().aggregate_fit(server_round, results, failures)
            if agg:
                params, _ = agg
                weights   = parameters_to_ndarrays(params)

                # Save after Phase 1 last round
                if phase == 1 and server_round == NUM_ROUNDS_P1:
                    print("\n[Server] Phase 1 complete — saving...")
                    save_model(weights, MODEL_PHASE1)
                    np.save(WEIGHTS_P1,
                            np.array(weights, dtype=object),
                            allow_pickle=True)
                    print(f"  Weights → {WEIGHTS_P1}")
                    print("  ✅ You can now test chatbot with model_phase1.pt\n")

                # Save after Phase 2 last round
                if phase == 2 and server_round == NUM_ROUNDS_P2:
                    print("\n[Server] Phase 2 complete — saving FINAL model...")
                    save_model(weights, MODEL_FINAL)
                    print("  ✅ Final model saved!\n")

            return agg

        def aggregate_evaluate(self, server_round, results, failures):
            agg = super().aggregate_evaluate(server_round, results, failures)
            if results:
                total = sum(r.num_examples for _, r in results)
                acc   = sum(r.num_examples * r.metrics["accuracy"]
                            for _, r in results) / total
                g     = server_round + (NUM_ROUNDS_P1 if phase == 2 else 0)
                accuracy_log.append((g, acc))
                print(f"\n{'='*50}")
                print(f"  [Phase {phase}] Round {server_round} "
                      f"| Global {g} | Acc: {acc:.4f}")
                print(f"{'='*50}\n")
            return agg

        def configure_fit(self, server_round, parameters, client_manager):
            pairs = super().configure_fit(
                server_round, parameters, client_manager)
            return [(p, (lambda fi, sr=server_round, ph=phase: (
                fi.config.update({"server_round": sr, "phase": ph}),
                fi)[1])(fit_ins))
                    for p, fit_ins in pairs]

    kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=lambda m: {
            "accuracy": sum(n*x["accuracy"] for n,x in m) / sum(n for n,_ in m)
        },
    )
    if initial_params is not None:
        kwargs["initial_parameters"] = initial_params
    return PhaseStrategy(**kwargs)

# ══════════════════════════════════════════════════════════════════════════════
#  CLIENT LAUNCHER — 3s stagger prevents socket buffer overflow
# ══════════════════════════════════════════════════════════════════════════════
def start_clients(server_address, partitions, delay=10):
    def launcher():
        print(f"\nWaiting {delay}s for server...")
        time.sleep(delay)
        print(f"Launching {NUM_CLIENTS} clients (3s stagger)...\n")
        threads = []
        for cid in range(NUM_CLIENTS):
            def run(c=cid):
                try:
                    fl.client.start_numpy_client(
                        server_address=server_address,
                        client=EmotionClient(c, partitions),
                    )
                except Exception as e:
                    print(f"[C{c}] ERROR: {e}")
            t = threading.Thread(target=run, daemon=True)
            threads.append(t)
            t.start()
            time.sleep(3)   # ← stagger prevents buffer overflow
        for t in threads:
            t.join()
        print("All clients done.")

    t = threading.Thread(target=launcher, daemon=True)
    t.start()
    return t

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_results():
    if not accuracy_log:
        return
    rounds     = [r for r, _ in accuracy_log]
    accuracies = [a for _, a in accuracy_log]

    plt.figure(figsize=(13, 5))
    plt.plot(rounds, accuracies, marker="o",
             color="#3B82F6", linewidth=2, markersize=8)
    plt.fill_between(rounds, accuracies, alpha=0.1, color="#3B82F6")

    if len(rounds) > NUM_ROUNDS_P1:
        b = NUM_ROUNDS_P1 + 0.5
        plt.axvline(x=b, color="#EF476F", linestyle="--",
                    linewidth=1.5, label="Phase 1 → Phase 2")
        plt.text(b+0.1, 0.03, "Phase 2", color="#EF476F", fontsize=9)

    plt.title(f"Federated Emotion Classifier — {len(rounds)} Rounds",
              fontsize=13)
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rounds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_graph.png", dpi=150)
    print("Graph → accuracy_graph.png")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("Pre-caching DistilBERT (one-time)...")
    DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    DistilBertModel.from_pretrained(TOKENIZER_NAME)
    print("Cache ready.\n")

    (p1_texts, p1_labels), (p2_texts, p2_labels) = load_all_data(CLIENT_DIR)

    print("="*55)
    print("  Emotion FL — Full Dataset Two-Phase")
    print(f"  Total     : {len(p1_texts)+len(p2_texts)} samples")
    print(f"  Clients   : {NUM_CLIENTS}  (2 = no socket overflow)")
    print(f"  Rounds    : {NUM_ROUNDS_P1}+{NUM_ROUNDS_P2} = 14 total")
    print(f"  Batch     : {BATCH_SIZE}  MaxLen: {MAX_LEN}")
    print(f"  Device    : {DEVICE}")
    print(f"  Saves     : {MODEL_PHASE1}")
    print(f"              {MODEL_FINAL}")
    print("="*55)

    # ── PHASE 1 ──────────────────────────────────────────────────────────────
    print("\n--- Phase 1 partitions ---")
    p1_parts = make_partitions(p1_texts, p1_labels, NUM_CLIENTS)
    l1 = start_clients(SERVER_ADDR_1, p1_parts, delay=10)

    print("\n### PHASE 1 SERVER ###\n")
    fl.server.start_server(
        server_address=SERVER_ADDR_1,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS_P1),
        strategy=build_strategy(phase=1),
    )
    l1.join(timeout=60)

    print("\nPhase 1 done. Pausing 8s...\n")
    time.sleep(8)

    # ── PHASE 2 ──────────────────────────────────────────────────────────────
    print("--- Phase 2 partitions ---")
    p2_parts = make_partitions(p2_texts, p2_labels, NUM_CLIENTS)

    init_params = None
    if os.path.exists(WEIGHTS_P1):
        raw         = np.load(WEIGHTS_P1, allow_pickle=True)
        init_params = ndarrays_to_parameters(list(raw))
        print("✅ Phase 1 weights loaded — model continues\n")
    else:
        print("⚠️  No Phase 1 checkpoint — Phase 2 starts fresh\n")

    l2 = start_clients(SERVER_ADDR_2, p2_parts, delay=10)

    print("### PHASE 2 SERVER ###\n")
    fl.server.start_server(
        server_address=SERVER_ADDR_2,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS_P2),
        strategy=build_strategy(phase=2, initial_params=init_params),
    )
    l2.join(timeout=60)

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    with open("accuracy.json", "w") as f:
        json.dump({
            "rounds":     [r for r, _ in accuracy_log],
            "accuracies": [a for _, a in accuracy_log],
        }, f, indent=2)
    print("accuracy.json saved")
    plot_results()

    if accuracy_log:
        accs = [a for _, a in accuracy_log]
        print(f"\n{'='*50}")
        print(f"  TRAINING COMPLETE")
        print(f"  Rounds completed : {len(accs)}")
        print(f"  Final accuracy   : {accs[-1]:.4f}")
        print(f"  Best  accuracy   : {max(accs):.4f}")
        print(f"  Phase 1 model    : {MODEL_PHASE1}")
        print(f"  Final model      : {MODEL_FINAL}")
        print(f"{'='*50}")