# MindBridge — CI/CD & Deployment Guide

## Why this architecture

MindBridge's whole premise is that personal data stays on the *user's*
device. That means deployment has to split cleanly:

| Component     | Where it runs                          | Why                                            |
|----------------|-----------------------------------------|-------------------------------------------------|
| `app.py`       | Your cloud VM (always-on)               | FL aggregation + model hosting must be central  |
| `retrain.py`   | Same VM, sidecar container, nightly     | Needs disk access to real `user_histories/`     |
| `client.py`    | Each user's own machine                 | This is where their private data must stay      |

Deploying `client.py` to your own cloud would defeat the entire privacy
design — so CD only ever deploys the **server**, and instead **packages**
the client as a Docker image + downloadable bundle for users to run
themselves.

---

## Pipeline overview

```
push to any branch
  └─ ci.yml: lint → unit tests → validate both Dockerfiles build
       (no model downloads, no GPU, fast — mocks MindBridge entirely)

push to main
  └─ cd.yml:
       1. re-runs ci.yml as a gate
       2. builds + pushes mindbridge-server and mindbridge-client images to GHCR
       3. SSHes into the VM, pulls the new server image, restarts via
          docker-compose (server + Caddy TLS proxy + Ollama + retrainer)
       4. runs a post-deploy health check against /status
       5. packages a client-bundle .zip as a downloadable artifact
```

Training (`retrain.py`) is **not** part of CI. It runs as a long-lived
sidecar container on the VM itself, on a nightly loop, because it needs
the real `user_histories/` volume that only exists on the production box —
a GitHub Actions runner has no access to that and shouldn't.

---

## One-time setup

### 1. Provision the VM
Pick a small VM — 2 vCPU / 4GB RAM is comfortable for DistilBERT + Flask +
Ollama (e.g. DigitalOcean Droplet, AWS EC2 `t3.small`, or similar). Then:

```bash
ssh you@your-vm-ip
curl -fsSL https://raw.githubusercontent.com/<org>/<repo>/main/deploy/provision_vm.sh | bash
```

This installs Docker, opens ports 80/443, and prints the values you need
for step 2.

### 2. Point a domain at the VM
Add an A record: `mindbridge.yourdomain.com → <VM IP>`.
Caddy (in `proxy/Caddyfile`) auto-provisions a real Let's Encrypt cert for
this domain on first boot — no more self-signed-cert browser warnings, and
the voice-input feature (which requires HTTPS) works for every visitor
without manual "proceed anyway" clicks.

### 3. Add GitHub Actions secrets
Repo → Settings → Secrets and variables → Actions:

| Secret               | Value                                          |
|-----------------------|-------------------------------------------------|
| `VM_HOST`             | Your VM's IP                                   |
| `VM_USER`             | SSH user (printed by provision script)         |
| `VM_SSH_KEY`          | Private key matching an `authorized_keys` entry on the VM |
| `MINDBRIDGE_DOMAIN`   | e.g. `mindbridge.yourdomain.com`               |

`GITHUB_TOKEN` (for pushing to GHCR) is provided automatically — no setup
needed.

### 4. First deploy
```bash
git push origin main
```
Watch the Actions tab. On success, `https://mindbridge.yourdomain.com`
should respond.

### 5. Pull the LLM into Ollama (once)
```bash
ssh you@your-vm-ip
cd /opt/mindbridge
docker compose exec ollama ollama pull llama3
```

---

## How end users get the client

After a successful CD run, two options exist:

**Docker (recommended)**
```bash
docker run -p 5001:5001 \
  -v mindbridge_data:/app/client_data \
  ghcr.io/<org>/mindbridge-client:latest \
  --server mindbridge.yourdomain.com
```

**Downloadable bundle**
Grab `mindbridge-client-bundle` from the latest successful CD run's
Actions artifacts (or attach it to a GitHub Release if you want a
public-facing download link), unzip, and follow `RUN_ME.md` inside.

Either way, their emotion history/profile/avatar stay in a volume or
folder on **their** machine — nothing in this pipeline ever uploads it
anywhere.

---

## What's intentionally NOT automated

- **Model retraining accuracy** is not gated in CI — `retrain.py` runs
  independently on its own nightly schedule on the VM. If you want CI to
  sanity-check the *training code itself* (not the model quality), add a
  tiny synthetic-data smoke test as a separate optional job — don't run
  real FedAvg rounds in CI, it's slow and non-deterministic.
- **Client deployment** — by design, never automated to a shared server.
- **Ollama model pulls** — one-time manual step; automating it means CD
  pulls several GB on every deploy for no reason (the model rarely changes).

---

## Rollback

Every image is tagged with both `:latest` and `:<git-sha>`. To roll back:

```bash
ssh you@your-vm-ip
cd /opt/mindbridge
export SERVER_IMAGE=ghcr.io/<org>/mindbridge-server:<previous-sha>
docker compose up -d
```

Or re-run the CD workflow via **workflow_dispatch** from an older commit.
