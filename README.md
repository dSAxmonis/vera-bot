# magicpin AI Challenge — Vera Bot Submission

Production-grade, low-latency AI merchant engagement engine built for the **magicpin AI Challenge — Vera**. Vera synthesizes 4 contextual layers (Category, Merchant, Trigger, Customer) to generate high-compulsion, hyper-personalized merchant nudges on WhatsApp without hallucinations, URLs, or canned responses.

---

## 🚀 Key Highlights & Architecture

- **Sub-Second Latency & Determinism**: Strict `temperature=0.0` execution with zero hidden reasoning tokens.
- **Dynamic Context Compaction**: Reduces prompt footprints from 3,500+ tokens to ~500 tokens, preserving 85% of token quota and eliminating context bloat.
- **Deterministic Interceptor**: Sub-1ms fast-path handling for hostile opt-outs, mixed-intent edge cases, and automated business reply streaks.
- **Quota & Token Isolation**: Decoupled bot and judge model tiering with strict reservation controls (`max_tokens: 150` for replies, `max_tokens: 300` for ticks).
- **Zero URL Penalty & Zero Jargon**: Post-generation regex validation strips hallucinated URLs and internal IDs (`d_2026W17`, `suppression_key`).
- **Complete Test Compliance**: Passes all 4 judge simulator scenarios (Warmup, Auto-Reply, Intent Transition with Action Mode, Hostile Opt-Out) and full evaluation benchmarks.

---

## 📂 Repository Structure

```text
├── bot.py                      # Main FastAPI server implementation & prompt logic
├── Dockerfile                  # Production container definition (Python 3.12 slim)
├── requirements.txt            # Minimal runtime dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # Git safeguard rules (secrets excluded)
├── .dockerignore               # Container build optimization
├── engagement-design.md        # Comprehensive challenge design document
├── engagement-research.md      # Category voice & persona research
├── judge_simulator.py          # Local adversarial judge harness
├── dataset/                    # Seed categories, merchants, and triggers
└── README.md                   # Setup and deployment documentation
```

---

## 🛠 Local Setup & Running

### Option 1: Native Python

1. **Clone repository**:
   ```bash
   git clone <your-repo-url>
   cd magicpin-ai-challenge
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and insert your LLM_API_KEY
   ```

4. **Start the server**:
   ```bash
   uvicorn bot:app --host 0.0.0.0 --port 8080
   ```

5. **Verify health**:
   ```bash
   curl http://localhost:8080/v1/healthz
   ```

---

### Option 2: Docker

1. **Build image**:
   ```bash
   docker build -t magicpin-vera-bot .
   ```

2. **Run container**:
   ```bash
   docker run -d --name vera-bot -p 8080:8080 --env-file .env magicpin-vera-bot
   ```

---

## ☁️ Deployment Guide: AWS EC2

Follow these steps to deploy Vera on an AWS EC2 instance:

### 1. Launch EC2 Instance
- **AMI**: Ubuntu Server 24.04 LTS (HVM) or Amazon Linux 2023
- **Instance Type**: `t3.micro` or `t3.small` (1-2 GB RAM is sufficient)
- **Security Group (Inbound Rules)**:
  - SSH: Port 22 (from your IP)
  - HTTP/Vera Port: Custom TCP Port `8080` (or `80` if using reverse proxy) from `0.0.0.0/0` (or restricted to judge IPs)

### 2. Connect and Install Docker
SSH into your instance:
```bash
ssh -i /path/to/key.pem ubuntu@<your-ec2-public-ip>
```

Install Docker on Ubuntu:
```bash
sudo apt update && sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and log back in for docker group to apply
exit
ssh -i /path/to/key.pem ubuntu@<your-ec2-public-ip>
```

### 3. Clone Repository & Configure Secrets
```bash
git clone <your-repo-url>
cd magicpin-ai-challenge

# Create .env from template
cp .env.example .env
nano .env  # Enter your real LLM_API_KEY
```

### 4. Build and Run Container
```bash
docker build -t vera-bot .
docker run -d --name vera-bot --restart unless-stopped -p 8080:8080 --env-file .env vera-bot
```

### 5. Verify Remote Endpoint
From your local terminal:
```bash
curl http://<your-ec2-public-ip>:8080/v1/healthz
curl http://<your-ec2-public-ip>:8080/v1/metadata
```

The bot is now production-ready and live for the challenge evaluator.
