# Synthetic Banking Fraud Simulator

Generate **realistic, labelled synthetic banking transaction logs** that mix legitimate customers and fraudsters, for training and benchmarking fraud-detection models when real banking data cannot be shared.

Real transaction data is sensitive, heavily regulated, and fraud labels are scarce. This project explores two complementary ways to produce a synthetic replacement:

| | **1. Statistical simulator (Markov chains)** | **2. Two-stage LLM generator** |
|---|---|---|
| Folder | [`markov_chain_approach/`](markov_chain_approach/) | [`LLMs_approach/`](LLMs_approach/) |
| How sequences are built | Transition matrices + sampling from statistical distributions | A *Strategist* LLM designs behaviour profiles, an *Activity Generator* LLM writes transactions that follow them |
| Strength | **Fast and scalable**: 1M activities in **~TODO minutes** on a laptop | **More realistic and context-aware**: coherent locations, IPs, devices, merchants, timing |
| Control / explainability | Fully transparent, every probability is in the catalog | Guided by structured strategies + strict JSON validation |
| Cost | CPU only, no external services | Requires LLM calls (IBM watsonx.ai) |
| Best for | Large-volume datasets, baselines, stress tests | High-fidelity fraud scenarios, smaller curated datasets |

---

## Approach 1: Statistical simulator (Markov chains)

A deterministic, rule-driven simulator. Each **agent** (legitimate user or fraudster) is assigned a behaviour, such as *Legitimate*, *Card Skimming*, *Identity Theft*, *Money Laundering* or *Synthetic Identity Fraud*. Its next activity is chosen from that behaviour's **Markov transition matrix**.

Transaction amounts are drawn from **log-normal distributions** (different for legitimate and fraudulent behaviour). Locations, devices and networks are sampled from probability tables. Balances are tracked, so transactions can be declined.

- Simple, interpretable and very fast: it scales to millions of rows.
- Limitation: realism is bounded by hand-written matrices and distributions. Fields are sampled largely independently, so cross-field consistency (e.g. IP vs. location vs. merchant) is limited.

➡️ Details, architecture and usage: [`markov_chain_approach/README.md`](markov_chain_approach/README.md)

## Approach 2: Two-stage LLM generator

To push realism further, generation is split between two LLMs with distinct roles:

```
 ┌──────────────────────┐   strategies (JSON)   ┌───────────────────────────┐   validated   ┌─────────┐
 │  Stage 1: STRATEGIST │ ────────────────────▶ │ Stage 2: ACTIVITY         │ ───────────▶  │ CSV log │
 │  (Llama 3 405B)      │  per fraud type /     │ GENERATOR (Mistral Large) │  JSON + retry │         │
 │  designs behaviour   │  legit profile        │ writes the next tx given  │               │         │
 │  profiles            │                       │ strategy + user history   │               │         │
 └──────────────────────┘                       └───────────────────────────┘               └─────────┘
```

1. **Strategist LLM** (`generate_strategies.py`): for each fraud type (Money Laundering, Account Takeover, Wire Fraud, …) and legitimate profile (Student, Retiree, Business Owner, …), it produces a structured JSON *strategy*. A strategy covers the number of accounts, transaction types, amount range, geographic focus, velocity, devices, networks, IP ranges, typical merchants and recipients, plus a short narrative context.
2. **Activity Generator LLM** (`generate_activities.py`): generates each user's transactions **one at a time**. Every prompt contains the user's strategy and full transaction history, so the sequence stays coherent: plausible travel times between cities, IPs consistent with location, and amounts within balance. Every output is **validated against a strict JSON schema** (field types, formats, strictly increasing timestamps). Failures are retried, and recurring errors are fed back into the next prompt.

To keep cost under control, the LLM is only used where it adds the most value: **fraudulent sequences after the first event**. Legitimate activity and first events reuse the fast statistical generator.

➡️ Details, prompt/JSON contract and usage: [`LLMs_approach/README.md`](LLMs_approach/README.md)

---

## Output

Both approaches produce a CSV with **one row per activity**: user/agent ID, timestamps, activity type, amount, balance, location and contextual fields, plus the **behaviour type and a binary fraud label**. The LLM approach adds richer context fields (local vs. bank timestamp, IP address, device, network, merchant, recipient account/bank).

Both folders also include evaluation scripts: clustering (DBSCAN, KMeans, hierarchical, Isolation Forest), feature importance (Random Forest, decision trees) and correlation analysis. Use them to check that the generated data has learnable but non-trivial fraud signal.

## Repository structure

```
fraud-detection-simulator/
├── README.md                  ← you are here (project overview)
├── requirements.txt
├── markov_chain_approach/     ← Approach 1: statistical simulator
│   ├── README.md
│   ├── src/                   simulator, agents, behaviour catalog, distributions, evaluation
│   ├── data/                  generated datasets
│   └── plots/                 analysis plots
└── LLMs_approach/             ← Approach 2: two-stage LLM generator
    ├── README.md
    ├── src/                   strategist, activity generator, validation, evaluation
    ├── strategies/            LLM-generated strategies (JSON)
    ├── data/                  generated datasets
    └── outputs/               run logs, JSON error tracking, success-rate plots
```

## Quick start

```bash
# Python 3.11
python3.11 -m venv fraud_env && source fraud_env/bin/activate
pip install -r requirements.txt

# Approach 1: statistical simulator
cd markov_chain_approach
python src/simulator.py --nb_activities 1000000 --min_n_agents 40 --fraudster_rate 0.1 --start_time "2025-01-06T12:00:00"
cd ..

# Approach 2: two-stage LLM generator (needs IBM watsonx.ai credentials, see its README)
cd LLMs_approach
python src/generate_strategies.py           # Stage 1: build strategies
python src/generate_activities.py --nb_activities 1000 --target_fraud_percentage 0.05   # Stage 2
```
