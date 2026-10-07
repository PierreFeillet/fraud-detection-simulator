# Approach 2: Two-Stage LLM Generator for Synthetic Banking Activity

> Part of the [Synthetic Banking Fraud Simulator](../README.md). This approach improves on the realism of the fast, statistical [Markov chain simulator](../markov_chain_approach/README.md) by using two LLMs with separate roles.

The statistical simulator is fast, but it samples each field from hand-written distributions, so a single transaction can be internally inconsistent: an IP that doesn't match the city, an impossible travel time, a merchant that doesn't fit the profile. This pipeline uses LLMs to generate **context-aware, coherent sequences**. It keeps cost and reliability under control with structured strategies, strict JSON validation and retries.

The output is a CSV of labelled legitimate and fraudulent transactions, suitable for fraud detection, anomaly detection and sequence modeling experiments.

---

## The two stages

```
 Stage 1: STRATEGIST LLM                    Stage 2: ACTIVITY GENERATOR LLM
 (meta-llama/llama-3-405b-instruct)         (mistralai/mistral-large)
 src/generate_strategies.py                 src/generate_activities.py

 fraud type / legit profile                 strategy + user's full history + past errors
            │                                              │
            ▼                                              ▼
 structured JSON strategy  ──────────────▶  next transaction (JSON) ──▶ validate ──▶ CSV
 saved in strategies/*.json                                   ▲            │ fail
                                                              └── retry ◀──┘
```

### Stage 1: Strategist
`src/generate_strategies.py` asks a large model to design a **behaviour strategy** for each fraud type and legitimate profile:

- **Fraud types:** Money Laundering, Account Takeover, Synthetic Identity Fraud, Identity Theft, Card Skimming, Loan Fraud, Check Fraud, Wire Fraud, Ponzi Scheme, Cryptocurrency Fraud, Insider Trading
- **Legitimate profiles:** Saver, Investor, Traveler, Everyday Spender, Business Owner, Student, Retiree, Frequent Online Shopper, Tech Professional, Freelancer

Each strategy is a JSON object with: `n_accounts`, `transaction_types_involved`, `typical_amount_range`, `geographic_focus` (constrained to the regions in `geography.py`), `velocity`, `network_types`, `common_devices`, IP ranges, `common_merchant_names`, `common_recipient_ids`, `common_recipient_banks` and a short narrative `context`.

Strategies are cached in `strategies/fraud_strategies_<model>.json` and `strategies/legitimate_strategies_<model>.json`. An existing strategy is reused, not regenerated, so this stage runs rarely and cheaply.

### Stage 2: Activity Generator
`src/generate_activities.py` builds each user's sequence **one transaction at a time**:

1. Accounts are created for the user according to the strategy (`n_accounts`), with initial balances.
2. For each activity:
   - **First activity, or legitimate user → static generation** (`static_behavior.py`), using the fast rule-based sampler.
   - **Fraudulent user, after the first activity → LLM generation**:
     - The prompt contains the strategy, the user's **full transaction history** across all their accounts, the available accounts, and **recently observed JSON errors** so the model avoids repeating them.
     - The model must keep timestamps strictly increasing, keep travel times between locations plausible, derive IPs from the location, and keep amounts inside the strategy range and the account balance.
   - The response is parsed and **validated** (see below). Failures are retried and logged as a −1 "reward", successes as +1.
3. Balances are recomputed in Python (not trusted from the LLM), and metadata and labels are attached.
4. Rows are written to CSV in buffered batches.

Fraudulent sequences are generated first until the target fraud rate is reached, then legitimate ones until the total target is reached. Each user produces 1–6 activities.

---

## LLM JSON contract

The activity model must return **exactly one activity** between `<<<JSON>>> ... <<<END_JSON>>>` (fenced ```` ```json ```` blocks are also accepted).

Required fields:
- `bank_timestamp` (ISO 8601 UTC, strictly increasing)
- `local_timestamp` (ISO 8601, consistent with the location's timezone)
- `account_id` (format `ACC-XXXXXXXX`)
- `type`, `amount`, `balance_before`
- `location`, `ip_address`, `device_id`, `network_type`
- `merchant_name` (required for purchases; must be null for transfers)
- `recipient_id`, `recipient_bank` (required for transfers; must be null otherwise)

---

## Output dataset

`data/<activity_model>/fraud_simulation_activities_<N>.csv`, one row per activity, with columns:

- `transaction_id`, `user_id`
- timestamps: `bank_timestamp`, `local_timestamp`
- geo/device/network: `location`, `ip_address`, `device_id`, `network_type`
- transaction fields: `type`, `amount`, `account_id`, `balance_before`, `granted`, `balance_after`
- recipient fields: `merchant_name`, `recipient_id`, `recipient_bank`
- labels: `behavior_type`, `fraud_label`

### Run logs
`outputs/<activity_model>/<N>/`:
- `json_errors.log`: JSON validation failures
- `error_tracking.json`: aggregated error counts over time
- `reward_progress.csv`: per-attempt signal (+1 valid / −1 invalid)
- `llm_chain_of_thought.txt`: raw model output
- `success_rate.png`: cumulative rate of valid JSON generations

---

## Setup

1. Install dependencies from the repo root (Python 3.11):
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file (never commit it) with your **IBM watsonx.ai** credentials:
   ```
   API_KEY=<your watsonx api key>
   PROJECT_ID=<your watsonx project id>
   ```
3. Models and generation parameters (temperature, top-p, max tokens, stop sequences) are configured in `src/watsonx_helper.py`.

## Running

From inside `LLMs_approach/`:

```bash
# Stage 1: generate (or reuse cached) strategies
python src/generate_strategies.py

# Stage 2: generate the dataset
python src/generate_activities.py --nb_activities 1000 --target_fraud_percentage 0.05
```

| Argument | Default | Description |
|---|---|---|
| `--nb_activities` | required | Total number of activities to generate |
| `--target_fraud_percentage` | 0.05 | Fraction of fraudulent activities |

## Key modules

| File | Role |
|---|---|
| `generate_strategies.py` | Stage 1: Strategist LLM |
| `generate_activities.py` | Stage 2: orchestration, prompt building, JSON validation, retries, logging |
| `watsonx_helper.py` | watsonx.ai client, model IDs, generation parameters |
| `static_behavior.py` | Rule-based generation for first/legitimate activities |
| `geography.py` | Allowed regions/cities and location helpers |
| `utilities.py` | Hashing, balance updates |
| `clustering.py`, `analyze_sample.py`, `checks.py`, `dashboard.py` | Evaluation and inspection of generated data |
| `benchmark_SGen.py`, `count_tokens.py` | Benchmarking and token/cost estimation |
