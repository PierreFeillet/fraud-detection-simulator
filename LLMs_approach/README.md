# Synthetic Banking Activity Generator (Fraud + Legit)

This repository contains a Python pipeline that generates **synthetic banking transaction logs** mixing **legitimate** and **fraudulent** user behavior. It combines:
- **Rule-based / static generation** (for first transactions and legitimate profiles)
- **LLM-based generation** (for fraudulent profiles after the first event), with JSON validation + retry
- Streaming writes to CSV + log files for debugging and quality tracking

The output is a CSV dataset suitable for fraud detection experiments, anomaly detection, sequence modeling, or simulation.

---

## What this pipeline produces

A CSV file with one row per activity and the following ordered columns:

- `transaction_id`, `user_id`
- timestamps: `bank_timestamp`, `local_timestamp`
- geo/device/network: `location`, `ip_address`, `device_id`, `network_type`
- transaction fields: `type`, `amount`, `account_id`, `balance_before`, `granted`, `balance_after`
- recipient fields: `merchant_name`, `recipient_id`, `recipient_bank`
- labels: `behavior_type`, `fraud_label`

Fraud rate is controlled via `--target_fraud_percentage`.

---

## High-level flow

1. **Load behavior strategies**
   - Fraud strategies: `strategies/fraud_strategies_<strategy_model>.json`
   - Legit strategies: `strategies/legitimate_strategies_<strategy_model>.json`

2. **Generate activities**
   - Generate fraudulent activities first until the fraud target is reached
   - Generate legitimate activities until total target is reached
   - Each “agent” (user) generates `1..6` activities in a sequence

3. **Per-sequence generation**
   - Accounts are created per user based on strategy (`n_accounts`)
   - For each activity:
     - If it’s the first activity OR the user is legitimate → **static generation**
     - Else (fraudulent, after the first) → **LLM generation**
       - Build an LLM prompt using strategy + full history + prior errors
       - Validate the produced JSON (schema, types, timestamps strictly increasing)
       - Retry on failure and log error signals as “reward” events

4. **Persist**
   - Buffered writes to CSV (`buffer_size` controls batching)
   - Reward and error logs saved into an output directory
   - Success-rate plot generated from the reward log

---

## Key modules and responsibilities

### Main script (this file)
Core orchestration:
- `generate_activities(...)` – main dataset generator (fraud first, then legit)
- `generate_activity_sequence(...)` – generates a sequence for one user
- `build_generation_prompt(...)` – builds LLM prompt from strategy + history + errors
- `validate_json(...)` – validates the LLM JSON output and enforces constraints
- `flush_buffer(...)` – batched appends to CSV

### Imported local modules (expected to exist)
- `watsonx_helper.py`
  - `watsonx_chat(...)`, model IDs and parameter defaults
- `static_behavior.py`
  - `generate_static_activity(...)`
  - `assign_activity_fields(...)`
  - `assign_initial_balance(...)`
  - `select_valid_location(...)`
  - `generate_local_and_bank_timestamp(...)`
- `utilities.py`
  - `generate_random_hash(...)`
  - `update_balance(...)`

> Note: this script also defines a `generate_random_hash(...)` locally, but it also imports one from `utilities`. Make sure you only rely on one version to avoid confusion.

---

## LLM JSON contract

When using the LLM path (fraudulent after first activity), the model must return **exactly one activity** inside delimiters.

Accepted delimiters include:
- `<<<JSON>>> ... <<<END_JSON>>>`
- or fenced blocks like:
  - ```json ... ```

The generated JSON must match the expected schema:

Required fields:
- `bank_timestamp` (ISO 8601, strictly increasing)
- `local_timestamp` (ISO 8601, consistent with location TZ)
- `account_id` (format `ACC-XXXXXXXX`)
- `type`, `amount`, `balance_before`
- `location`, `ip_address`, `device_id`, `network_type`
- `merchant_name` (required for purchases; must be null for transfers)
- `recipient_id`, `recipient_bank` (required for transfers; must be null otherwise)

Validation failures trigger retries and are logged.

---

## Outputs

The pipeline writes into two main locations:

### Dataset CSV
Stored under:
- `data/<activity_model>/fraud_simulation_activities_<N>.csv`

### Run logs and charts
Stored under:
- `outputs/<activity_model>/<N>/`

Files include:
- `json_errors.log` — JSON validation failures (one JSON list per line)
- `error_tracking.json` — aggregated error counts over time
- `reward_progress.csv` — per-attempt reward signals (+1 success / -1 failure)
- `llm_chain_of_thought.txt` — raw model output (whatever the model returned)
- `success_rate.png` — plot of cumulative valid JSON generations

---

## Requirements

Python packages used:
- `pandas`
- `matplotlib`
- `python-dateutil`
- `ollama` (optional, depending on your setup)
- `IPython` (only used for debugging via `embed`)

Also required:
- local modules: `watsonx_helper.py`, `static_behavior.py`, `utilities.py`
- strategy files under `strategies/`

Install dependencies (example):
```bash
pip install pandas matplotlib python-dateutil ipython
