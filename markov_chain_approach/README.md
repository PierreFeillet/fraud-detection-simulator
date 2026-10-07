# Approach 1: Bank Fraud Simulation Using Markov Chains

> Part of the [Synthetic Banking Fraud Simulator](../README.md). This is the **fast, statistical** approach. For the more realistic LLM-based approach, see [`../LLMs_approach/`](../LLMs_approach/README.md).

This simulator models legitimate and fraudulent user behaviour with **Markov chains**. Activity amounts, locations, devices and networks are drawn from **statistical distributions**. Because no external model is involved, it is fully transparent and scales to millions of activities: 1M activities in **~TODO minutes**. The folder also includes scripts for clustering analysis and feature evaluation of the generated data.

## Table of Contents
1. Introduction
2. System Architecture
3. Markov Chain Approach
4. Agent and Activity Definitions
5. Behavior Catalog
6. Location Modeling
7. Transaction Amount Distributions
8. Balance Computation
9. Simulation Workflow
10. Evaluation and Goodness Check
11. Limitations and Advantages
12. Project Structure
13. Running the Simulation
14. Output and Data Interpretation
15. Customization and Extensions
16. Known Issues and Next Steps

## 1. Introduction
This system simulates banking activities using Markov chains, so researchers and developers can generate realistic datasets for testing fraud detection models. The simulation replicates the patterns of everyday users and fraudsters, and saves detailed transaction logs as CSV files.

## 2. System Architecture
The system is modular, consisting of:
- **simulator.py**: Main script that runs the simulation.
- **activity.py**: Defines and manages user activities, including transaction details and locations.
- **agent.py**: Represents legitimate users and fraudsters, defining their behaviors and attributes.
- **bank.py**: Manages transaction logging.
- **catalog.py**: Stores behavior definitions, transition matrices, and location data.
- **distributions.py**: Defines transaction amount distributions.
- **clustering.py**: Analyzes clusters emerging from the dataset using DBSCAN, KMeans, Hierarchical Clustering, and Isolation Forest.
- **analyze_sample.py**: Evaluates feature importance and correlations using Random Forest and other statistical methods.

## 3. Markov Chain Approach
The Markov chain approach models the probability of a sequence of activities using a transition matrix. In `catalog.py`, each activity has a probability array that gives the likelihood of transitioning to each other activity. These probabilities are currently constant; future versions could make them depend on variables such as location and transaction amount.

Future enhancements could include tailored distributions for specific profiles, such as High-Spender, Student and Investor for legitimate users, and distinct patterns for each type of fraud.

## 4. Agent and Activity Definitions
### Agent (agent.py)
- **Real ID:** Unique identifier for the agent.
- **Virtual ID:** Used when impersonating victims.
- **Initial Balance:** Starting balance assigned randomly within a defined range.
- **Initial Country:** Geographic location where the agent primarily transacts.
- **Agent Type:** Can be either “traveler” or “static,” influencing location probabilities.
- **Visited Countries:** A dictionary storing countries and their transaction probabilities.

### Activity (activity.py)
- **Inheritance:** The `Activity` class extends the `Agent` class, inheriting key attributes.
- **Transaction Location:** Determined based on the agent’s type and visited countries.
- **Device and Network:** Randomly chosen from predefined lists with associated probabilities.
- **Compromised Status:** Simulates whether the device or network is compromised. A planned improvement is to model this as a pattern instead of assigning it randomly.
- **Transaction Amount:** Defined based on the activity type and agent’s behavior.
- **Balance Update:** Adjusted after each transaction, with validation for insufficient funds.
- **Merchant Categories (Planned):** Future versions could include merchant categories to provide additional context for transaction data.

## 5. Behavior Catalog
The `catalog.py` file defines behaviors and their properties, including:
- **Legitimate**
- **Card Skimming**
- **Identity Theft**
- **Money Laundering**
- **Synthetic Identity Fraud**

Each behavior is associated with:
- **Set of Activities:** Possible actions for the behavior.
- **Transition Matrix:** Probability of transitioning between activities.
- **Time Limit:** Maximum time between activities (e.g., 15 minutes for card skimming).
- **Fraud Label:** Indicates whether the behavior is considered fraudulent.

For identity theft, two users share the same virtual ID but have different real IDs.

## 6. Location Modeling
Location assignments are based on probability distributions:
- **Travelers:** Probability shifts gradually from the home country to foreign countries.
- **Static Agents:** Primarily transact in their home country, with rare foreign transactions.

A possible improvement is to pick locations from a distribution centred on a defined barycenter, so that locations further from it are less likely.

## 7. Transaction Amount Distributions
Transaction amounts follow a log-normal distribution defined by parameters μ and σ:
- **Legitimate behavior:** μ = 5, σ = 1, with a peak around $150.
- **Fraudulent behavior:** μ = 8, σ = 1.2, with a peak around $3000.

Future versions could include specific log-normal distributions for different legitimate profiles and fraud scenarios.

## 8. Balance Computation
In `catalog.py`, each activity is labeled "neutral", "positive" or "negative", depending on its impact on the balance. For example, a "Purchase" is "negative". A "Login" is "neutral": its `amount` is 0 and the balance does not change. The initial balance is assigned randomly, and later balances are updated from transaction amounts. If a transaction amount exceeds the current balance, the transaction is declined and the `granted` field is set to False; otherwise it is set to True.

## 9. Simulation Workflow
1. **Activity Selection:** Determined using Markov chain transition probabilities.
2. **Transaction Execution:** Amounts are drawn from the appropriate distributions.
3. **Location Assignment:** Based on agent type, visited countries, and fraud behavior.
4. **Balance Validation:** Transactions are approved if the balance is sufficient.
5. **Logging:** Each transaction is recorded with timestamps, amounts, and labels.
6. **Account Closure:** Agents are removed after performing the “Close Account” activity.

## 10. Evaluation and Goodness Check
### Clustering Analysis (clustering.py)
- **DBSCAN:** Density-based clustering
- **KMeans:** Partition-based clustering
- **Hierarchical Clustering:** Visualized using dendrograms
- **Isolation Forest:** Anomaly detection

### Feature Evaluation (analyze_sample.py)
- **Random Forest Classifier:** Measures feature importance and predicts fraud labels
- **Decision Tree:** Visualizes decision rules
- **Correlation Analysis:** Uses Pearson correlation and Cramér's V
- **Heatmaps:** Visualizes feature correlations

## 11. Limitations and Advantages
### Advantages
- Very fast and scalable: generates millions of activities on a laptop.
- Simple to control and interpret.
- Transparent and easily explainable modeling process.

### Limitations
- Large and sparse transition matrices are required for modeling complex sequences.
- Transition probabilities should ideally depend on variables like amount, location, and time.
- Fields are sampled largely independently, so cross-field consistency (location vs. IP vs. merchant vs. timing) is limited.
- Realistic modeling requires domain expertise in banking.

These limitations motivated the [two-stage LLM approach](../LLMs_approach/README.md).

## 12. Project Structure
```
markov_chain_approach/
├── src/
│   ├── simulator.py
│   ├── activity.py
│   ├── agent.py
│   ├── bank.py
│   ├── catalog.py
│   ├── distributions.py
│   ├── generate_matrix.py
│   ├── clustering.py
│   ├── analyze_sample.py
│   └── dashboard.py
├── data/                Output datasets are saved here
├── plots/               Analysis plots
└── clustering_results/
```

## 13. Running the Simulation
From inside `markov_chain_approach/`:
```bash
python src/simulator.py --nb_activities 1000000 --min_n_agents 40 --fraudster_rate 0.1 --data_folder data --start_time "2025-01-06T12:00:00"
```

| Argument | Default | Description |
|---|---|---|
| `--nb_activities` | required | Total number of activities to generate |
| `--min_n_agents` | 40 | Number of legitimate agents |
| `--fraudster_rate` | 0.1 | Fraction of fraudulent agents (≈0.5–2% low risk, 2–5% moderate, 5–10% high risk) |
| `--data_folder` | `data` | Where to save the dataset |
| `--start_time` | now | Initial timestamp (ISO 8601) |

## 14. Output and Data Interpretation
Simulation results are saved as CSV files, with each row representing an activity:
- **`agent_id`**: Unique identifier of the agent.
- **`timestamp`**: Activity execution time.
- **`behavior`**: Type of behavior (legitimate or fraudulent).
- **`activity_type`**: Specific activity performed.
- **`amount`**: Transaction amount, positive or negative.
- **`balance`**: Agent’s balance after the activity.
- **`fraudulent`**: Binary flag indicating fraud.

## 15. Customization and Extensions
- **Adding New Behaviors:** Define additional behaviors in `catalog.py`.
- **Modifying Transaction Amounts:** Adjust log-normal parameters in `distributions.py`.
- **Changing Agent Logic:** Customize `assign_behavior()` and `run_simulation_step()` in `simulator.py`.

## 16. Known Issues and Next Steps
- Legitimate and fraudulent activities are currently quite well separated. More overlap would make the data more realistic, which requires bigger matrices or less variability in activities across the different sequences.
