# Bank Fraud Simulation System Using Markov Chains

This README provides a comprehensive overview of the bank fraud simulation system using Markov chains, designed to model both legitimate and fraudulent user behaviors within a financial environment. It also includes scripts for clustering analysis and feature evaluation using machine learning techniques.

## Table of Contents
1. Introduction
2. System Architecture
3. Markov Chain Modeling
4. Agent Activities
5. Fraudulent and Legitimate Behaviors
6. Transaction Amount Distributions
7. Simulation Workflow
8. Clustering Analysis (clustering.py)
9. Feature Evaluation (analyze_sample.py)
10. Project Structure
11. Running the Simulation
12. Output and Data Interpretation
13. Customization and Extensions

## 1. Introduction
This system simulates banking activities using Markov chains, allowing researchers and developers to generate realistic datasets for testing fraud detection models. The simulation replicates patterns of everyday users and fraudsters, with detailed transaction logs saved as CSV files.

## 2. System Architecture
The system is modular, consisting of:
- **simulator.py**: Main script that runs the simulation.
- **activity.py**: Defines possible user activities.
- **agent.py**: Represents legitimate users and fraudsters.
- **bank.py**: Manages transaction logging.
- **catalog.py**: Stores behavior definitions and transition matrices.
- **distributions.py**: Defines transaction amount distributions.
- **clustering.py**: Analyzes clusters emerging from the dataset using DBSCAN, KMeans, Hierarchical Clustering, and Isolation Forest.
- **analyze_sample.py**: Evaluates feature importance and correlations using Random Forest and other statistical methods.

## 3. Markov Chain Modeling
User behaviors are represented as states in a Markov chain, with transition matrices determining activity probabilities. Legitimate users have slower transitions, while fraudsters perform actions more rapidly.

## 4. Agent Activities
Agents are classified as legitimate users or fraudsters. Each agent has:
- **Real ID**: Unique identifier.
- **Virtual ID**: Used in identity theft cases.
- **Balance**: Updated after each transaction.
- **Timestamp**: Tracks activity timing.

## 5. Fraudulent and Legitimate Behaviors
### Legitimate
- Activities include deposits, purchases, bill payments, and account reviews.
- Time intervals are longer and more predictable.

### Fraudulent
- **Identity Theft:** Unauthorized logins and fund transfers.
- **Card Skimming:** Rapid ATM withdrawals and purchases.
- **Money Laundering:** Wire transfers and asset investments.
- **Synthetic Identity Fraud:** Creating fake identities and maxing out credit.

## 6. Transaction Amount Distributions
- Legitimate amounts are smaller and follow a log-normal distribution.
- Fraudulent amounts are larger, reflecting financial exploitation.
- Negative amounts represent expenses and withdrawals.

## 7. Simulation Workflow
1. **Activity Selection:** Determined by the Markov chain.
2. **Transaction Execution:** Amounts are drawn from appropriate distributions.
3. **Activity Logging:** Each transaction is recorded with timestamps and fraud labels.
4. **Account Closure:** Agents are removed after closing accounts.

## 8. Clustering Analysis (clustering.py)
This script analyzes the generated dataset using clustering algorithms such as:
- **DBSCAN**: Density-based clustering
- **KMeans**: Partition-based clustering
- **Hierarchical Clustering**: Agglomerative clustering with dendrogram plots
- **Isolation Forest**: Anomaly detection

Results are visualized using PCA-reduced 2D scatter plots, and plots are saved in the `clustering_results` directory.

### Command to run:
```bash
python src/clustering.py --input_file data/fraud_simulation_activities.csv
```

## 9. Feature Evaluation (analyze_sample.py)
This script evaluates the importance and correlation of features using:
- **Random Forest Classifier:** Measures feature importance and predicts fraud labels
- **Decision Tree:** Visualizes decision rules
- **Correlation Analysis:** Uses Pearson correlation and Cramér's V for mixed data types
- **Heatmaps:** Visualizes feature correlations

Results are saved in the `plots` directory.

### Command to run:
```bash
python src/analyze_sample.py --input_file data/fraud_simulation_activities.csv
```

## 10. Project Structure
```
project_root/
├── src/
│   ├── simulator.py
│   ├── activity.py
│   ├── agent.py
│   ├── bank.py
│   ├── catalog.py
│   ├── distributions.py
│   ├── clustering.py
│   ├── analyze_sample.py
└── data/
    └── Output files are saved here
```

## 11. Running the Simulation
To run the simulation, use the command:
```bash
python src/simulator.py --nb_activities 1000000 --min_n_agents 40 --fraudster_rate 0.1 --data_folder data --start_time "2025-01-06T12:00:00"
```

Arguments:
- `--nb_activities`: Total number of activities to generate.
- `--min_n_agents`: Minimum number of agents per batch.
- `--fraudster_rate`: Probability of an agent being a fraudster.
- `--data_folder`: Output directory.
- `--start_time`: Initial timestamp.

## 12. Output and Data Interpretation
Output files are saved as CSVs, with each row representing an activity:
- **`agent_id`**: Unique agent identifier.
- **`timestamp`**: Activity time.
- **`behavior`**: Legitimate or fraudulent.
- **`activity_type`**: Type of activity performed.
- **`amount`**: Transaction amount (positive or negative).
- **`balance`**: Agent’s balance after the activity.
- **`fraudulent`**: Binary indicator of fraud.

## 13. Customization and Extensions
- **Adding New Behaviors:** Define new behaviors in `catalog.py`.
- **Modifying Transaction Amounts:** Adjust log-normal distributions in `distributions.py`.
- **Changing Agent Logic:** Update `assign_behavior()` and `run_simulation_step()` in `simulator.py`.

This system is designed for flexibility and scalability, supporting various fraud scenarios for both academic research and industry applications.

