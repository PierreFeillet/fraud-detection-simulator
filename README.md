# Bank Fraud Simulation System Using Markov Chains

This README provides a comprehensive overview of the bank fraud simulation system using Markov chains, designed to model both legitimate and fraudulent user behaviors within a financial environment. It also includes scripts for clustering analysis and feature evaluation using machine learning techniques.

## Table of Contents
1. Introduction
2. System Architecture
3. Markov Chain Approach
4. Agent and Activity Definitions
5. Behavior Catalog
6. Location Modeling
7. Transaction Amount Distributions
8. Simulation Workflow
9. Evaluation and Goodness Check
10. Limitations and Advantages of the Markov Chain Approach
11. Project Structure
12. Running the Simulation
13. Output and Data Interpretation
14. Customization and Extensions

## 1. Introduction
This system simulates banking activities using Markov chains, allowing researchers and developers to generate realistic datasets for testing fraud detection models. The simulation replicates patterns of everyday users and fraudsters, with detailed transaction logs saved as CSV files.

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
The Markov chain approach models the probability of a sequence of activities using a transition matrix. Each activity is associated in the `catalog.py` to a probability array that determines the likelihood of transitioning to other activities. Currently, these probabilities are constant, but future implementations may incorporate variables such as location and transaction amount.

Future enhancements could include tailored distributions for specific profiles, such as High-Spender, Student, and Investor for legitimate users, and distinct patterns for each type of fraud.

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
- **Compromised Status:** Simulates whether the device or network is compromised, with planned improvements to model patterns rather than using random assignments.
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

Future improvements could introduce location assignment based on a probability distribution centered around a defined barycenter, decreasing the likelihood of selecting locations as the distance from the barycenter increases.

## 7. Transaction Amount Distributions

Transaction amounts follow a log-normal distribution defined by parameters \(\mu\) and \(\sigma\):
- **Legitimate behavior:** \(\mu = 5\), \(\sigma = 1\), with a peak around $150.
- **Fraudulent behavior:** \(\mu = 8\), \(\sigma = 1.2\), with a peak around $3000.
Future versions could include specific log-normal distributions for different legitimate profiles and fraud scenarios.

## Balance computation
In catalog.py, each activity is labeled as "neutral", "positive", or "negative", referring to the impact on the balance. For example, a "Purchase" results in a "negative" impact, while a "Login" is "neutral" and has no effect on the balance (the `amount` associated to that activyt is 0 and the balance is not changed). The initial balance is randomly assigned, and subsequent balances are updated based on transaction amounts. If a transaction amount exceeds the current balance, the transaction is declined, and the granted field is set to False; otherwise, it is set to True.

The remaining sections follow the previously detailed structure, ensuring clarity and modularity for users looking to customize and extend the simulation.


## 8. Simulation Workflow
1. **Activity Selection:** Determined using Markov chain transition probabilities.
2. **Transaction Execution:** Amounts are drawn from the appropriate distributions.
3. **Location Assignment:** Based on agent type, visited countries, and fraud behavior.
4. **Balance Validation:** Transactions are approved if the balance is sufficient. 
5. **Logging:** Each transaction is recorded with timestamps, amounts, and labels.
6. **Account Closure:** Agents are removed after performing the “Close Account” activity.

## 9. Evaluation and Goodness Check
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

## 10. Limitations and Advantages of the Markov Chain Approach
### Limitations
- Large and sparse transition matrices are required for modeling complex sequences.
- Transition probabilities should ideally depend on variables like amount, location, and time.
- Realistic modeling requires domain expertise in banking.

### Advantages
- Simple to control and interpret.
- Transparent and easily explainable modeling process.

## 11. Project Structure
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

## 12. Running the Simulation
To run the simulation, use the command:
```bash
python src/simulator.py --nb_activities 1000000 --min_n_agents 40 --fraudster_rate 0.1 --data_folder data --start_time "2025-01-06T12:00:00"
```

## 13. Output and Data Interpretation
Simulation results are saved as CSV files, with each row representing an activity:
- **`agent_id`**: Unique identifier of the agent.
- **`timestamp`**: Activity execution time.
- **`behavior`**: Type of behavior (legitimate or fraudulent).
- **`activity_type`**: Specific activity performed.
- **`amount`**: Transaction amount, positive or negative.
- **`balance`**: Agent’s balance after the activity.
- **`fraudulent`**: Binary flag indicating fraud.

## 14. Customization and Extensions
- **Adding New Behaviors:** Define additional behaviors in `catalog.py`.
- **Modifying Transaction Amounts:** Adjust log-normal parameters in `distributions.py`.
- **Changing Agent Logic:** Customize `assign_behavior()` and `run_simulation_step()` in `simulator.py`.

This system is designed to be flexible, allowing users to simulate various fraud scenarios for research and industry applications.

