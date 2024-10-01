import random
import pandas as pd

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from agent import LegitimateCustomer, Fraudster
from catalog import behavioral_catalog

class BankWithClientActivities:
    def __init__(self):
        self.transaction_log = pd.DataFrame(columns=["agent_id", "type", "amount", "balance", "timestamp", "fraud", "risk_level"])
        self.current_time = datetime.now()

    def process_transaction(self, transaction):
        transaction["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transaction_log = pd.concat([self.transaction_log, pd.DataFrame([transaction])], ignore_index=True)

    def process_activity(self, activity):
        activity["timestamp"] = self.current_time
        self.current_time += timedelta(minutes=random.randint(1, 30))
        self.transaction_log = pd.concat([self.transaction_log, pd.DataFrame([activity])], ignore_index=True)

def run_simulation_with_activities(legitimate_agents, fraudster_agents, bank, steps=100):
    for step in range(steps):
        if step % 100 == 0:
            print(f"step: {step}")
        for agent in legitimate_agents:
            transaction = agent.generate_transaction()
            bank.process_transaction(transaction)
            if random.random() < 0.2:
                activity = agent.account_activity()
                bank.process_activity(activity)

        for fraudster in fraudster_agents:
            fraud_type = random.choice(list(behavioral_catalog.keys()))
            fraud = fraudster.commit_fraud(fraud_type)
            bank.process_transaction(fraud)
            if random.random() < 0.5:
                activity = fraudster.account_activity(fraud_type)
                bank.process_activity(activity)

def extract_features(transaction_log):
    X = transaction_log[["amount", "balance", "risk_level"]]
    y = transaction_log["fraud"]
    return X, y

def benchmark_models(transaction_log):
    X, y = extract_features(transaction_log)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    models = {
        "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True),
        "One-Class SVM": OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    }
    
    results = {}
    for model_name, model in models.items():
        if model_name == "Local Outlier Factor":
            model.fit(X_train)
            y_pred_test = model.predict(X_test)
        else:
            model.fit(X_train)
            y_pred_test = model.predict(X_test)
        y_pred_test = np.where(y_pred_test == -1, 1, 0)
        results[model_name] = {
            "F1 Score": f1_score(y_test, y_pred_test),
            "Accuracy": accuracy_score(y_test, y_pred_test),
            "Classification Report": classification_report(y_test, y_pred_test)
        }
    return results

if __name__ == "__main__":
    legitimate_agents = [LegitimateCustomer(agent_id=i, balance=random.uniform(1000, 5000)) for i in range(100)]
    fraudster_agents = [Fraudster(agent_id=i+100, balance=random.uniform(1000, 5000)) for i in range(3)]
    bank_with_activities = BankWithClientActivities()
    run_simulation_with_activities(legitimate_agents, fraudster_agents, bank_with_activities, steps=1000)
    
    transaction_log_with_fraud_features = bank_with_activities.transaction_log
    transaction_log_with_fraud_features.to_csv("data/fraud_simulation_1000_activities.csv", index=False)

    # Benchmark anomaly detection models
    results = benchmark_models(transaction_log_with_fraud_features)
    for model, metrics in results.items():
        print(f"Model: {model}")
        print(f"F1 Score: {metrics['F1 Score']}")
        print(f"Accuracy: {metrics['Accuracy']}")
        print(f"Classification Report: \n{metrics['Classification Report']}")
