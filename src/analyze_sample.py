import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from IPython import embed
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for generating the dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_file', help='CSV Input file for producing the dashboard', type=str,)
    cfg = parser.parse_args()

    data = pd.read_csv(cfg.input_file)

    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Extract datetime components
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['second'] = data['timestamp'].dt.second

    # Drop the original timestamp column
    data.drop(columns=['timestamp'], inplace=True)

    # Drop columns which are not optimized for the simulation yet
    data.drop(columns=['device', 'network', 'compromised_device', 'compromised_network'], inplace=True)

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['behavior', 'activity_type', 'initial_country', 'location', 'agent_type']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Features and target selection
    target_col = 'is_fraudster'  
    X = data.drop(columns=['real_id', 'behavior', 'is_fraudster'])
    y = data[target_col]

    print(f"Dataset columns: {X.columns}")
    print(f"Label: {target_col}")

    # Normalize numerical columns
    scaler = StandardScaler()
    numeric_cols = ['initial_balance', 'amount', 'balance', 'year', 'month', 'day', 'hour', 'minute', 'second']
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Evaluate model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Get the feature importances from the trained classifier
importances = clf.feature_importances_

# Create a DataFrame to view the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(feature_importance_df)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title(f'Feature Importance for predicting label {target_col}')
#os.make
#plt.savefig(f"{output_dir}
plt.show()

