import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz 
from sklearn.metrics import classification_report, accuracy_score
from IPython import embed
import matplotlib.pyplot as plt
import os
import time
import scipy.stats as stats
import seaborn as sns


def cramers_v(x, y):
    """Compute Cramér's V statistic for categorical-categorical correlation."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

def correlation_matrix(data, categorical_columns, numeric_columns):
    """Compute correlation matrix for numerical and categorical features."""
    corr_matrix = pd.DataFrame(index=data.columns, columns=data.columns)

    for col1 in data.columns:
        for col2 in data.columns:
            if col1 in numeric_columns and col2 in numeric_columns:
                # Pearson correlation for numerical features
                corr_matrix.loc[col1, col2] = data[col1].corr(data[col2])
            elif col1 in categorical_columns and col2 in categorical_columns:
                # Cramér's V for categorical features
                corr_matrix.loc[col1, col2] = cramers_v(data[col1], data[col2])
            elif col1 in categorical_columns and col2 in numeric_columns:
                # ANOVA F-test correlation for categorical vs numerical
                f_val, p_val = stats.f_oneway(*(data[data[col1] == cat][col2] for cat in data[col1].unique()))
                corr_matrix.loc[col1, col2] = f_val
            else:
                corr_matrix.loc[col1, col2] = None  # Avoid duplicate calculations
    
    return corr_matrix.astype(float)


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
    data['delta_time'] = pd.to_timedelta(data['delta_time']).dt.total_seconds()/60 #time interval between sequence from the same agent converted into time difference in minutes

    # Drop columns which are not optimized for the simulation yet
    data.drop(columns=['device', 'network', 'compromised_device', 'compromised_network',], inplace=True)

    # Encode categorical features
    label_encoders = {}
    categorical_columns = ['behavior', 'activity_type', 'initial_country', 'location', 'agent_type']
    numeric_cols = ['initial_balance', 'amount', 'balance', 'year', 'month', 'day', 'hour', 'minute', 'second', 'delta_time']

    #for col in categorical_columns:
    #    le = LabelEncoder()
    #    data[col] = le.fit_transform(data[col])
    #    label_encoders[col] = le
    # One-Hot Encoding for categorical columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)  # drop_first=True to avoid dummy variable trap   
    # Features and target selection
    target_col = 'is_fraud'  
    X = data.drop(columns=[col for col in data.columns if col.startswith('behavior')] + ['real_id', 'is_fraud'])
    y = data[target_col]

    print(f"Dataset columns: {X.columns}")
    print(f"Label: {target_col}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # Compute the correlation matrix
    #all_correlation_matrix = correlation_matrix(X_train, categorical_columns, numeric_cols)
    # Plot the correlation heatmap
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(all_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    #plt.title("Mixed Feature Correlation Heatmap (Numerical + Categorical)")
    #plt.show()
    os.makedirs('plots', exist_ok=True)

    pos=0
    flag='is_fraud'
    plt.figure(figsize=(24,25))
    for i, col in enumerate(numeric_cols):
        plt.subplot(4, 3 , pos + 1)
        plt.hist(X_train[col][y_train==0], density = True, bins=60, label = f"{flag} = 0",color='b', alpha=0.5, )
        plt.hist(X_train[col][y_train==1], density = True, bins=60, label = f"{flag} = 1",color='r', alpha=0.5, )
        plt.xlabel(col)
        plt.legend()
        pos+=1
    plt.tight_layout()
    plt.savefig(f"plots/numerical_variables.pdf")

    plt.figure(figsize=(50, 50))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"plots/feature_correlation.png")


    # Identify all behavior-related columns
    behavior_cols = [col for col in data.columns if col.startswith('behavior_')]

    # Define subplot grid size
    num_behaviors = len(behavior_cols)
    rows = int(np.ceil(num_behaviors / 3))  # Adjust for up to 3 plots per row
    cols = min(num_behaviors, 2)  # At most 3 columns per row

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20 * rows))  # Adjust height dynamically
    axes = axes.flatten()  # Flatten axes for easier indexing

    # Loop through each behavior column and plot its heatmap
    for i, behavior in enumerate(behavior_cols):
        # Compute correlation matrix for the specific behavior column
        behavior_corr = data.corr()[[behavior]].sort_values(by=behavior, ascending=False)

        # Plot heatmap in subplot
        sns.heatmap(behavior_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=axes[i])

        # Set title
        axes[i].set_title(f"Correlation Heatmap for {behavior}")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("plots/behavior_correlation_heatmaps.pdf")  # Save the heatmaps

   
    # Normalize numerical columns
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    
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
        'Feature': X_train.columns,
        'Importance': importances
    })

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Print the feature importances
    print(feature_importance_df)

    # Plotting the feature importances
    plt.figure(figsize=(20, 12))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance for predicting label {target_col}')
    plt.savefig(f"plots/feature_importance.png")

    '''
    # Compute the correlation matrix
    all_correlation_matrix = correlation_matrix(X_train, categorical_columns, numeric_cols)

    # Plot the correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(all_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Mixed Feature Correlation Heatmap (Numerical + Categorical)")
    plt.show()
    '''


    # Try decision tree
    features = X_train.columns
    start_fit = time.time()
    clf = tree.DecisionTreeClassifier(max_depth = 4,class_weight='balanced', ) #min_impurity_decrease=0.009
    clf.fit(X_train, y_train)
    print(f'Decision Tree training required: {round(time.time()-start_fit, 2)}s')
    # Save to a .pkl file
    #with open(f"{output_path}/decision_tree_model.pkl", "wb") as f:
     #   pickle.dump(clf, f)
    # Visualize the decision tree
    #dot_data = tree.export_graphviz(clf,feature_names=features,class_names=list(particle_type.keys()),filled=True, rounded=True,special_characters=True) 
    dot_data = tree.export_graphviz(clf,feature_names=features,filled=True, rounded=True, special_characters=True, proportion=True) 
    graph = graphviz.Source(dot_data) 
    graph.render(f"plots/treeSchema")
    print("Model saved successfully!")
