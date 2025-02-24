import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


from IPython import embed
import os


def plot_hierarchical_dendrogram(X, df, output_dir="clustering_results"):
    # Generate linkage matrix
    linkage_matrix = linkage(X, method='ward')

    # Create dendrogram
    plt.figure(figsize=(20, 8))
    dendro = dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90, leaf_font_size=8, )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Transaction Index")
    plt.ylabel("Distance")
    plt.xticks(rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Save the dendrogram plot
    plt.savefig(f"{output_dir}/hierarchical_dendrogram.png", dpi=300)
    plt.show()

    return dendro

def apply_clustering(df, numerical_features, categorical_features, output_dir="clustering_results"):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)  # Ensure dense output
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Apply the transformations to the dataframe
    X = pipeline.fit_transform(df.drop(columns=['is_fraud', 'behavior']))  # Transformed data is now dense

    # 1. DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    # 2. KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Choose n_clusters based on your dataset
    kmeans_labels = kmeans.fit_predict(X)

    # 3. Hierarchical Clustering
    linkage_matrix = linkage(X, method='ward')  
    agg_clustering = AgglomerativeClustering(n_clusters=3)  
    hierarchical_labels = agg_clustering.fit_predict(X)

    # Isolation Forest for Anomaly Detection
    isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    anomaly_scores = isolation_forest.fit_predict(X)

    # Reduce dimensionality for visualization purposes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    '''    
    # --- DBSCAN Visualization ---
    plt.figure(figsize=(12, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', edgecolors='k')
    plt.title("DBSCAN Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(os.path.join(output_dir, "dbscan_clustering.png"))  # Save plot
    plt.close()

    # --- KMeans Visualization ---
    plt.figure(figsize=(12, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')
    plt.title("KMeans Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(os.path.join(output_dir, "kmeans_clustering.png"))  # Save plot
    plt.close()

    # --- Hierarchical Clustering Visualization ---
    plt.figure(figsize=(12, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', edgecolors='k')
    plt.title("Hierarchical Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(os.path.join(output_dir, "hierarchical_clustering.png"))  # Save plot
    plt.close()

    # --- Dendrogram ---
    plot_hierarchical_dendrogram(X, df)

    # Isolation Forest
    plt.figure(figsize=(12, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=anomaly_scores, cmap='viridis', edgecolors='k')
    plt.title("Isolation Forest Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.savefig(os.path.join(output_dir, "isolation_forest_clustering.png"))  # Save plot
    plt.close()
    '''
    # Visualization
    for labels, title in zip([
        dbscan_labels, kmeans_labels, hierarchical_labels, anomaly_scores
    ], ["dbscan", "kmeans", "hierarchical", "isolation_forest"]):
        plt.figure(figsize=(12, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.title(f"{title} Clustering")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.savefig(f"{output_dir}/{title}_clustering.png")
        plt.close()
    

    print(f"Plots saved in: {output_dir}")

    return dbscan_labels, kmeans_labels, hierarchical_labels, anomaly_scores


if __name__ == "__main__":
    # Set up argument parser to get the input file path
    parser = argparse.ArgumentParser(
        description='Script for running clustering algorithms on the generated dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input_file', help='CSV Input file for producing the dashboard', type=str)
    cfg = parser.parse_args()

    df = pd.read_csv(f'{cfg.input_file}')  # Load your dataframe

    print(f"Dataset columns: {df.columns}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Step 2: Extract datetime components
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    #df['day_of_week'] = df['timestamp'].dt.dayofweek
    #df['is_weekend'] = df['timestamp'].dt.weekday >= 5  # 1 for weekends, 0 for weekdays

    # Drop the original timestamp column
    df.drop(columns=['timestamp'], inplace=True)

    # Apply clustering algorithms
    # Identify numerical and categorical columns
    numerical_features = ['initial_balance', 'amount', 'balance', 'year', 'month', 'day', 'hour', 'minute', 'second']
    categorical_features = ['activity_type', 'initial_country', 'device', 'network']
    os.makedirs('clustering_results', exist_ok=True)
    output_dir = f"clustering_results/data_{len(df)}"
    os.makedirs(output_dir, exist_ok=True)

    dbscan_labels, kmeans_labels, hierarchical_labels, isolation_forest_labels = apply_clustering(df=df, categorical_features=categorical_features, numerical_features=numerical_features, output_dir=output_dir)
    
    # Optionally, add the labels back to the dataframe
    df['dbscan_label'] = dbscan_labels
    df['kmeans_label'] = kmeans_labels
    df['hierarchical_label'] = hierarchical_labels
    df['isolation_forest_label'] = isolation_forest_labels
    
    # Display the dataframe with labels
    print(df.head())
