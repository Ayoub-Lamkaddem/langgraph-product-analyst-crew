from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .state import DataState

def detect_patterns(state: DataState) -> DataState:
    df = state["processed_df"].copy()
    
    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("Aucune colonne numérique pour détecter les patterns.")
        state["processed_df"] = df
        return state
    
    features = df[numeric_cols]

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    state["pca_df"] = df.copy()
    return state
