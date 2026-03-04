import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import streamlit as st

@st.cache_data
def preprocess_data(df, apply_log=False):
    """
    df: DataFrame from build_feature_matrix.
    Returns scaled dataframe and the list of metadata columns.
    """
    meta_cols = [c for c in df.columns if c in ['country_iso3c', 'country']]
    numeric_cols = [c for c in df.columns if c not in meta_cols]
    
    numeric_df = df[numeric_cols].copy()
    
    if apply_log:
        # Apply log1p on values >= 0, keep others as is or shift (simplest is log1p of non-negative)
        numeric_df = numeric_df.applymap(lambda x: np.log1p(x) if x > 0 else 0)
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df.index)
    
    return scaled_df, meta_cols

@st.cache_data
def run_clustering(scaled_df, algorithm="kmeans", n_clusters=4):
    """
    Run K-Means or Hierarchical clustering on the scaled data.
    """
    if scaled_df.empty:
        return []
        
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    elif algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")
        
    labels = model.fit_predict(scaled_df)
    return labels

@st.cache_data
def run_pca(scaled_df, n_components=2):
    """
    Run PCA for 2D visualization of the clusters.
    """
    if scaled_df.empty or scaled_df.shape[1] < 2:
        return pd.DataFrame()
        
    # If fewer columns than components, adjust
    n_comp = min(n_components, scaled_df.shape[1])
    
    pca = PCA(n_components=n_comp, random_state=42)
    pca_result = pca.fit_transform(scaled_df)
    
    cols = [f"PCA{i+1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(pca_result, columns=cols, index=scaled_df.index)
    
    # If we only got 1 component due to very few features, pad with 0 for 2D plotting
    if n_comp == 1 and n_components == 2:
        pca_df['PCA2'] = 0.0
        
    return pca_df
