import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import streamlit as st

@st.cache_data(ttl=86400)
def build_feature_matrix(indicator_dfs, year_mode="latest", target_year=2022, missing_strategy="drop"):
    """
    indicator_dfs: list of DataFrames from get_indicator
    year_mode: 'latest' or 'specific'
    target_year: int/str if 'specific'
    missing_strategy: 'drop', 'median', 'knn'
    """
    if not indicator_dfs:
        return pd.DataFrame()
        
    # Drop any empty dataframes from the list before concat
    indicator_dfs = [df for df in indicator_dfs if not df.empty]
    if not indicator_dfs:
        return pd.DataFrame()
        
    combined = pd.concat(indicator_dfs, ignore_index=True)
    
    if year_mode == 'specific':
        df_year = combined[combined['date'] == str(target_year)].copy()
    else:  # latest
        # Sort by date descending so latest is first
        combined_valid = combined.dropna(subset=['value'])
        combined_valid = combined_valid.sort_values(by='date', ascending=False)
        df_year = combined_valid.drop_duplicates(subset=['country_iso3c', 'indicator'], keep='first').copy()

    if df_year.empty:
        return pd.DataFrame()

    # Pivot table so rows are countries, columns are indicators
    pivot_df = df_year.pivot(index='country_iso3c', columns='indicator', values='value')
    
    # Add country names back
    country_names = combined.drop_duplicates('country_iso3c').set_index('country_iso3c')['country']
    pivot_df = pivot_df.join(country_names)
    
    # Missing value handling
    numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
    
    if missing_strategy == "drop":
        pivot_df = pivot_df.dropna(subset=numeric_cols)
    elif missing_strategy == "median":
        if not pivot_df.empty and len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            pivot_df[numeric_cols] = imputer.fit_transform(pivot_df[numeric_cols])
    elif missing_strategy == "knn":
        if not pivot_df.empty and len(numeric_cols) > 0:
            n_samples = len(pivot_df)
            imputer = KNNImputer(n_neighbors=min(5, n_samples))
            pivot_df[numeric_cols] = imputer.fit_transform(pivot_df[numeric_cols])
            
    return pivot_df.reset_index()
