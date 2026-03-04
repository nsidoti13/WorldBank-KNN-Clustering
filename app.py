import streamlit as st
import pandas as pd
import plotly.express as px

# Import custom modules
from wb_client import get_countries, get_indicator
from features import build_feature_matrix
from models import preprocess_data, run_clustering, run_pca
from viz import plot_pca_scatter, plot_choropleth, get_cluster_summary

# Set page config
st.set_page_config(page_title="World Bank Cluster Dashboard", layout="wide")

# Curated indicators
INDICATORS = {
    "GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD",
    "Life expectancy at birth, total (years)": "SP.DYN.LE00.IN",
    "CO2 emissions (metric tons per capita)": "EN.ATM.CO2E.PC",
    "School enrollment, primary (% gross)": "SE.PRM.ENRR",
    "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
    "Unemployment, total (% of total labor force)": "SL.UEM.TOTL.ZS",
    "Infant mortality rate (per 1,000 live births)": "SP.DYN.IMRT.IN"
}

st.title("World Bank Country Clustering")

with st.sidebar:
    st.header("1. Data Selection")
    
    # Load country metadata
    with st.spinner("Loading country metadata..."):
        countries_df = get_countries()
    
    selected_ind_names = st.multiselect(
        "Select Indicators", 
        options=list(INDICATORS.keys()),
        default=list(INDICATORS.keys())[:3] # Default 3
    )
    
    year_mode = st.radio("Year Mode", ["latest", "specific"])
    target_year = 2022
    if year_mode == "specific":
        target_year = st.slider("Select Year", 1990, 2024, 2022)
    else:
        # Give a slider to decide what 'latest' means roughly, 
        # or anchor it near 2024
        target_year = 2024
        
    st.header("2. Preprocessing")
    missing_strategy = st.selectbox("Missing Data Strategy", ["drop", "median", "knn"])
    apply_log = st.checkbox("Apply Log Transform (log1p)", value=False)
    
    st.header("3. Clustering")
    algo = st.selectbox("Algorithm", ["kmeans", "hierarchical"])
    k = st.slider("Number of Clusters (K)", 2, 10, 4)

if not selected_ind_names:
    st.warning("Please select at least one indicator from the sidebar.")
    st.stop()
    
selected_indicator_codes = [INDICATORS[name] for name in selected_ind_names]

# --- 1) Fetch Data ---
with st.spinner("Fetching indicator data from World Bank..."):
    # date_range helps reduce data size.
    # For latest, fetching last 10 years increases chance of finding data.
    date_range = f"{target_year}" if year_mode == 'specific' else f"{target_year-10}:{target_year}"
    
    dfs = []
    for code in selected_indicator_codes:
        df = get_indicator(code, date_range)
        dfs.append(df)

# --- 2) Build Matrix ---
with st.spinner("Building feature matrix..."):
    feature_matrix = build_feature_matrix(dfs, year_mode, target_year, missing_strategy)

if feature_matrix.empty:
    st.error("No data available for the selected indicators and year. Try expanding the criteria or changing the missing strategy.")
    st.stop()

# Print KPIs
st.subheader("Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Countries Included", feature_matrix['country'].nunique())
col2.metric("Indicators Used", len(selected_indicator_codes))

total_countries = len(countries_df) if not countries_df.empty else 217
percent_retained = min(feature_matrix['country'].nunique() / total_countries * 100, 100.0)
col3.metric("Coverage (% of world)", f"{percent_retained:.1f}%")

# --- 3) Preprocessing & Models ---
with st.spinner("Running clustering and dimensionality reduction..."):
    scaled_df, meta_cols = preprocess_data(feature_matrix, apply_log)
    
    clusters = run_clustering(scaled_df, algorithm=algo, n_clusters=k)
    
    pca_df = run_pca(scaled_df, n_components=2)

# --- 4) Visualization ---
st.subheader("Cluster Visualizations")
tab1, tab2, tab3 = st.tabs(["🌎 World Map", "📊 PCA Scatter", "📋 Summary Table"])

with tab1:
    fig_map = plot_choropleth(feature_matrix, clusters)
    st.plotly_chart(fig_map, use_container_width=True)
    
with tab2:
    fig_pca = plot_pca_scatter(pca_df, clusters, feature_matrix)
    st.plotly_chart(fig_pca, use_container_width=True)
    
with tab3:
    summary_df = get_cluster_summary(feature_matrix, clusters)
    st.dataframe(summary_df, use_container_width=True)

# --- 5) Drill down ---
st.markdown("---")
st.subheader("Country Drill-down")
search_country = st.selectbox("Select a country to view details", feature_matrix['country'].sort_values().unique())
if search_country:
    country_data = feature_matrix[feature_matrix['country'] == search_country].iloc[0]
    country_cluster = clusters[feature_matrix['country'] == search_country][0]
    st.write(f"**Cluster Assignment:** Cluster {country_cluster}")
    st.write(country_data.drop(['country_iso3c', 'country']))
    
st.markdown("---")
st.markdown("Data Source: [World Bank Open Data](https://data.worldbank.org/) | Indicators licensed under CC BY 4.0")
