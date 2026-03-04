import plotly.express as px
import pandas as pd

def plot_pca_scatter(pca_df, clusters, original_df):
    """
    pca_df: DataFrame with PCA1, PCA2 and index.
    clusters: array of labels.
    original_df: The unscaled dataframe with country names and features for hover.
    """
    if pca_df.empty or pca_df.shape[1] < 2:
        return px.scatter(title="Not enough components for PCA")
        
    plot_df = pca_df.copy()
    plot_df['Cluster'] = [f'Cluster {c}' for c in clusters]
    plot_df['Country'] = original_df['country']
    
    numeric_cols = original_df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plot_df[col] = original_df[col]
        
    fig = px.scatter(
        plot_df, x='PCA1', y='PCA2', 
        color='Cluster', 
        hover_name='Country',
        hover_data=list(numeric_cols),
        title="2D Cluster Visualization (PCA)",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    return fig

def plot_choropleth(df, clusters):
    """
    df: original_df which has country_iso3c.
    """
    if df.empty or 'country_iso3c' not in df.columns:
        return px.choropleth()
        
    plot_df = df.copy()
    plot_df['Cluster'] = [f'Cluster {c}' for c in clusters]
    
    fig = px.choropleth(
        plot_df,
        locations='country_iso3c',
        locationmode='ISO-3',
        color='Cluster',
        hover_name='country',
        title="World Map of Clusters",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def get_cluster_summary(df, clusters):
    summary_df = df.copy()
    summary_df['Cluster'] = [f'Cluster {c}' for c in clusters]
    numeric_cols = summary_df.select_dtypes(include='number').columns
    
    summary = summary_df.groupby('Cluster')[numeric_cols].mean().reset_index()
    counts = summary_df.groupby('Cluster').size().reset_index(name='Count')
    summary = pd.merge(summary, counts, on='Cluster')
    
    cols = ['Cluster', 'Count'] + list(numeric_cols)
    return summary[cols]
