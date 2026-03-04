# World Bank Indicator Clustering

This repository contains an interactive Streamlit dashboard that allows users to cluster countries based on various World Bank indicators.

## Goal
To provide a Streamlit dashboard that:
* Lets a user choose a set of World Bank indicators (e.g., GDP per capita, life expectancy).
* Chooses a year or "latest available" mapping.
* Clusters countries based on those features (K-Means, Hierarchical).
* Shows the clusters in a PCA 2D scatterplot, on a world map, and in a data table.

## Architecture

This application strictly follows a component-based model as per the Implementation Plan:
1. `app.py`: The Streamlit web application.
2. `wb_client.py`: The API request handling and pagination for World Bank endpoints.
3. `features.py`: Aggregation of timeseries into a clean ML-ready feature matrix including handling missing data limits.
4. `models.py`: Implementations for StandarScaler, KNN/Simple missing value handling, KMeans, Agglomerative clustering, and PCA.
5. `viz.py`: Handling Plotly visualizations (scatterplots, choropleths, summary tables).

## Usage

### 1) Environment Setup

Initialize a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the App

Start the Streamlit application:
```bash
streamlit run app.py
```

### 3) Navigate Dashboard

In the browser:
- Modify Data Selection to curate explicit indicators from World Bank.
- Fine-tune Preprocessing handling to dictate log transforms and how empty country data is imputed/dropped.
- Alter the Clustering Algorithm parameters dynamically to identify structural trends.

## Data Notice

Data originates from the World Bank API (v2). Use of Indicators is subject to licensing restrictions denoted by World Bank (commonly CC BY 4.0).
