import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=86400)
def get_countries():
    """Fetches list of countries from World Bank API, filtering out aggregate regions."""
    url = "https://api.worldbank.org/v2/country"
    params = {"format": "json", "per_page": 300}
    response = requests.get(url, params=params).json()
    
    if len(response) > 1:
        data = response[1]
        # Filter out aggregates (their region typically has value 'Aggregates')
        countries = [c for c in data if c.get('region', {}).get('value') != 'Aggregates']
        df = pd.DataFrame(countries)
        
        # Unpack nested dicts like Region and Income Level
        if not df.empty:
            df['region_name'] = df['region'].apply(lambda x: x['value'] if isinstance(x, dict) else x)
            df['income_level'] = df['incomeLevel'].apply(lambda x: x['value'] if isinstance(x, dict) else x)
            return df[['id', 'iso2Code', 'name', 'region_name', 'income_level', 'longitude', 'latitude']]
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_indicator(indicator, date_range="2010:2024"):
    """
    Fetches indicator timeseries for all countries. 
    `indicator` string, e.g. "SP.POP.TOTL"
    `date_range` string, e.g. "2015:2024" or "2022"
    """
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    all_data = []
    page = 1
    
    while True:
        params = {
            "format": "json",
            "date": date_range,
            "per_page": 2000,
            "page": page
        }
        res = requests.get(url, params=params).json()
        
        if len(res) < 2:
            break
            
        page_info = res[0]
        data = res[1]
        
        if not data:
            break
            
        all_data.extend(data)
        
        if page >= page_info.get('pages', 1):
            break
        page += 1
        
    if all_data:
        df = pd.DataFrame(all_data)
        if df.empty:
            return pd.DataFrame()
            
        df['country_iso3c'] = df['countryiso3code']
        df['country'] = df['country'].apply(lambda x: x['value'] if isinstance(x, dict) else x)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['indicator'] = indicator
        
        # Filter out aggregates which often have empty ISO3 codes or are not proper countries
        # A simple check is that actual countries have non-empty countryiso3code 
        # and regions typically don't have regions assigned to them if we look up the countries list.
        # But we can also just use the valid list from get_countries()
        valid_countries = get_countries()
        if not valid_countries.empty:
            df = df[df['country_iso3c'].isin(valid_countries['id'])]
            
        return df[['country_iso3c', 'country', 'date', 'value', 'indicator']]
        
    return pd.DataFrame()
