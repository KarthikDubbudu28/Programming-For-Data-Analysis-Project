import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set page configuration (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Temperature Predictior",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("city_day.csv")
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['month'] = pd.DatetimeIndex(df['Date']).month
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    return df

df = load_data()

# Page 1: General Data Information
def general_info():
    st.title("General Data Information")
    st.write("This page provides an overview of the dataset.")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

  
