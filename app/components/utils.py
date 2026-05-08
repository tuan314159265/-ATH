"""
Streamlit Utility Components

Reusable functions and components for the dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple


def load_cache_data(path: str, cache_key: str = None) -> pd.DataFrame:
    """Load data with Streamlit caching."""
    @st.cache_data
    def _load(filepath):
        return pd.read_csv(filepath)
    
    return _load(path)


def metric_card(title: str, value: str, change: str = None, icon: str = "📊"):
    """Display a metric card."""
    col = st.columns(1)[0]
    with col:
        st.metric(f"{icon} {title}", value, change)


def plot_distribution(data: pd.DataFrame, column: str, title: str = None):
    """Plot distribution of a column."""
    fig = px.histogram(data, x=column, nbins=30, title=title or f"Distribution of {column}")
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation(data: pd.DataFrame, title: str = "Correlation Matrix"):
    """Plot correlation heatmap."""
    corr = data.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
    fig.update_layout(title=title, height=600)
    st.plotly_chart(fig, use_container_width=True)


def plot_scatter(data: pd.DataFrame, x: str, y: str, color: str = None, title: str = None):
    """Plot scatter diagram."""
    fig = px.scatter(data, x=x, y=y, color=color, title=title)
    st.plotly_chart(fig, use_container_width=True)


def plot_bar(data: Dict[str, int], title: str = "Bar Chart"):
    """Plot bar chart from dictionary."""
    fig = px.bar(x=list(data.keys()), y=list(data.values()), title=title)
    st.plotly_chart(fig, use_container_width=True)


def render_sidebar_info():
    """Render sidebar information panel."""
    st.sidebar.markdown("""
    ### 📊 Platform Stats
    - **Data Sources**: 3
    - **Records**: 1.2M+
    - **Customers**: 50K+
    """)


def render_footer():
    """Render footer with links."""
    st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 12px; color: #888;'>
        <p>Built with ❤️ using Python, Streamlit & Scikit-learn</p>
        <p>© 2024 Retail Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)


def display_dataframe_with_filters(data: pd.DataFrame, filters: List[str] = None):
    """Display dataframe with optional filters."""
    if filters:
        for column in filters:
            if column in data.columns:
                unique_values = data[column].unique()
                selected_values = st.multiselect(f"Filter by {column}", unique_values)
                if selected_values:
                    data = data[data[column].isin(selected_values)]
    
    st.dataframe(data, use_container_width=True)
    
    return data


def display_summary_stats(data: pd.DataFrame):
    """Display summary statistics."""
    st.subheader("📊 Summary Statistics")
    
    stats = {
        "Total Records": len(data),
        "Columns": len(data.columns),
        "Missing Values": data.isnull().sum().sum(),
        "Duplicate Rows": data.duplicated().sum()
    }
    
    cols = st.columns(len(stats))
    for col, (key, value) in zip(cols, stats.items()):
        with col:
            st.metric(key, value)
