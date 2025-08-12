import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np

class FootballDataLoader:
    """
    Data loader class for football statistics
    Handles data loading, cleaning, and preprocessing
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data" / "football_stats.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.metric_categories = {
            'Attack': ['Akurasi Tembakan', 'Tembakan ke Gawang', 'Total Tembakan', 'Tembakan Diblok', 'Tendangan Sudut'],
            'Defense': ['Tekel Sukses', 'Pelanggaran'],
            'Progression': ['Akurasi Umpan', 'Umpan Sukses', 'Umpan Gagal', 'Total Umpan', 'Penguasaan Bola'],
            'General': ['Offside', 'Kartu Kuning', 'Kartu Merah']
        }
    
    @st.cache_data
    def load_data(_self):
        """Load and cache the football statistics data"""
        try:
            df = pd.read_csv(_self.data_path)
            return _self._preprocess_data(df)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {_self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _preprocess_data(self, df):
        """Preprocess the loaded data"""
        # Clean team names (remove extra spaces)
        df['TEAM'] = df['TEAM'].str.strip()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Remove any duplicate teams
        df = df.drop_duplicates(subset=['TEAM'], keep='first')
        
        # Sort teams alphabetically
        df = df.sort_values('TEAM').reset_index(drop=True)
        
        return df
    
    def get_metric_categories(self):
        """Return the metric categories dictionary"""
        return self.metric_categories
    
    def get_all_metrics(self):
        """Get all available metrics (excluding TEAM column)"""
        if self.df is None:
            self.df = self.load_data()
        return [col for col in self.df.columns if col != 'TEAM']
    
    def get_teams(self):
        """Get list of all team names"""
        if self.df is None:
            self.df = self.load_data()
        return self.df['TEAM'].tolist()
    
    def get_team_data(self, team_names):
        """Get data for specific teams"""
        if self.df is None:
            self.df = self.load_data()
        
        if isinstance(team_names, str):
            team_names = [team_names]
        
        return self.df[self.df['TEAM'].isin(team_names)]
    
    def get_category_metrics(self, category):
        """Get metrics for a specific category"""
        return self.metric_categories.get(category, [])
    
    def calculate_category_scores(self, df=None):
        """Calculate average scores for each category"""
        if df is None:
            df = self.df if self.df is not None else self.load_data()
        
        category_scores = {}
        for category, metrics in self.metric_categories.items():
            # Only use metrics that exist in the dataframe
            available_metrics = [m for m in metrics if m in df.columns]
            if available_metrics:
                category_scores[f'{category}_Score'] = df[available_metrics].mean(axis=1)
        
        return pd.DataFrame(category_scores, index=df.index)
    
    def get_top_performers(self, metric, n=5):
        """Get top N performers for a specific metric"""
        if self.df is None:
            self.df = self.load_data()
        
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        return self.df.nlargest(n, metric)[['TEAM', metric]]
    
    def get_team_rankings(self, metric):
        """Get team rankings for a specific metric"""
        if self.df is None:
            self.df = self.load_data()
        
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        rankings = self.df[['TEAM', metric]].sort_values(metric, ascending=False).reset_index(drop=True)
        rankings.index = rankings.index + 1  # Start ranking from 1
        return rankings
    
    def get_correlation_matrix(self, metrics=None):
        """Get correlation matrix for specified metrics or all numeric metrics"""
        if self.df is None:
            self.df = self.load_data()
        
        if metrics is None:
            metrics = self.get_all_metrics()
        
        # Filter to only include metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        return self.df[available_metrics].corr()
    
    def get_summary_stats(self, metrics=None):
        """Get summary statistics for specified metrics"""
        if self.df is None:
            self.df = self.load_data()
        
        if metrics is None:
            metrics = self.get_all_metrics()
        
        # Filter to only include metrics that exist in dataframe
        available_metrics = [m for m in metrics if m in self.df.columns]
        
        return self.df[available_metrics].describe()
    
    def search_teams(self, query):
        """Search for teams by name"""
        if self.df is None:
            self.df = self.load_data()
        
        query = query.lower()
        matching_teams = self.df[self.df['TEAM'].str.lower().str.contains(query, na=False)]
        return matching_teams['TEAM'].tolist()
    
    def validate_data(self):
        """Validate the loaded data for completeness and consistency"""
        if self.df is None:
            self.df = self.load_data()
        
        validation_results = {
            'total_teams': len(self.df),
            'total_metrics': len(self.get_all_metrics()),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_teams': self.df['TEAM'].duplicated().sum(),
            'categories_coverage': {}
        }
        
        # Check category coverage
        for category, metrics in self.metric_categories.items():
            available = sum(1 for m in metrics if m in self.df.columns)
            validation_results['categories_coverage'][category] = {
                'available_metrics': available,
                'total_metrics': len(metrics),
                'coverage_percentage': (available / len(metrics)) * 100
            }
        
        return validation_results

# Utility functions for easy access
@st.cache_data
def load_football_data():
    """Simple function to load football data with caching"""
    loader = FootballDataLoader()
    return loader.load_data()

def get_metric_categories():
    """Get metric categories"""
    loader = FootballDataLoader()
    return loader.get_metric_categories()

def calculate_team_category_scores(df):
    """Calculate category scores for teams"""
    loader = FootballDataLoader()
    return loader.calculate_category_scores(df)