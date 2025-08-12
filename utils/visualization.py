import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st

class FootballVisualizer:
    """
    Visualization utilities for football analytics dashboard
    Contains various chart creation methods using Plotly
    """
    
    def __init__(self):
        # Color schemes
        self.team_colors = [
            '#FF6B35', '#F7931E', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE',
            '#85C1E9', '#F8C471', '#82E0AA', '#F1948A', '#AED6F1'
        ]
        
        self.category_colors = {
            'Attack': '#FF6B35',
            'Defense': '#4ECDC4', 
            'Progression': '#F7931E',
            'General': '#45B7D1'
        }
        
        # Default layout settings
        self.default_layout = {
            'font': dict(family="Arial, sans-serif", size=12),
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(l=50, r=50, t=50, b=50)
        }
    
    def create_radar_chart(self, df, teams=None, metrics=None, title="Team Comparison Radar Chart"):
        """Create radar chart for team comparison"""
        if teams is None:
            teams = df['TEAM'].tolist()
        if metrics is None:
            metrics = df.select_dtypes(include=['number']).columns.tolist()
        
        # Filter data
        team_data = df[df['TEAM'].isin(teams)]
        
        fig = go.Figure()
        
        for i, (_, team_row) in enumerate(team_data.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=team_row[metrics].values,
                theta=metrics,
                fill='toself',
                name=team_row['TEAM'],
                line_color=self.team_colors[i % len(self.team_colors)],
                fillcolor=self.team_colors[i % len(self.team_colors)],
                opacity=0.6
            ))
        
        # Calculate max value for radial axis
        max_val = max(df[metrics].max()) if len(metrics) > 0 else 100
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_val * 1.1]
                )
            ),
            showlegend=True,
            title=title,
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_category_bar_chart(self, df, teams, metrics, category_name, orientation='v'):
        """Create bar chart for category metrics comparison"""
        team_data = df[df['TEAM'].isin(teams)]
        
        fig = go.Figure()
        
        if orientation == 'v':
            for i, (_, team_row) in enumerate(team_data.iterrows()):
                fig.add_trace(go.Bar(
                    name=team_row['TEAM'],
                    x=metrics,
                    y=team_row[metrics].values,
                    marker_color=self.team_colors[i % len(self.team_colors)]
                ))
            fig.update_layout(
                xaxis_title="Metrics",
                yaxis_title="Values",
                barmode='group'
            )
        else:
            for i, (_, team_row) in enumerate(team_data.iterrows()):
                fig.add_trace(go.Bar(
                    name=team_row['TEAM'],
                    y=metrics,
                    x=team_row[metrics].values,
                    marker_color=self.team_colors[i % len(self.team_colors)],
                    orientation='h'
                ))
            fig.update_layout(
                xaxis_title="Values",
                yaxis_title="Metrics",
                barmode='group'
            )
        
        fig.update_layout(
            title=f"{category_name} Metrics Comparison",
            height=400,
            **self.default_layout
        )
        
        return fig
    
    def create_performance_heatmap(self, df, metrics=None, title="Team Performance Heatmap"):
        """Create performance heatmap"""
        if metrics is None:
            metrics = df.select_dtypes(include=['number']).columns.tolist()
        
        # Normalize data for better visualization
        normalized_data = df[metrics].copy()
        for col in metrics:
            normalized_data[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100
        
        fig = px.imshow(
            normalized_data.T,
            labels=dict(x="Team Index", y="Metrics", color="Normalized Score"),
            x=df['TEAM'],
            y=metrics,
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title=title
        )
        
        fig.update_layout(
            height=max(400, len(metrics) * 25),
            xaxis_title="Teams",
            yaxis_title="Metrics",
            **self.default_layout
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_scatter_plot(self, df, x_metric, y_metric, size_metric=None, color_metric=None, 
                           show_trendline=False, title=None):
        """Create scatter plot for correlation analysis"""
        if title is None:
            title = f"{x_metric} vs {y_metric}"
        
        # Prepare hover text
        hover_text = df['TEAM']
        
        fig = px.scatter(
            df,
            x=x_metric,
            y=y_metric,
            hover_name='TEAM',
            color=color_metric if color_metric else df.select_dtypes(include=['number']).mean(axis=1),
            size=size_metric if size_metric else [20] * len(df),
            color_continuous_scale='viridis',
            title=title,
            trendline="ols" if show_trendline else None
        )
        
        fig.update_traces(
            marker=dict(opacity=0.7, line=dict(width=2, color='white')),
            selector=dict(mode='markers')
        )
        
        fig.update_layout(
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_top_performers_chart(self, df, metric, n=10, chart_type='bar'):
        """Create chart showing top performers for a specific metric"""
        top_data = df.nlargest(n, metric)
        
        if chart_type == 'bar':
            fig = px.bar(
                top_data,
                x='TEAM',
                y=metric,
                title=f"Top {n} Teams: {metric}",
                color=metric,
                color_continuous_scale='blues'
            )
            fig.update_xaxes(tickangle=45)
            
        elif chart_type == 'horizontal_bar':
            fig = px.bar(
                top_data.sort_values(metric),
                x=metric,
                y='TEAM',
                orientation='h',
                title=f"Top {n} Teams: {metric}",
                color=metric,
                color_continuous_scale='blues'
            )
            
        elif chart_type == 'pie':
            fig = px.pie(
                top_data,
                values=metric,
                names='TEAM',
                title=f"Top {n} Teams: {metric}"
            )
        
        fig.update_layout(
            height=400,
            showlegend=False if chart_type in ['bar', 'horizontal_bar'] else True,
            **self.default_layout
        )
        
        return fig
    
    def create_distribution_histogram(self, df, metric, bins=20, title=None):
        """Create histogram showing distribution of a metric"""
        if title is None:
            title = f"Distribution of {metric}"
        
        fig = px.histogram(
            df,
            x=metric,
            nbins=bins,
            title=title,
            color_discrete_sequence=[self.category_colors['Attack']]
        )
        
        # Add vertical line for mean
        mean_val = df[metric].mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_val:.1f}")
        
        fig.update_layout(
            height=400,
            xaxis_title=metric,
            yaxis_title="Frequency",
            **self.default_layout
        )
        
        return fig
    
    def create_category_comparison_chart(self, df, categories_data, chart_type='radar'):
        """Create comparison chart for category scores"""
        if chart_type == 'radar':
            fig = go.Figure()
            
            for i, (_, team_row) in enumerate(df.iterrows()):
                category_scores = []
                category_names = []
                
                for category in categories_data.columns:
                    if category.endswith('_Score'):
                        category_scores.append(team_row[category] if category in df.columns else 0)
                        category_names.append(category.replace('_Score', ''))
                
                fig.add_trace(go.Scatterpolar(
                    r=category_scores,
                    theta=category_names,
                    fill='toself',
                    name=team_row['TEAM'],
                    line_color=self.team_colors[i % len(self.team_colors)]
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                title="Category Performance Comparison"
            )
            
        elif chart_type == 'bar':
            # Create grouped bar chart for categories
            fig = go.Figure()
            categories = [col.replace('_Score', '') for col in categories_data.columns if col.endswith('_Score')]
            
            for i, (_, team_row) in enumerate(df.iterrows()):
                scores = [team_row.get(f"{cat}_Score", 0) for cat in categories]
                
                fig.add_trace(go.Bar(
                    name=team_row['TEAM'],
                    x=categories,
                    y=scores,
                    marker_color=self.team_colors[i % len(self.team_colors)]
                ))
            
            fig.update_layout(
                barmode='group',
                title="Category Performance Comparison",
                xaxis_title="Categories",
                yaxis_title="Average Score"
            )
        
        fig.update_layout(height=500, **self.default_layout)
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix, title="Metrics Correlation Heatmap"):
        """Create correlation heatmap"""
        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu',
            aspect="auto",
            title=title,
            color_continuous_midpoint=0
        )
        
        # Add correlation values as text
        fig.update_traces(
            text=correlation_matrix.round(2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
        
        fig.update_layout(
            height=max(400, len(correlation_matrix) * 30),
            **self.default_layout
        )
        
        return fig
    
    def create_multi_metric_line_chart(self, df, teams, metrics, title="Multi-Metric Comparison"):
        """Create line chart comparing multiple metrics for selected teams"""
        fig = go.Figure()
        
        team_data = df[df['TEAM'].isin(teams)]
        
        for i, (_, team_row) in enumerate(team_data.iterrows()):
            fig.add_trace(go.Scatter(
                x=metrics,
                y=team_row[metrics].values,
                mode='lines+markers',
                name=team_row['TEAM'],
                line=dict(color=self.team_colors[i % len(self.team_colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Values",
            height=500,
            **self.default_layout
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_gauge_chart(self, value, title, max_value=100, color_scheme='green'):
        """Create gauge chart for single metric display"""
        color_map = {
            'green': ['#ff4444', '#ffaa00', '#00aa00'],
            'blue': ['#ff4444', '#ffaa00', '#0088ff'],
            'red': ['#00aa00', '#ffaa00', '#ff4444']
        }
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            gauge = {
                'axis': {'range': [None, max_value]},
                'bar': {'color': color_map[color_scheme][1]},
                'steps': [
                    {'range': [0, max_value*0.33], 'color': color_map[color_scheme][0]},
                    {'range': [max_value*0.33, max_value*0.67], 'color': color_map[color_scheme][1]},
                    {'range': [max_value*0.67, max_value], 'color': color_map[color_scheme][2]}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300, **self.default_layout)
        
        return fig

# Utility functions
def get_color_palette(n_colors):
    """Get a color palette with n colors"""
    visualizer = FootballVisualizer()
    return visualizer.team_colors[:n_colors] if n_colors <= len(visualizer.team_colors) else visualizer.team_colors * (n_colors // len(visualizer.team_colors) + 1)

def create_metric_summary_table(df, metrics):
    """Create a summary table for metrics"""
    summary = df[metrics].describe().round(2)
    return summary