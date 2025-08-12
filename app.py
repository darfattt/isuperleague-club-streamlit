import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Indonesia Super League Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load the football statistics data"""
    data_path = Path(__file__).parent / "data" / "football_stats.csv"
    return pd.read_csv(data_path)

# Define metric categories
METRIC_CATEGORIES = {
    'Attack': ['Akurasi Tembakan', 'Tembakan ke Gawang', 'Total Tembakan', 'Tembakan Diblok', 'Tendangan Sudut'],
    'Defense': ['Tekel Sukses', 'Pelanggaran'],
    'Progression': ['Akurasi Umpan', 'Umpan Sukses', 'Umpan Gagal', 'Total Umpan', 'Penguasaan Bola'],
    'General': ['Offside', 'Kartu Kuning', 'Kartu Merah']
}

# Define negative metrics (lower values are better)
NEGATIVE_METRICS = [
    'Tembakan Diblok', 'Pelanggaran', 'Umpan Gagal', 
    'Offside', 'Kartu Merah', 'Kartu Kuning'
]

# Define weights for all negative metrics (higher weight = more severe impact)
NEGATIVE_METRIC_WEIGHTS = {
    # General category - discipline and tactical violations
    'Kartu Merah': 5.0,    # Most severe - player ejection, significant team disadvantage
    'Kartu Kuning': 2.0,   # Moderate - caution/warning, risk of ejection
    'Offside': 1.0,        # Least severe - tactical violation, opportunity lost
    
    # Attack category - shot blocking
    'Tembakan Diblok': 1.5, # Defensive success against attack
    
    # Defense category - fouls committed  
    'Pelanggaran': 2.5,    # High impact - can lead to free kicks, penalties, cards
    
    # Progression category - passing failures
    'Umpan Gagal': 1.2     # Moderate impact - loss of possession
}

# Legacy weights for backward compatibility (General category only)
GENERAL_WEIGHTS = {
    'Kartu Merah': 5,    # Most severe - player ejection
    'Kartu Kuning': 2,   # Moderate - caution/warning  
    'Offside': 1         # Least severe - tactical violation
}

# Main navigation
def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Indonesia Super League Football Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Data file not found! Please make sure football_stats.csv exists in the data folder.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏆 Top Stats Overview", "📊 Club Comparison", "🎯 Scatter Analysis", "📈 Category Analysis"],
        index=0
    )
    
    # Display team count and data info
    st.sidebar.markdown("### 📈 Data Summary")
    st.sidebar.info(f"**Teams**: {len(df)} clubs\n**Metrics**: {len(df.columns)-1} statistics")
    
    # Page routing
    if page == "🏆 Top Stats Overview":
        show_top_stats_overview(df)
    elif page == "📊 Club Comparison":
        show_club_comparison(df)
    elif page == "🎯 Scatter Analysis":
        show_scatter_analysis(df)
    elif page == "📈 Category Analysis":
        show_category_analysis(df)

def show_top_stats_overview(df):
    """Display the top stats overview page"""
    st.header("🏆 Team Performance Analysis")
    
    # Performers by category
    st.subheader("📊 Performance by Category")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏅 Overall", "⚔️ Attack", "🛡️ Defense", "📈 Progression", "⚽ General"])
    
    with tab1:
        show_overall_performance(df)
    
    with tab2:
        show_category_all_performers(df, "Attack", METRIC_CATEGORIES['Attack'])
    
    with tab3:
        show_category_all_performers(df, "Defense", METRIC_CATEGORIES['Defense'])
    
    with tab4:
        show_category_all_performers(df, "Progression", METRIC_CATEGORIES['Progression'])
    
    with tab5:
        show_category_all_performers(df, "General", METRIC_CATEGORIES['General'])
    
    # Overall performance heatmap
    st.subheader("🔥 Performance Heatmap")
    show_performance_heatmap(df)

def show_category_all_performers(df, category_name, metrics):
    """Show all performers for a specific category with integrated progress bars"""
    st.write(f"**All Teams - {category_name} Performance**")
    
    # Special handling for General category with weighted scores
    if category_name == "General":
        # Calculate weighted scores
        performance_scores, penalty_scores = calculate_weighted_general_score(df)
        category_avg = pd.Series(performance_scores, index=df.index)
        
        # Create rankings dataframe with additional penalty information
        rankings_df = df[['TEAM'] + metrics].copy()
        rankings_df[f'{category_name}_Performance'] = performance_scores
        rankings_df['Weighted_Penalty'] = penalty_scores
        rankings_df = rankings_df.sort_values(f'{category_name}_Performance', ascending=False).reset_index(drop=True)
        rankings_df.index = rankings_df.index + 1  # Start ranking from 1
        
        # Add percentage column (normalize to 0-1 for progress column)
        max_score = max(performance_scores)
        rankings_df['Performance_%'] = [(score / max_score) for score in rankings_df[f'{category_name}_Performance']]
        
        # Add explanation for General category
        st.info("💡 **General Category Weighting**: Red Cards (×5) > Yellow Cards (×2) > Offside (×1). " +
               "Lower penalty scores indicate better discipline. Performance score is inverted for ranking.")
        
        # Display table with weighted penalty information
        st.dataframe(
            rankings_df,
            column_config={
                "Performance_%": st.column_config.ProgressColumn(
                    "Performance %",
                    help="Performance percentage based on weighted discipline score",
                    min_value=0,
                    max_value=1,
                ),
                f'{category_name}_Performance': st.column_config.NumberColumn(
                    "Performance Score",
                    help="Weighted performance score (higher is better)",
                    format="%.2f"
                ),
                'Weighted_Penalty': st.column_config.NumberColumn(
                    "Penalty Score",
                    help="Weighted penalty score (lower is better)",
                    format="%.1f"
                )
            },
            hide_index=False,
            use_container_width=True
        )
        
        chart_y_column = f'{category_name}_Performance'
        chart_title = f"All Teams - {category_name} Weighted Performance"
        chart_y_label = "Weighted Performance Score"
        
    else:
        # Regular category calculation
        category_avg = calculate_category_average(df, metrics, category_name)
        
        # Create rankings dataframe
        rankings_df = df[['TEAM'] + metrics].copy()
        rankings_df[f'{category_name}_Average'] = category_avg
        rankings_df = rankings_df.sort_values(f'{category_name}_Average', ascending=False).reset_index(drop=True)
        rankings_df.index = rankings_df.index + 1  # Start ranking from 1
        
        # Add percentage column (normalize to 0-1 for progress column)
        max_score = category_avg.max()
        rankings_df['Performance_%'] = (rankings_df[f'{category_name}_Average'] / max_score).round(3)
        
        # Display table with integrated progress column
        st.dataframe(
            rankings_df,
            column_config={
                "Performance_%": st.column_config.ProgressColumn(
                    "Performance %",
                    help=f"Performance percentage relative to top {category_name} team",
                    min_value=0,
                    max_value=1,
                ),
            },
            hide_index=False,
            use_container_width=True
        )
        
        chart_y_column = f'{category_name}_Average'
        chart_title = f"All Teams - {category_name} Performance"
        chart_y_label = f"Average {category_name} Score"
    
    st.markdown("---")
    
    # Bar chart showing all teams
    fig = px.bar(
        rankings_df,
        x='TEAM', 
        y=chart_y_column,
        title=chart_title,
        labels={'TEAM': 'Team', chart_y_column: chart_y_label},
        color=chart_y_column,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, key=f"category_all_{category_name.lower()}_bar_chart")

def show_performance_heatmap(df):
    """Display performance heatmap with per-metric normalization and proper negative metrics handling"""
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create normalized data with per-metric scaling
    normalized_data = pd.DataFrame(index=df.index, columns=numeric_cols)
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max == col_min:  # Handle case where all values are the same
            normalized_data[col] = 50  # Set to middle value
        else:
            if col in NEGATIVE_METRICS:
                # For negative metrics, invert the normalization (lower values = better = higher score)
                normalized_data[col] = 100 - ((df[col] - col_min) / (col_max - col_min) * 100)
            else:
                # For positive metrics, normal normalization (higher values = better = higher score)
                normalized_data[col] = (df[col] - col_min) / (col_max - col_min) * 100
    
    # Create heatmap with normalized data
    fig = px.imshow(
        normalized_data.values,  # Use normalized data
        labels=dict(x="Metrics", y="Teams", color="Normalized Score (0-100)"),
        x=numeric_cols,  # Metrics on X-axis
        y=df['TEAM'],   # Teams on Y-axis
        aspect="auto",
        color_continuous_scale='RdYlGn',  # Red (poor) to Green (good)
        zmin=0,
        zmax=100,
        text_auto=False  # We'll add custom text with actual values
    )
    
    # Add text annotations with actual values
    actual_values = df[numeric_cols].values
    text_annotations = [[f"{val:.1f}" if val != 0 else "0" for val in row] for row in actual_values]
    
    fig.update_traces(
        text=text_annotations,
        texttemplate="%{text}",
        textfont={"size": 9, "color": "black"},
        hovertemplate="<b>%{y}</b><br>%{x}<br>Actual Value: %{text}<br>Normalized: %{z:.0f}<extra></extra>"
    )
    
    fig.update_layout(
        title="Team Performance Heatmap (Per-Metric Normalized)",
        height=max(400, len(df) * 25),  # Adjust height based on number of teams
        xaxis_title="Metrics",
        yaxis_title="Teams"
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    # Add explanation
    st.info("💡 **Enhanced Heatmap Guide**: All 15 metrics are displayed with actual values shown on each cell. " +
            "Each metric is normalized individually (0-100 scale) for color coding. " +
            "Green = Better performance, Red = Poorer performance. " +
            "For negative metrics (cards, fouls, etc.), lower actual values appear green. " +
            "Zero values are clearly marked and properly handled. " +
            "Hover for detailed actual and normalized values.")
    
    st.plotly_chart(fig, use_container_width=True, key="performance_heatmap")

def calculate_weighted_general_score(df):
    """Calculate weighted General category scores (lower penalty = better performance)"""
    weighted_performance_scores = []
    
    for _, row in df.iterrows():
        # Calculate total weighted penalty score
        penalty_score = (
            row['Kartu Merah'] * GENERAL_WEIGHTS['Kartu Merah'] +
            row['Kartu Kuning'] * GENERAL_WEIGHTS['Kartu Kuning'] + 
            row['Offside'] * GENERAL_WEIGHTS['Offside']
        )
        weighted_performance_scores.append(penalty_score)
    
    # Convert to performance score (invert so lower penalty = higher performance)
    max_penalty = max(weighted_performance_scores) if weighted_performance_scores else 0
    
    # Add small buffer to avoid zero scores
    performance_scores = [(max_penalty + 1 - penalty) for penalty in weighted_performance_scores]
    
    return performance_scores, weighted_performance_scores

def calculate_category_average(df, metrics, category_name=None):
    """Calculate category average with special handling for General category"""
    if category_name == "General":
        performance_scores, _ = calculate_weighted_general_score(df)
        return pd.Series(performance_scores, index=df.index)
    else:
        return df[metrics].mean(axis=1)

def show_overall_performance(df):
    """Show overall performance combining all metrics with full data"""
    st.write("**Overall Team Performance - All Metrics Combined**")
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate category averages with weighted General category
    category_averages = {}
    for category, metrics in METRIC_CATEGORIES.items():
        category_averages[category] = calculate_category_average(df, metrics, category)
    
    # Calculate overall average using category averages (not simple mean of all metrics)
    overall_avg = pd.DataFrame(category_averages).mean(axis=1)
    
    # Create comprehensive overall dataframe with all metrics
    overall_df = df[['TEAM'] + numeric_cols].copy()
    overall_df['Overall_Average'] = overall_avg
    overall_df = overall_df.sort_values('Overall_Average', ascending=False).reset_index(drop=True)
    overall_df.index = overall_df.index + 1  # Start ranking from 1
    
    # Add percentage column (normalize to 0-1 for progress column)
    max_score = overall_avg.max()
    overall_df['Performance_%'] = (overall_df['Overall_Average'] / max_score).round(3)
    
    # Reorder columns to show TEAM, Overall_Average, Performance_%, then all metrics
    cols_order = ['TEAM', 'Overall_Average', 'Performance_%'] + numeric_cols
    overall_df = overall_df[cols_order]
    
    # Display comprehensive table with integrated progress column
    st.dataframe(
        overall_df,
        column_config={
            "Performance_%": st.column_config.ProgressColumn(
                "Performance %",
                help="Overall performance percentage relative to top team",
                min_value=0,
                max_value=1,
            ),
            "Overall_Average": st.column_config.NumberColumn(
                "Overall Average",
                help="Average score across all metrics",
                format="%.2f"
            )
        },
        hide_index=False,
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Overall performance bar chart
    fig = px.bar(
        overall_df,
        x='TEAM',
        y='Overall_Average',
        title="Overall Team Performance - All Metrics",
        labels={'TEAM': 'Team', 'Overall_Average': 'Overall Average Score'},
        color='Overall_Average',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, key="overall_performance_bar_chart")

def show_club_comparison(df):
    """Display club comparison page"""
    st.header("📊 Club Comparison")
    
    # Team selection
    selected_teams = st.multiselect(
        "Select teams to compare (2-3 teams recommended):",
        options=df['TEAM'].tolist(),
        default=df['TEAM'].tolist()[:2] if len(df) >= 2 else [],
        max_selections=3
    )
    
    if len(selected_teams) < 2:
        st.warning("⚠️ Please select at least 2 teams to compare.")
        return
    
    # Filter data for selected teams
    comparison_df = df[df['TEAM'].isin(selected_teams)]
    
    # Create tabs for different visualization types
    tab1, tab2 = st.tabs(["📊 Bar Chart", "🎯 Radar Chart"])
    
    with tab1:
        st.subheader("📊 Bar Chart Comparison")
        show_category_metrics_comparison(comparison_df, df)
    
    with tab2:
        st.subheader("🎯 Radar Chart Comparison")
        show_radar_comparison(comparison_df)

def show_radar_comparison(df):
    """Create radar chart for team comparison with normalized values"""
    
    def hex_to_rgba(hex_color, alpha=0.1):
        """Convert hex color to RGBA format for Plotly compatibility"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Create normalized data for radar chart
    normalized_df = df.copy()
    
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max == col_min:  # Handle case where all values are the same
            normalized_df[col] = 50  # Set to middle value
        else:
            if col in NEGATIVE_METRICS:
                # For negative metrics, invert normalization (lower values = better = higher score)
                normalized_df[col] = 100 - ((df[col] - col_min) / (col_max - col_min) * 100)
            else:
                # For positive metrics, normal normalization (higher values = better = higher score)
                normalized_df[col] = (df[col] - col_min) / (col_max - col_min) * 100
    
    fig = go.Figure()
    
    # Dark theme color palette that works well on dark backgrounds
    colors = ['#00D4FF', '#FF6B9D', '#4ECDC4', '#FFD93D', '#6BCF7F', '#FF8C42']
    
    for i, (idx, team_data) in enumerate(normalized_df.iterrows()):
        original_team_data = df.loc[idx]
        
        # Create hover text with original values, handling zero values properly
        hover_text = []
        for metric in numeric_cols:
            actual_val = original_team_data[metric]
            normalized_val = team_data[metric]
            # Format zero values properly
            actual_str = "0" if actual_val == 0 else f"{actual_val:.1f}"
            hover_text.append(f"{metric}: {actual_str} (normalized: {normalized_val:.1f}%)")
        
        fig.add_trace(go.Scatterpolar(
            r=team_data[numeric_cols].values,
            theta=numeric_cols,
            fill='toself',
            name=original_team_data['TEAM'],
            line=dict(
                color=colors[i % len(colors)],
                width=3  # Thicker lines for better visibility
            ),
            fillcolor=hex_to_rgba(colors[i % len(colors)], 0.1),  # Semi-transparent fill
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                line=dict(color='white', width=2)  # White border for contrast
            ),
            hovertemplate='<b>%{fullData.name}</b><br>%{text}<extra></extra>',
            text=hover_text
        ))
    
    # Add team name annotations in corners (like scatter plot style)
    positions = [
        {'x': 0.95, 'y': 0.95, 'xanchor': 'right', 'yanchor': 'top'},
        {'x': 0.05, 'y': 0.95, 'xanchor': 'left', 'yanchor': 'top'},
        {'x': 0.95, 'y': 0.05, 'xanchor': 'right', 'yanchor': 'bottom'},
        {'x': 0.05, 'y': 0.05, 'xanchor': 'left', 'yanchor': 'bottom'},
        {'x': 0.5, 'y': 0.95, 'xanchor': 'center', 'yanchor': 'top'},
        {'x': 0.5, 'y': 0.05, 'xanchor': 'center', 'yanchor': 'bottom'}
    ]
    
    for i, (idx, team_data) in enumerate(normalized_df.iterrows()):
        original_team_data = df.loc[idx]
        team_name = original_team_data['TEAM']
        pos = positions[i % len(positions)]
        
        fig.add_annotation(
            x=pos['x'], y=pos['y'],
            text=f"<b>{team_name}</b>",
            showarrow=False,
            font=dict(
                color=colors[i % len(colors)],
                size=12,
                family="Arial Black"
            ),
            bgcolor='rgba(30, 38, 49, 0.9)',  # Semi-transparent dark background
            bordercolor=colors[i % len(colors)],
            borderwidth=1,
            borderpad=4,
            xref="paper", yref="paper",  # Use paper coordinates
            xanchor=pos['xanchor'], yanchor=pos['yanchor']
        )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linecolor='#4a90a4',  # Subtle cyan grid lines
                gridcolor='#4a90a4',
                gridwidth=1,
                tickfont=dict(color='#e6e6e6', size=10),
                tickcolor='#4a90a4'
            ),
            angularaxis=dict(
                showline=True,
                linecolor='#4a90a4',
                gridcolor='#4a90a4',
                gridwidth=1,
                tickfont=dict(color='#ffffff', size=11, family='Arial'),
                tickcolor='#4a90a4'
            ),
            bgcolor='rgba(0,0,0,0)'  # Transparent polar background
        ),
        showlegend=True,
        height=600,
        title=dict(
            text="Team Performance Radar Chart",
            font=dict(color='#ffffff', size=16),
            x=0.5
        ),
        # Dark theme background with enhanced styling
        plot_bgcolor='#1e2631',
        paper_bgcolor='#1e2631',
        font=dict(color='#ffffff'),
        legend=dict(
            font=dict(color='#ffffff', size=12),
            bgcolor='rgba(0,0,0,0.4)',
            bordercolor='#4a90a4',
            borderwidth=1,
            orientation="h",  # Horizontal legend
            x=0.5, xanchor="center",
            y=-0.05, yanchor="top"
        ),
        margin=dict(t=70, b=70, l=70, r=70),  # Add margins for annotations
        hovermode='closest'
    )
    
    # Add explanation with enhanced information
    # st.info("💡 **Professional Dark Theme Radar Chart**: All 15 metrics displayed with modern sports analytics styling. " +
    #        "Metrics scaled 0-100 for fair comparison. Team names positioned around the chart for easy identification. " +
    #        "For negative metrics (cards, fouls, etc.), lower values appear as higher scores. " +
    #        "Enhanced with weighted scoring and professional dark theme design. " +
    #        "Hover over data points to see detailed actual and normalized values.")
    
    st.plotly_chart(fig, use_container_width=True, key="radar_comparison_chart")

def show_category_metrics_comparison(selected_teams_df, full_dataset_df):
    """Show horizontal bar charts organized by teams in columns"""
    # Create columns based on number of teams (max 3)
    num_teams = len(selected_teams_df)
    if num_teams == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    # Create team comparison charts
    for idx, (_, team_data) in enumerate(selected_teams_df.iterrows()):
        if idx < len(cols):  # Limit to available columns
            with cols[idx]:
                create_team_performance_chart(selected_teams_df, full_dataset_df, team_data)

def create_team_performance_chart(selected_teams_df, full_dataset_df, team_data):
    """Create individual team performance chart with horizontal bars organized by categories"""
    team_name = team_data['TEAM']
    
    # Create a dictionary to store ALL 15 metrics from CSV in logical categories
    metric_order = {
        'General': ['Offside', 'Kartu Kuning', 'Kartu Merah'],
        'Defense': ['Tekel Sukses', 'Pelanggaran'],
        'Progression': ['Akurasi Umpan', 'Umpan Sukses', 'Umpan Gagal', 'Total Umpan', 'Penguasaan Bola'],
        'Attack': ['Akurasi Tembakan', 'Tembakan ke Gawang', 'Total Tembakan', 'Tembakan Diblok', 'Tendangan Sudut'],
    }
    
    # Get all numeric columns from the dataset to ensure we don't miss any
    all_numeric_cols = selected_teams_df.select_dtypes(include=['number']).columns.tolist()
    
    # Verify all metrics are included in categories and add any missing ones
    categorized_metrics = set()
    for metrics in metric_order.values():
        categorized_metrics.update(metrics)
    
    missing_metrics = [col for col in all_numeric_cols if col not in categorized_metrics]
    if missing_metrics:
        # Add missing metrics to appropriate categories or create a new category
        if 'Other' not in metric_order:
            metric_order['Other'] = []
        metric_order['Other'].extend(missing_metrics)
    
    # Debug: Show metrics count for verification
    total_categorized = sum(len(metrics) for metrics in metric_order.values())
    # Uncomment for debugging: st.write(f"🔍 **Debug Info for {team_name}**: Processing {total_categorized} metrics out of {len(all_numeric_cols)} available")
    if missing_metrics:
        st.write(f"📝 Added {len(missing_metrics)} missing metrics to 'Other' category: {missing_metrics}")
    
    # Additional debug info for bar lengths
    # st.write(f"📊 Available metrics: {all_numeric_cols}")
    
    # Create a mapping of metrics to their order within each category
    metric_position = {}
    for category, metrics in metric_order.items():
        for i, metric in enumerate(metrics):
            metric_position[metric] = i
    
    # Prepare data for the chart
    chart_data = []
    category_order = {'General': 0, 'Defense': 1, 'Progression': 2, 'Attack': 3, 'Other': 4}
    
    # Use all_numeric_cols we already calculated above
    
    for category, metrics in metric_order.items():
        for metric in metrics:
            if metric in all_numeric_cols:
                current_value = team_data[metric]
                
                # Calculate percentile using FULL dataset (all teams)
                full_metric_values = full_dataset_df[metric].values
                percentile = (full_metric_values < current_value).sum() / len(full_metric_values) * 100
                
                if metric in NEGATIVE_METRICS:
                    percentile = 100 - percentile  # Invert for negative metrics
                    # Apply weighting for more nuanced color grading, but cap at 100%
                    weight = NEGATIVE_METRIC_WEIGHTS.get(metric, 1.0)
                    # Adjust percentile boundaries based on weight (higher weight = stricter grading)
                    weight_adjustment = (weight - 1.0) * 0.05  # Reduced multiplier to prevent overflow
                    percentile = min(percentile * (1.0 + weight_adjustment), 100.0)  # Cap at 100%
                
                # Calculate bar length using actual values for true proportional representation
                if metric in NEGATIVE_METRICS:
                    # For negative metrics, invert the scale so lower values = longer bars
                    # Find the maximum value across ALL teams for this metric to create consistent scale
                    full_metric_values = full_dataset_df[metric].values
                    metric_max = full_metric_values.max()
                    # Invert: higher actual value = shorter bar, lower actual value = longer bar
                    bar_length = (metric_max - current_value) + 2  # +2 ensures visibility
                else:
                    # For positive metrics, use actual value directly
                    bar_length = current_value + 1  # +1 ensures zero values are visible
                
                # Ensure minimum bar length for visibility
                bar_length = max(bar_length, 1)
                
                # Determine color based on percentile from full dataset
                if percentile >= 81:
                    color = '#1a9641'  # Dark green
                elif percentile >= 61:
                    color = '#73c378'  # Light green
                elif percentile >= 41:
                    color = '#f9d057'  # Yellow
                elif percentile >= 21:
                    color = '#fc8d59'  # Orange
                else:
                    color = '#d73027'  # Red
                
                chart_data.append({
                    'Metric': metric,
                    'Value': current_value,
                    'BarLength': bar_length,
                    'Percentile': percentile,
                    'Color': color,
                    'Category': category,
                    'CategoryOrder': category_order[category],
                    'MetricOrder': metric_position[metric],
                    'HoverText': f"{metric}<br>Actual Value: {current_value:.1f}<br>League Percentile: {percentile:.0f}%<br>Bar Length: {bar_length:.1f}"
                })
    
    # Create DataFrame and sort
    df_chart = pd.DataFrame(chart_data)
    df_chart = df_chart.sort_values(['CategoryOrder', 'MetricOrder'], ascending=[True, True])
    
    # Calculate max_bar_length early for use in shapes and annotations
    max_bar_length = df_chart['BarLength'].max() if not df_chart.empty else 100
    x_range = [0, max_bar_length * 1.1]  # Add 10% padding
    
    # Debug: Verify all metrics are included
    processed_metrics = len(df_chart)
    if processed_metrics != len(all_numeric_cols):
        st.warning(f"⚠️ Only {processed_metrics} out of {len(all_numeric_cols)} metrics processed for {team_name}")
    # else:
    #     st.success(f"✅ All {processed_metrics} metrics processed for {team_name}")
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for each category, showing ALL metrics (including zero values)
    for category in ['General', 'Defense', 'Progression', 'Attack', 'Other']:
        category_df = df_chart[df_chart['Category'] == category]
        if not category_df.empty:
            # Show ALL metrics, including those with zero values
            fig.add_trace(go.Bar(
                x=category_df['BarLength'],  # Use actual-value-based bar lengths
                y=category_df['Metric'],
                orientation='h',
                marker=dict(
                    color=category_df['Color'],
                    line=dict(width=0.5, color='white')
                ),
                text=[f"{val:.1f}" if val > 0 else "0" for val in category_df['Value']],  # Always show actual values
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial Black'),
                hovertext=category_df['HoverText'],
                hoverinfo='text',
                name=category,
                showlegend=False
            ))
    
    # Add category dividers and labels
    prev_category = None
    
    for i, (_, row) in enumerate(df_chart.iterrows()):
        if prev_category is not None and row['Category'] != prev_category:
            # Add horizontal line between categories
            y_pos = i - 0.5
            fig.add_shape(
                type="line",
                x0=0, y0=y_pos,
                x1=max_bar_length * 1.1, y1=y_pos,  # Use dynamic range
                line=dict(color="#888888", width=0.8, dash="solid"),
                opacity=0.3,
                layer="below"
            )
        prev_category = row['Category']
    
    # Add category annotations
    for category in ['General', 'Defense', 'Progression', 'Attack', 'Other']:
        category_df = df_chart[df_chart['Category'] == category]
        if not category_df.empty:
            # Find middle position for category label
            mid_idx = len(category_df) // 2
            mid_metric = category_df['Metric'].iloc[mid_idx]
            
            fig.add_annotation(
                x=max_bar_length * 1.05,  # Position based on actual data range
                y=mid_metric,
                text=category,
                showarrow=False,
                font=dict(size=12, color="#333333"),
                align="center",
                textangle=270,
                xanchor="left",
                yanchor="middle"
            )
    
    # Add team name at the top
    fig.add_annotation(
        x=0.01,
        y=1.05,
        text=f"<b>{team_name}</b>",
        showarrow=False,
        font=dict(size=14, color="#333333"),
        align="left",
        xanchor="left",
        yanchor="bottom",
        xref="paper",
        yref="paper"
    )
    
    # Add performance legend at the bottom
    legend_items = [
        {'color': '#d73027', 'label': '1-20%'},
        {'color': '#fc8d59', 'label': '21-40%'},
        {'color': '#f9d057', 'label': '41-60%'},
        {'color': '#73c378', 'label': '61-80%'},
        {'color': '#1a9641', 'label': '81-100%'}
    ]
    
    legend_text = " | ".join([f'<span style="color:{item["color"]}">\u25A0</span> {item["label"]}' 
                             for item in legend_items])
    
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        text=legend_text,
        showarrow=False,
        font=dict(size=10, color="#333333"),
        align="center",
        xanchor="center",
        yanchor="top",
        xref="paper",
        yref="paper"
    )
    
    # Update layout (using x_range calculated earlier)
    fig.update_layout(
        title=None,
        xaxis=dict(
            title="Actual Values (with offset for visibility)",
            range=x_range,
            showgrid=True,
            gridcolor='rgba(136, 136, 136, 0.2)',
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color='#666666')
        ),
        yaxis=dict(
            title=None,
            autorange="reversed",
            tickfont=dict(size=10, color='#333333')
        ),
        margin=dict(l=10, r=50, t=120, b=80),
        plot_bgcolor='#F9F7F2',
        paper_bgcolor='#F9F7F2',
        height=600,
        width=400,
        barmode='stack',
        bargap=0.15,
        hovermode='closest'
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(136, 136, 136, 0.2)')
    
    st.plotly_chart(fig, use_container_width=True, key=f"team_performance_{team_name.lower().replace(' ', '_')}")

def show_scatter_analysis(df):
    """Display scatter analysis page"""
    st.header("🎯 Scatter Plot Analysis")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        x_axis = st.selectbox("Select X-axis metric:", numeric_cols, index=0)
    
    with col2:
        y_axis = st.selectbox("Select Y-axis metric:", numeric_cols, index=1)
    
    with col3:
        show_trendline = st.checkbox("Show trend line", value=False)
    
    with col4:
        show_median_lines = st.checkbox("Show Median lines", value=True)
    
    with col5:
        show_team_names = st.checkbox("Show Team Name", value=True)
    
    # Create scatter plot (removed trendline to avoid statsmodels dependency)
    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        hover_name='TEAM',
        color=df[numeric_cols].mean(axis=1),
        size=df['Penguasaan Bola'],
        color_continuous_scale='viridis',
        title=f"{x_axis} vs {y_axis}"
    )
    
    # Add manual trendline if requested
    if show_trendline:
        # Simple linear regression using numpy
        x_vals = df[x_axis].values
        y_vals = df[y_axis].values
        # Remove any NaN values
        mask = ~(pd.isna(x_vals) | pd.isna(y_vals))
        x_clean = x_vals[mask]
        y_clean = y_vals[mask]
        
        if len(x_clean) > 1:
            # Calculate simple linear regression
            coef = np.polyfit(x_clean, y_clean, 1)
            line_x = np.array([x_clean.min(), x_clean.max()])
            line_y = coef[0] * line_x + coef[1]
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ))
    
    # Add median lines if requested
    if show_median_lines:
        x_median = df[x_axis].median()
        y_median = df[y_axis].median()
        
        # Add vertical median line (X-axis median)
        fig.add_vline(
            x=x_median,
            line=dict(color='gray', width=2, dash='dot'),
            annotation_text=f"X Median: {x_median:.1f}",
            annotation_position="top"
        )
        
        # Add horizontal median line (Y-axis median)  
        fig.add_hline(
            y=y_median,
            line=dict(color='gray', width=2, dash='dot'),
            annotation_text=f"Y Median: {y_median:.1f}",
            annotation_position="right"
        )
    
    # Add team name labels if requested
    if show_team_names:
        for i, (_, team_row) in enumerate(df.iterrows()):
            # Alternate positioning to reduce overlap
            x_offset = 25 if i % 2 == 0 else -25
            y_offset = 20 if i % 4 < 2 else -20
            
            fig.add_annotation(
                x=team_row[x_axis],
                y=team_row[y_axis],
                text=team_row['TEAM'],
                showarrow=False,
                font=dict(
                    size=11,
                    color='white',  # White for dark theme consistency
                    family='Arial Bold'
                ),
                xshift=x_offset,  # Larger offset to avoid overlapping with marker
                yshift=y_offset
            )
    
    fig.update_layout(height=600)
    fig.update_traces(marker=dict(size=12, opacity=0.7))
    
    st.plotly_chart(fig, use_container_width=True, key="scatter_analysis_plot")
    
    # Correlation info
    correlation = df[x_axis].corr(df[y_axis])
    st.info(f"📈 Correlation between {x_axis} and {y_axis}: **{correlation:.3f}**")

def show_category_analysis(df):
    """Display category-specific analysis"""
    st.header("📈 Category Analysis")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏅 Overall", "⚔️ Attack", "🛡️ Defense", "📈 Progression", "⚽ General"])
    
    with tab1:
        show_detailed_overall_analysis(df)
    
    with tab2:
        show_detailed_category_analysis(df, "Attack", METRIC_CATEGORIES['Attack'])
    
    with tab3:
        show_detailed_category_analysis(df, "Defense", METRIC_CATEGORIES['Defense'])
    
    with tab4:
        show_detailed_category_analysis(df, "Progression", METRIC_CATEGORIES['Progression'])
    
    with tab5:
        show_detailed_category_analysis(df, "General", METRIC_CATEGORIES['General'])

def show_detailed_overall_analysis(df):
    """Show detailed overall analysis combining all metrics"""
    st.subheader("🔍 Overall Performance Analysis")
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate category averages with weighted General category
    category_averages = {}
    for category, metrics in METRIC_CATEGORIES.items():
        category_averages[category] = calculate_category_average(df, metrics, category)
    
    # Calculate overall average using category averages
    overall_avg = pd.DataFrame(category_averages).mean(axis=1)
    
    # Add explanation for weighted calculation
    st.info("💡 **Overall Score Calculation**: Uses category averages where General category is weighted " +
           "(Red Cards ×5, Yellow Cards ×2, Offside ×1) to reflect discipline impact accurately.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall Rankings
        st.write("**🏆 Overall Rankings**")
        overall_df = df[['TEAM']].copy()
        overall_df['Overall_Score'] = overall_avg
        rankings = overall_df.sort_values('Overall_Score', ascending=False)
        rankings.index = range(1, len(rankings) + 1)
        
        # Add percentage column (normalize to 0-1 for progress column)
        max_score = overall_avg.max()
        rankings['Performance_%'] = (rankings['Overall_Score'] / max_score).round(3)
        
        st.dataframe(
            rankings,
            column_config={
                "Performance_%": st.column_config.ProgressColumn(
                    "Performance %",
                    help="Overall performance percentage relative to top team",
                    min_value=0,
                    max_value=1,
                ),
                "Overall_Score": st.column_config.NumberColumn(
                    "Overall Score",
                    help="Average score across all metrics",
                    format="%.2f"
                )
            },
            hide_index=False,
            use_container_width=True
        )
    
    with col2:
        # Distribution histogram
        fig = px.histogram(
            overall_df,
            x='Overall_Score',
            title="Overall Score Distribution",
            nbins=10,
            color_discrete_sequence=['#FF6B35']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="overall_distribution_histogram")
    
    # Individual metrics analysis for all metrics
    st.write("**📊 All Metrics Breakdown**")
    
    # Create tabs for better organization
    metrics_per_tab = 5
    tabs_needed = (len(numeric_cols) + metrics_per_tab - 1) // metrics_per_tab
    
    tab_names = [f"Metrics {i*metrics_per_tab+1}-{min((i+1)*metrics_per_tab, len(numeric_cols))}" for i in range(tabs_needed)]
    tabs = st.tabs(tab_names)
    
    for tab_idx, tab in enumerate(tabs):
        with tab:
            start_idx = tab_idx * metrics_per_tab
            end_idx = min((tab_idx + 1) * metrics_per_tab, len(numeric_cols))
            tab_metrics = numeric_cols[start_idx:end_idx]
            
            for metric in tab_metrics:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Determine if this is a negative metric for proper sorting
                    ascending = metric in NEGATIVE_METRICS
                    sorted_df = df.sort_values(metric, ascending=ascending)[:10]
                    
                    fig = px.bar(
                        sorted_df,
                        x='TEAM',
                        y=metric,
                        title=f"{'Bottom' if ascending else 'Top'} 10: {metric}",
                        color=metric,
                        color_continuous_scale='reds' if metric in NEGATIVE_METRICS else 'blues'
                    )
                    fig.update_layout(height=300, showlegend=False)
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True, key=f"overall_metric_{metric}_{tab_idx}")
                
                with col2:
                    st.metric(
                        f"Avg {metric}",
                        f"{df[metric].mean():.1f}",
                        f"{df[metric].std():.1f}"
                    )
                    
                    # Show best/worst team based on metric type
                    if metric in NEGATIVE_METRICS:
                        best_idx = df[metric].idxmin()  # Lowest is best for negative metrics
                        st.metric(
                            "Best Team",
                            df.loc[best_idx, 'TEAM'],
                            f"{df[metric].min()}"
                        )
                    else:
                        best_idx = df[metric].idxmax()  # Highest is best for positive metrics
                        st.metric(
                            "Best Team",
                            df.loc[best_idx, 'TEAM'],
                            f"{df[metric].max()}"
                        )

def show_detailed_category_analysis(df, category_name, metrics):
    """Show detailed analysis for a specific category"""
    st.subheader(f"🔍 {category_name} Analysis")
    
    # Special handling for General category with weighted scores
    if category_name == "General":
        # Calculate weighted scores
        performance_scores, penalty_scores = calculate_weighted_general_score(df)
        category_scores = pd.Series(performance_scores, index=df.index)
        
        # Create enhanced dataframe with penalty information
        df_with_scores = df.copy()
        df_with_scores[f'{category_name}_Performance'] = performance_scores
        df_with_scores['Weighted_Penalty'] = penalty_scores
        df_with_scores[f'{category_name}_Score'] = performance_scores  # For consistency
        
        # Add explanation for General category weighting
        st.info("💡 **General Category Weighting System**: Red Cards (×5) > Yellow Cards (×2) > Offside (×1). " +
               "This weighting reflects the real impact of discipline issues in football. " +
               "Performance Score = Inverted Penalty Score (higher is better).")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced rankings with penalty information
            st.write("**🏆 Weighted Performance Rankings**")
            rankings = df_with_scores[['TEAM', f'{category_name}_Performance', 'Weighted_Penalty']].sort_values(f'{category_name}_Performance', ascending=False)
            rankings.index = range(1, len(rankings) + 1)
            
            # Add percentage for progress column
            max_performance = rankings[f'{category_name}_Performance'].max()
            rankings['Performance_%'] = rankings[f'{category_name}_Performance'] / max_performance
            
            st.dataframe(
                rankings,
                column_config={
                    "Performance_%": st.column_config.ProgressColumn(
                        "Performance %",
                        help="Weighted performance percentage (higher is better)",
                        min_value=0,
                        max_value=1,
                    ),
                    f'{category_name}_Performance': st.column_config.NumberColumn(
                        "Performance Score",
                        help="Weighted performance score (higher is better)",
                        format="%.2f"
                    ),
                    'Weighted_Penalty': st.column_config.NumberColumn(
                        "Penalty Score", 
                        help="Weighted penalty score (lower is better)",
                        format="%.1f"
                    )
                },
                use_container_width=True
            )
        
        with col2:
            # Distribution histogram using performance scores
            fig = px.histogram(
                df_with_scores,
                x=f'{category_name}_Performance',
                title=f"{category_name} Weighted Performance Distribution",
                nbins=10,
                color_discrete_sequence=['#FF6B35']
            )
            fig.update_layout(height=400)
            fig.update_xaxes(title="Weighted Performance Score")
            st.plotly_chart(fig, use_container_width=True, key=f"{category_name.lower()}_distribution_histogram")
        
    else:
        # Regular category analysis
        category_scores = calculate_category_average(df, metrics, category_name)
        df_with_scores = df.copy()
        df_with_scores[f'{category_name}_Score'] = category_scores
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rankings
            st.write("**🏆 Category Rankings**")
            rankings = df_with_scores[['TEAM', f'{category_name}_Score']].sort_values(f'{category_name}_Score', ascending=False)
            rankings.index = range(1, len(rankings) + 1)
            st.dataframe(rankings, use_container_width=True)
        
        with col2:
            # Distribution histogram
            fig = px.histogram(
                df_with_scores,
                x=f'{category_name}_Score',
                title=f"{category_name} Score Distribution",
                nbins=10,
                color_discrete_sequence=['#FF6B35']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"{category_name.lower()}_distribution_histogram")
    
    # Individual metrics analysis
    st.write(f"**📊 {category_name} Metrics Breakdown**")
    
    for metric in metrics:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = px.bar(
                df.sort_values(metric, ascending=False)[:10],
                x='TEAM',
                y=metric,
                title=f"Top 10: {metric}",
                color=metric,
                color_continuous_scale='blues'
            )
            fig.update_layout(height=300, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True, key=f"{category_name.lower()}_metric_{metric}")
        
        with col2:
            st.metric(
                f"Avg {metric}",
                f"{df[metric].mean():.1f}",
                f"{df[metric].std():.1f}"
            )
            st.metric(
                "Best Team",
                df.loc[df[metric].idxmax(), 'TEAM'],
                f"{df[metric].max()}"
            )

if __name__ == "__main__":
    main()