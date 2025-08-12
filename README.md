# ⚽ Indonesia Super League Football Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing Indonesian football statistics with interactive visualizations using Plotly.

## Features

### 🏆 Top Stats Overview
- Performance summary cards showing key metrics
- Top performers by category (Attack, Defense, Progression, General)
- Interactive performance heatmap for all teams
- Category-wise analysis tabs

### 📊 Club Comparison
- Multi-select team comparison (2-3 teams)
- Interactive radar chart comparing all metrics
- Category-wise bar chart comparisons
- Side-by-side metric analysis

### 🎯 Scatter Analysis
- Custom scatter plots with X/Y axis selection
- Interactive Plotly visualizations with hover details
- Optional trend line analysis
- Correlation insights between metrics

### 📈 Category Analysis
- Detailed analysis by metric categories:
  - **Attack**: Shot accuracy, shots on target, total shots, blocked shots, corner kicks
  - **Defense**: Successful tackles, fouls
  - **Progression**: Pass accuracy, successful/failed passes, total passes, ball possession
  - **General**: Offside, yellow cards, red cards
- Performance distribution histograms
- Category rankings and individual metric breakdowns

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Ensure data file exists**:
   - Make sure `data/football_stats.csv` contains your football statistics data
   - The CSV should have a 'TEAM' column and various metric columns

## Usage

### Local Development
1. **Run the dashboard**:
```bash
streamlit run app.py
```

2. **Navigate through pages**:
   - Use the sidebar to switch between different analysis pages
   - Interactive controls allow customization of visualizations
   - Data is automatically cached for better performance

### Deployment to Streamlit Cloud
For production deployment, see the detailed [Deployment Guide](DEPLOYMENT.md).

**Quick Deployment Steps:**
1. Push your code to a public GitHub repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy by selecting your repository and setting main file to `app.py`

Your app will be available at: `https://your-app-name.streamlit.app`

## Data Format

The CSV file should have the following structure:
```
TEAM,Akurasi Umpan,Akurasi Tembakan,Kartu Kuning,Kartu Merah,Offside,...
TEAM_NAME_1,85,60,2,0,1,...
TEAM_NAME_2,82,44,3,0,0,...
```

### Required Columns:
- `TEAM`: Team names
- Various metric columns as per your football statistics

### Metric Categories:
- **Attack**: Shooting and offensive metrics
- **Defense**: Defensive actions and fouls
- **Progression**: Passing and ball control metrics  
- **General**: Cards and game infractions

## Project Structure

```
isuperleague-club-streamlit/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── data_loader.py     # Data loading and preprocessing
│   └── visualization.py   # Chart creation utilities
├── data/
│   └── football_stats.csv # Football statistics data
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── packages.txt           # System dependencies (if needed)
├── .gitignore            # Git ignore rules
├── DEPLOYMENT.md         # Detailed deployment guide
└── README.md             # This file
```

## Features in Detail

### Interactive Visualizations
- **Radar Charts**: Multi-dimensional team comparisons
- **Heatmaps**: Performance overview across all metrics
- **Scatter Plots**: Correlation analysis between any two metrics
- **Bar Charts**: Category-wise and metric-specific comparisons
- **Histograms**: Distribution analysis for performance metrics

### Data Analysis
- **Top Performers**: Identify best teams in each category
- **Correlation Analysis**: Understand relationships between metrics
- **Performance Rankings**: Team standings by different metrics
- **Category Scoring**: Aggregate performance in Attack/Defense/Progression/General

### User Experience
- **Responsive Design**: Works on desktop and mobile
- **Interactive Controls**: Customizable visualizations
- **Real-time Updates**: Dynamic chart updates based on selections
- **Performance Optimized**: Data caching for faster loading

## Customization

### Adding New Metrics
1. Update the CSV file with new columns
2. Modify `METRIC_CATEGORIES` in `app.py` if needed
3. The dashboard will automatically detect new metrics

### Styling
- Modify the CSS in `app.py` for custom styling
- Update color schemes in `utils/visualization.py`
- Adjust layout and spacing as needed

## Troubleshooting

### Common Issues
1. **Data file not found**: Ensure `data/football_stats.csv` exists
2. **Module not found**: Run `pip install -r requirements.txt`
3. **Chart not displaying**: Check data format and column names
4. **Performance issues**: Reduce data size or increase cache settings

### Support
- Check that all required columns exist in your CSV
- Verify data types are numeric for metric columns
- Ensure team names are unique and properly formatted

## Technology Stack
- **Streamlit**: Web framework for data apps
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing support

---

Made with ❤️ for Indonesian Football Analytics