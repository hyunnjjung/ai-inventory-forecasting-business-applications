import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Define paths for the new project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

# Import our time series forecaster
from time_series_forecasting import TimeSeriesInventoryForecaster

# Page configuration
st.set_page_config(
    page_title="Lahn Inc. Inventory Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    body, .stApp {
        background-color: #f0f2f6 !important;
        color: #222 !important;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .forecast-header {
        color: #1f77b4;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_forecaster():
    """Load and run the time series forecaster"""
    forecaster = TimeSeriesInventoryForecaster()
    forecaster.load_and_prepare_data()
    forecaster.build_prophet_models()
    forecaster.build_arima_models()
    forecaster.seasonal_decomposition_analysis()
    forecaster.generate_inventory_recommendations()
    return forecaster

@st.cache_data
def load_recommendations():
    """Load the recommendations CSV"""
    try:
        return pd.read_csv(os.path.join(REPORTS_DIR, 'time_series_inventory_recommendations.csv'))
    except:
        return None

def main():
    # Title and header
    st.title("🏪 Lahn Inc. Inventory Forecasting Dashboard")
    st.markdown("### Advanced Time Series Predictions for Smart Inventory Management")
    
    # Sidebar
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading forecasting models..."):
        forecaster = load_forecaster()
        recommendations = load_recommendations()
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Dashboard Section",
        ["Executive Summary", "Demand Forecasting", "Inventory Recommendations", "Seasonal Analysis", "Model Performance"]
    )
    
    if page == "Executive Summary":
        show_executive_summary(forecaster, recommendations)
    elif page == "Demand Forecasting":
        show_demand_forecasting(forecaster)
    elif page == "Inventory Recommendations":
        show_inventory_recommendations(forecaster, recommendations)
    elif page == "Seasonal Analysis":
        show_seasonal_analysis(forecaster)
    elif page == "Model Performance":
        show_model_performance(forecaster)

def show_executive_summary(forecaster, recommendations):
    """Display executive summary page"""
    st.markdown('<p class="forecast-header">📊 Executive Summary</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if recommendations is not None:
        total_stock_needed = recommendations['Optimal_Stock_Level'].sum()
        total_daily_demand = recommendations['Avg_Daily_Demand'].sum()
        highest_demand_category = recommendations.loc[recommendations['Avg_Daily_Demand'].idxmax(), 'Product_Category']
        avg_confidence = "High" if len(recommendations[recommendations['Forecast_Confidence'] == 'High']) >= 2 else "Medium"
        
        with col1:
            st.metric(
                label="Total Daily Demand",
                value=f"{total_daily_demand:.0f} units",
                delta="Across all categories"
            )
        
        with col2:
            st.metric(
                label="Total Stock Needed",
                value=f"{total_stock_needed:,.0f} units",
                delta="Optimal inventory level"
            )
        
        with col3:
            st.metric(
                label="Highest Demand Category",
                value=highest_demand_category,
                delta=f"{recommendations.loc[recommendations['Avg_Daily_Demand'].idxmax(), 'Avg_Daily_Demand']:.0f} units/day"
            )
        
        with col4:
            st.metric(
                label="Forecast Confidence",
                value=avg_confidence,
                delta="Overall model reliability"
            )
    
    # Summary insights
    st.markdown("---")
    st.subheader("🎯 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Demand Patterns:**
        - Accessories show highest daily demand (1,584 units)
        - Clothing has moderate demand (393 units)
        - Bikes have lowest but high-value demand (50 units)
        
        **🔄 Inventory Strategy:**
        - Accessories: High-volume, frequent restocking
        - Clothing: Seasonal planning required
        - Bikes: Low-volume, high-value management
        """)
    
    with col2:
        st.markdown("""
        **⚠️ Critical Actions:**
        - Monitor Accessories inventory closely
        - Implement automated reorder for high-demand items
        - Consider seasonal variations in stocking
        
        **💰 Financial Impact:**
        - Optimize cash flow with demand-based ordering
        - Reduce carrying costs through precise forecasting
        - Minimize stockouts with safety stock calculations
        """)
    
    # Recommendations summary chart
    if recommendations is not None:
        st.markdown("---")
        st.subheader("📊 Category Comparison")
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Daily Demand', 'Optimal Stock Level', 'Reorder Point'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        categories = recommendations['Product_Category']
        
        fig.add_trace(
            go.Bar(x=categories, y=recommendations['Avg_Daily_Demand'], name='Daily Demand'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=categories, y=recommendations['Optimal_Stock_Level'], name='Optimal Stock'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=categories, y=recommendations['Reorder_Point'], name='Reorder Point'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_demand_forecasting(forecaster):
    """Display demand forecasting page"""
    st.markdown('<p class="forecast-header">📈 Demand Forecasting</p>', unsafe_allow_html=True)
    
    # Category selection
    categories = list(forecaster.category_data.keys())
    selected_category = st.selectbox("Select Product Category", categories)
    
    # Forecast parameters
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Days", min_value=30, max_value=180, value=90)
    with col2:
        show_components = st.checkbox("Show Forecast Components", value=False)
    
    # Get data for selected category
    historical_data = forecaster.category_data[selected_category]
    prophet_forecast = forecaster.forecasts.get(f'{selected_category}_prophet')
    
    if prophet_forecast is not None:
        # Main forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=historical_data['Date'],
                y=historical_data['Order_Quantity'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue', width=2)
            )
        )
        
        # Future predictions
        future_data = prophet_forecast[prophet_forecast['ds'] > historical_data['Date'].max()]
        future_data = future_data.head(forecast_days)
        
        fig.add_trace(
            go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            )
        )
        
        fig.update_layout(
            title=f'{selected_category} Demand Forecast ({forecast_days} days)',
            xaxis_title='Date',
            yaxis_title='Order Quantity',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast components
        if show_components:
            st.subheader("🔍 Forecast Components")
            
            components_fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Trend', 'Yearly Seasonality', 'Weekly Seasonality'),
                vertical_spacing=0.1
            )
            
            components_fig.add_trace(
                go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['trend'], name='Trend'),
                row=1, col=1
            )
            
            components_fig.add_trace(
                go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yearly'], name='Yearly'),
                row=2, col=1
            )
            
            components_fig.add_trace(
                go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['weekly'], name='Weekly'),
                row=3, col=1
            )
            
            components_fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(components_fig, use_container_width=True)
        
        # Forecast statistics
        st.subheader("📊 Forecast Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_forecast = future_data['yhat'].mean()
            st.metric("Average Daily Demand", f"{avg_forecast:.1f}", "units/day")
        
        with col2:
            max_forecast = future_data['yhat'].max()
            st.metric("Peak Demand", f"{max_forecast:.1f}", "units")
        
        with col3:
            min_forecast = future_data['yhat'].min()
            st.metric("Minimum Demand", f"{min_forecast:.1f}", "units")
        
        with col4:
            volatility = future_data['yhat'].std()
            st.metric("Demand Volatility", f"{volatility:.1f}", "std dev")

def show_inventory_recommendations(forecaster, recommendations):
    """Display inventory recommendations page"""
    st.markdown('<p class="forecast-header">📦 Inventory Recommendations</p>', unsafe_allow_html=True)
    
    if recommendations is not None:
        # Recommendations table
        st.subheader("📋 Detailed Recommendations")
        
        # Format the dataframe for better display
        display_df = recommendations.copy()
        display_df['Avg_Daily_Demand'] = display_df['Avg_Daily_Demand'].round(1)
        display_df['Optimal_Stock_Level'] = display_df['Optimal_Stock_Level'].astype(int)
        display_df['Reorder_Point'] = display_df['Reorder_Point'].astype(int)
        display_df['Safety_Stock'] = display_df['Safety_Stock'].astype(int)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for recommendations
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations",
            data=csv,
            file_name=f"inventory_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Category-specific recommendations
        st.markdown("---")
        st.subheader("🎯 Category-Specific Actions")
        
        for _, row in recommendations.iterrows():
            category = row['Product_Category']
            
            with st.expander(f"📊 {category} Strategy"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Daily Demand", f"{row['Avg_Daily_Demand']:.1f} units")
                    st.metric("Optimal Stock", f"{row['Optimal_Stock_Level']:,.0f} units")
                    st.metric("Reorder Point", f"{row['Reorder_Point']:,.0f} units")
                
                with col2:
                    st.metric("Safety Stock", f"{row['Safety_Stock']:,.0f} units")
                    st.metric("Demand Volatility", f"{row['Demand_Volatility']:.2f}")
                    st.metric("Forecast Confidence", row['Forecast_Confidence'])
                
                # Strategic recommendations
                if category == "Accessories":
                    st.info("💡 **Strategy**: High-frequency ordering with automated restock triggers. Monitor daily for stockouts.")
                elif category == "Bikes":
                    st.info("💡 **Strategy**: Low-volume, high-value management. Focus on just-in-time delivery to minimize carrying costs.")
                elif category == "Clothing":
                    st.info("💡 **Strategy**: Seasonal planning essential. Build inventory before peak seasons, clear during low periods.")
        
        # Investment summary
        st.markdown("---")
        st.subheader("💰 Investment Summary")
        
        # Assuming average unit costs (you can adjust these based on actual data)
        unit_costs = {'Accessories': 50, 'Bikes': 1500, 'Clothing': 40}
        
        investment_data = []
        for _, row in recommendations.iterrows():
            category = row['Product_Category']
            cost = unit_costs.get(category, 100)
            investment = row['Optimal_Stock_Level'] * cost
            investment_data.append({
                'Category': category,
                'Stock_Level': row['Optimal_Stock_Level'],
                'Est_Unit_Cost': cost,
                'Total_Investment': investment
            })
        
        investment_df = pd.DataFrame(investment_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                investment_df, 
                values='Total_Investment', 
                names='Category',
                title='Investment Distribution by Category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total_investment = investment_df['Total_Investment'].sum()
            st.metric("Total Investment Needed", f"${total_investment:,.0f}")
            
            for _, row in investment_df.iterrows():
                st.metric(
                    f"{row['Category']} Investment",
                    f"${row['Total_Investment']:,.0f}",
                    f"{row['Stock_Level']:,.0f} units"
                )

def show_seasonal_analysis(forecaster):
    """Display seasonal analysis page"""
    st.markdown('<p class="forecast-header">📅 Seasonal Analysis</p>', unsafe_allow_html=True)
    
    # Category selection for seasonal analysis
    categories = list(forecaster.decomposition_results.keys())
    selected_category = st.selectbox("Select Category for Seasonal Analysis", categories)
    
    decomposition = forecaster.decomposition_results[selected_category]
    
    # Seasonal decomposition plot
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original Data', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Original'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=False, title=f'{selected_category} Seasonal Decomposition')
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal insights
    st.subheader("🔍 Seasonal Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trend_slope = np.polyfit(range(len(decomposition.trend.dropna())), decomposition.trend.dropna(), 1)[0]
        st.metric("Trend Direction", "Increasing" if trend_slope > 0 else "Decreasing", f"{trend_slope:.4f}")
        
        seasonality_strength = decomposition.seasonal.std()
        st.metric("Seasonality Strength", f"{seasonality_strength:.2f}", "Standard deviation")
    
    with col2:
        peak_season = decomposition.seasonal.idxmax().strftime('%B')
        st.metric("Peak Season", peak_season)
        
        low_season = decomposition.seasonal.idxmin().strftime('%B')
        st.metric("Low Season", low_season)

def show_model_performance(forecaster):
    """Display model performance page"""
    st.markdown('<p class="forecast-header">⚡ Model Performance</p>', unsafe_allow_html=True)
    
    # Performance metrics
    st.subheader("📊 Prophet Model Performance")
    
    performance_data = []
    for model_name, metrics in forecaster.performance_metrics.items():
        category = model_name.replace('_prophet', '')
        performance_data.append({
            'Category': category,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE']
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(performance_df, use_container_width=True)
    
    with col2:
        fig = px.bar(performance_df, x='Category', y='MAPE', title='Model Accuracy (Lower MAPE = Better)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model interpretation
    st.subheader("🔍 Model Interpretation")
    
    st.markdown("""
    **Performance Metrics Explained:**
    - **RMSE (Root Mean Square Error)**: Average prediction error in units
    - **MAE (Mean Absolute Error)**: Average absolute prediction error
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage error (lower is better)
    
    **Model Quality:**
    - MAPE < 10%: Excellent forecast accuracy
    - MAPE 10-20%: Good forecast accuracy  
    - MAPE 20-50%: Reasonable forecast accuracy
    - MAPE > 50%: Poor forecast accuracy
    """)
    
    # Feature importance (if available)
    st.subheader("🎯 Key Success Factors")
    st.markdown("""
    **Strong Seasonal Patterns**: All categories show clear seasonal trends that improve predictability
    
    **Stable Trends**: Long-term trends are consistent, making future predictions reliable
    
    **Weekly Patterns**: Day-of-week effects help predict short-term demand variations
    
    **Model Robustness**: Multiple validation metrics confirm forecast reliability
    """)

if __name__ == "__main__":
    main() 