import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Time series specific imports
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Plotting
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Define paths for the new project structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models')

# Ensure output directories exist
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("=== LAHN INC. TIME SERIES FORECASTING MODEL ===")
print("Building advanced time series models for inventory predictions...")

class TimeSeriesInventoryForecaster:
    """
    Advanced Time Series Forecasting for Inventory Management
    
    This class implements multiple time series models:
    1. Prophet - For seasonal patterns and trend analysis
    2. ARIMA - For traditional time series forecasting
    3. Seasonal Decomposition - For understanding patterns
    4. Ensemble Methods - Combining multiple models
    """
    
    def __init__(self):
        self.data = None
        self.prophet_models = {}
        self.arima_models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
    def load_and_prepare_data(self, filepath=None):
        """Load and prepare time series data"""
        print("\n=== LOADING TIME SERIES DATA ===")
        
        # Use default path if not specified
        if filepath is None:
            filepath = os.path.join(DATA_DIR, 'sales_data.csv')
        
        # Load data
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create daily aggregated data by product category
        daily_sales = df.groupby(['Date', 'Product_Category']).agg({
            'Order_Quantity': 'sum',
            'Revenue': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        # Create separate time series for each product category
        categories = daily_sales['Product_Category'].unique()
        self.category_data = {}
        
        for category in categories:
            category_data = daily_sales[daily_sales['Product_Category'] == category]
            
            # Create complete date range
            date_range = pd.date_range(start=category_data['Date'].min(), 
                                     end=category_data['Date'].max(), 
                                     freq='D')
            
            # Reindex to fill missing dates
            category_ts = category_data.set_index('Date').reindex(date_range, fill_value=0)
            category_ts.reset_index(inplace=True)
            category_ts.rename(columns={'index': 'Date'}, inplace=True)
            
            # Add category column back
            category_ts['Product_Category'] = category
            
            self.category_data[category] = category_ts
        
        print(f"[OK] Data prepared for {len(categories)} product categories")
        print(f"Date range: {daily_sales['Date'].min()} to {daily_sales['Date'].max()}")
        
        return self.category_data
    
    def build_prophet_models(self, forecast_days=90):
        """Build Prophet models for each product category"""
        print("\n=== BUILDING PROPHET MODELS ===")
        
        for category, data in self.category_data.items():
            print(f"Training Prophet model for {category}...")
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_data = data[['Date', 'Order_Quantity']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize Prophet with seasonality components
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
            
            # Fit the model
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_days)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Store model and forecast
            self.prophet_models[category] = model
            self.forecasts[f'{category}_prophet'] = forecast
            
            # Calculate performance metrics
            actual = prophet_data['y']
            predicted = forecast['yhat'][:len(actual)]
            
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            
            self.performance_metrics[f'{category}_prophet'] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': np.mean(np.abs((actual - predicted) / actual)) * 100
            }
            
            print(f"  [OK] Model trained - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    def build_arima_models(self, forecast_days=90):
        """Build ARIMA models for each product category"""
        print("\n=== BUILDING ARIMA MODELS ===")
        
        for category, data in self.category_data.items():
            print(f"Training ARIMA model for {category}...")
            
            # Prepare time series data
            ts_data = data.set_index('Date')['Order_Quantity']
            
            # Check stationarity
            adf_result = adfuller(ts_data)
            is_stationary = adf_result[1] < 0.05
            
            if not is_stationary:
                # Difference the series to make it stationary
                ts_data = ts_data.diff().dropna()
                print(f"  - Applied differencing to achieve stationarity")
            
            # Fit ARIMA model (using auto-selection for now)
            # For production, you'd want to use auto_arima from pmdarima
            try:
                model = ARIMA(ts_data, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Make forecast
                forecast = fitted_model.forecast(steps=forecast_days)
                forecast_index = pd.date_range(start=data['Date'].max() + timedelta(days=1), 
                                             periods=forecast_days, freq='D')
                
                # Store results
                self.arima_models[category] = fitted_model
                self.forecasts[f'{category}_arima'] = pd.DataFrame({
                    'Date': forecast_index,
                    'Forecast': forecast
                })
                
                print(f"  [OK] ARIMA model trained successfully")
                
            except Exception as e:
                print(f"  ⚠ ARIMA model failed for {category}: {str(e)}")
    
    def seasonal_decomposition_analysis(self):
        """Perform seasonal decomposition analysis"""
        print("\n=== SEASONAL DECOMPOSITION ANALYSIS ===")
        
        self.decomposition_results = {}
        
        for category, data in self.category_data.items():
            print(f"Analyzing seasonal patterns for {category}...")
            
            # Set date as index
            ts_data = data.set_index('Date')['Order_Quantity']
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=30)
            
            self.decomposition_results[category] = decomposition
            
            # Extract key insights
            trend_slope = np.polyfit(range(len(decomposition.trend.dropna())), 
                                   decomposition.trend.dropna(), 1)[0]
            seasonality_strength = decomposition.seasonal.std()
            
            print(f"  - Trend slope: {trend_slope:.4f}")
            print(f"  - Seasonality strength: {seasonality_strength:.2f}")
    
    def generate_inventory_recommendations(self, safety_stock_factor=1.5):
        """Generate inventory recommendations based on forecasts"""
        print("\n=== GENERATING INVENTORY RECOMMENDATIONS ===")
        
        recommendations = {}
        
        for category in self.category_data.keys():
            prophet_forecast = self.forecasts.get(f'{category}_prophet')
            
            if prophet_forecast is not None:
                # Get future predictions
                future_predictions = prophet_forecast[prophet_forecast['ds'] > self.category_data[category]['Date'].max()]
                
                # Calculate recommendation metrics
                avg_daily_demand = future_predictions['yhat'].mean()
                max_daily_demand = future_predictions['yhat'].max()
                min_daily_demand = future_predictions['yhat'].min()
                demand_volatility = future_predictions['yhat'].std()
                
                # Calculate inventory recommendations
                safety_stock = demand_volatility * safety_stock_factor
                reorder_point = avg_daily_demand * 7 + safety_stock  # 7-day lead time
                optimal_stock = avg_daily_demand * 30 + safety_stock  # 30-day supply
                
                recommendations[category] = {
                    'avg_daily_demand': avg_daily_demand,
                    'max_daily_demand': max_daily_demand,
                    'min_daily_demand': min_daily_demand,
                    'demand_volatility': demand_volatility,
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'optimal_stock_level': optimal_stock,
                    'forecast_confidence': 'High' if demand_volatility < avg_daily_demand * 0.5 else 'Medium'
                }
                
                print(f"{category}:")
                print(f"  - Average daily demand: {avg_daily_demand:.1f} units")
                print(f"  - Optimal stock level: {optimal_stock:.0f} units")
                print(f"  - Reorder point: {reorder_point:.0f} units")
                print(f"  - Safety stock: {safety_stock:.0f} units")
        
        self.inventory_recommendations = recommendations
        return recommendations
    
    def create_forecast_visualizations(self):
        """Create comprehensive forecast visualizations"""
        print("\n=== CREATING FORECAST VISUALIZATIONS ===")
        
        # Create subplots for each category
        fig = make_subplots(
            rows=len(self.category_data), 
            cols=1,
            subplot_titles=[f"{category} Demand Forecast" for category in self.category_data.keys()],
            vertical_spacing=0.1
        )
        
        for i, (category, data) in enumerate(self.category_data.items(), 1):
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=data['Date'],
                    y=data['Order_Quantity'],
                    mode='lines',
                    name=f'{category} Historical',
                    line=dict(color='blue', width=2)
                ),
                row=i, col=1
            )
            
            # Prophet forecast
            prophet_forecast = self.forecasts.get(f'{category}_prophet')
            if prophet_forecast is not None:
                future_data = prophet_forecast[prophet_forecast['ds'] > data['Date'].max()]
                
                fig.add_trace(
                    go.Scatter(
                        x=future_data['ds'],
                        y=future_data['yhat'],
                        mode='lines',
                        name=f'{category} Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=i, col=1
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
                    ),
                    row=i, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=future_data['ds'],
                        y=future_data['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        name=f'{category} Confidence Interval',
                        hoverinfo='skip'
                    ),
                    row=i, col=1
                )
        
        fig.update_layout(
            title="Time Series Demand Forecasting - All Categories",
            height=300 * len(self.category_data),
            showlegend=True
        )
        
        fig.write_html('time_series_forecast_dashboard.html')
        print("[OK] Interactive forecast dashboard saved as 'time_series_forecast_dashboard.html'")
        
        return fig
    
    def create_recommendations_summary(self):
        """Create a summary of recommendations"""
        print("\n=== CREATING RECOMMENDATIONS SUMMARY ===")
        
        if not hasattr(self, 'inventory_recommendations'):
            self.generate_inventory_recommendations()
        
        # Create summary DataFrame
        summary_data = []
        for category, recs in self.inventory_recommendations.items():
            summary_data.append({
                'Product_Category': category,
                'Avg_Daily_Demand': round(recs['avg_daily_demand'], 1),
                'Optimal_Stock_Level': round(recs['optimal_stock_level'], 0),
                'Reorder_Point': round(recs['reorder_point'], 0),
                'Safety_Stock': round(recs['safety_stock'], 0),
                'Demand_Volatility': round(recs['demand_volatility'], 2),
                'Forecast_Confidence': recs['forecast_confidence']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(REPORTS_DIR, 'time_series_inventory_recommendations.csv'), index=False)
        
        print("[OK] Recommendations summary saved as 'time_series_inventory_recommendations.csv'")
        return summary_df
    
    def run_complete_time_series_analysis(self):
        """Run the complete time series analysis"""
        print("=== RUNNING COMPLETE TIME SERIES ANALYSIS ===")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Build models
        self.build_prophet_models()
        self.build_arima_models()
        
        # Seasonal analysis
        self.seasonal_decomposition_analysis()
        
        # Generate recommendations
        self.generate_inventory_recommendations()
        
        # Create visualizations
        self.create_forecast_visualizations()
        
        # Create summary
        summary = self.create_recommendations_summary()
        
        print("\n" + "="*60)
        print("TIME SERIES ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated Files:")
        print("- time_series_forecast_dashboard.html: Interactive forecast dashboard")
        print("- time_series_inventory_recommendations.csv: Detailed recommendations")
        print("\nKey Insights:")
        
        for category, recs in self.inventory_recommendations.items():
            print(f"\n{category}:")
            print(f"  Average daily demand: {recs['avg_daily_demand']:.1f} units")
            print(f"  Optimal stock level: {recs['optimal_stock_level']:.0f} units")
            print(f"  Reorder point: {recs['reorder_point']:.0f} units")
            print(f"  Safety stock: {recs['safety_stock']:.0f} units")
            print(f"  Forecast confidence: {recs['forecast_confidence']}")
        
        return summary

# Main execution
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = TimeSeriesInventoryForecaster()
    
    # Run complete analysis
    results = forecaster.run_complete_time_series_analysis()
    
    print("\nTIME SERIES FORECASTING COMPLETE!")
    print("Open 'time_series_forecast_dashboard.html' in your browser to view interactive predictions!") 