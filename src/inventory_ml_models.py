import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

print("=== LAHN INC. INVENTORY MANAGEMENT ML MODELS ===")
print("Loading ML libraries for inventory optimization...")

# Basic data loading and exploration
def load_sales_data():
    """Load and prepare sales data for ML analysis"""
    print("Loading sales data...")
    data_path = os.path.join(DATA_DIR, 'sales_data.csv')
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add time-based features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    print(f"Data loaded: {df.shape[0]} records with {df.shape[1]} features")
    return df

# Test the basic loading
if __name__ == "__main__":
    df = load_sales_data()
    print("[OK] Basic data loading successful")

# =============================================================================
# LAHN INC. INVENTORY MANAGEMENT ML MODELS
# =============================================================================

class InventoryMLSuite:
    """
    Comprehensive ML suite for inventory management and stocking decisions
    
    This class provides:
    1. Demand Forecasting
    2. Inventory Optimization
    3. Product Performance Analysis
    4. Seasonal Pattern Analysis
    5. Customer Lifetime Value Prediction
    6. Actionable Inventory Recommendations
    """
    
    def __init__(self):
        """Initialize the ML suite with empty models"""
        self.data = None
        self.demand_model = None
        self.inventory_model = None
        self.seasonal_model = None
        self.clv_model = None
        self.performance_classifier = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath=None):
        """
        Load and prepare data for ML models
        
        Args:
            filepath (str): Path to the sales data CSV file (optional)
            
        Returns:
            pandas.DataFrame: Processed data ready for ML models
        """
        print("=== LOADING DATA FOR ML MODELS ===")
        
        # Use default path if not specified
        if filepath is None:
            filepath = os.path.join(DATA_DIR, 'sales_data.csv')
        
        print(f"Loading data from {filepath}...")
        
        # Load the data
        self.data = pd.read_csv(filepath)
        
        # Convert date column
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Extract additional time features
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
        self.data['DayOfMonth'] = self.data['Date'].dt.day
        self.data['DayOfYear'] = self.data['Date'].dt.dayofyear
        self.data['WeekOfYear'] = self.data['Date'].dt.isocalendar().week
        self.data['Quarter'] = self.data['Date'].dt.quarter
        self.data['IsWeekend'] = self.data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Calculate additional business metrics
        self.data['Revenue_per_Unit'] = self.data['Revenue'] / self.data['Order_Quantity']
        self.data['Profit_per_Unit'] = self.data['Profit'] / self.data['Order_Quantity']
        self.data['Profit_Margin'] = (self.data['Profit'] / self.data['Revenue'] * 100).round(2)
        
        print(f"Data loaded successfully: {self.data.shape[0]} records, {self.data.shape[1]} features")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        return self.data
    
    def prepare_features(self, df):
        """
        Prepare features for ML models
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with encoded features
        """
        # Create a copy to avoid modifying original data
        df_encoded = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['Customer_Gender', 'Age_Group', 'Country', 'State', 
                          'Product_Category', 'Sub_Category', 'Product', 'Month']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def demand_forecasting_model(self):
        """
        Build demand forecasting model to predict future sales
        
        This model predicts:
        - Order quantities for different products
        - Revenue forecasts by category
        - Demand patterns by time period
        
        Returns:
            dict: Model performance metrics and predictions
        """
        print("\n=== DEMAND FORECASTING MODEL ===")
        print("Building demand forecasting model...")
        
        # Prepare aggregated data by product category and time
        daily_demand = self.data.groupby(['Date', 'Product_Category']).agg({
            'Order_Quantity': 'sum',
            'Revenue': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        # Pivot to get separate columns for each product category
        demand_pivot = daily_demand.pivot(index='Date', columns='Product_Category', values='Order_Quantity').fillna(0)
        
        # Create features for time series forecasting
        demand_features = pd.DataFrame(index=demand_pivot.index)
        demand_features['DayOfWeek'] = demand_features.index.dayofweek
        demand_features['DayOfMonth'] = demand_features.index.day
        demand_features['Month'] = demand_features.index.month
        demand_features['Quarter'] = demand_features.index.quarter
        demand_features['WeekOfYear'] = demand_features.index.isocalendar().week
        demand_features['IsWeekend'] = demand_features['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add rolling averages as features
        for col in demand_pivot.columns:
            demand_features[f'{col}_rolling_7'] = demand_pivot[col].rolling(window=7).mean()
            demand_features[f'{col}_rolling_30'] = demand_pivot[col].rolling(window=30).mean()
            demand_features[f'{col}_lag_1'] = demand_pivot[col].shift(1)
            demand_features[f'{col}_lag_7'] = demand_pivot[col].shift(7)
        
        # Remove rows with NaN values
        demand_features = demand_features.dropna()
        demand_pivot = demand_pivot.loc[demand_features.index]
        
        # Build models for each product category
        demand_models = {}
        demand_results = {}
        
        for category in demand_pivot.columns:
            print(f"Training demand model for {category}...")
            
            # Prepare features (exclude current category features to avoid data leakage)
            feature_cols = [col for col in demand_features.columns if not col.startswith(category)]
            X = demand_features[feature_cols]
            y = demand_pivot[category]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            demand_models[category] = model
            demand_results[category] = {
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Feature_Importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            print(f"  - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
        
        self.demand_model = demand_models
        
        # Create demand forecast visualization
        self.visualize_demand_forecast(demand_pivot, demand_results)
        
        return demand_results
    
    def inventory_optimization_model(self):
        """
        Build inventory optimization model to determine optimal stock levels
        
        This model calculates:
        - Optimal stock levels for each product
        - Reorder points based on demand patterns
        - Safety stock requirements
        
        Returns:
            pandas.DataFrame: Inventory optimization recommendations
        """
        print("\n=== INVENTORY OPTIMIZATION MODEL ===")
        print("Building inventory optimization model...")
        
        # Calculate inventory metrics by product
        inventory_metrics = self.data.groupby(['Product_Category', 'Sub_Category', 'Product']).agg({
            'Order_Quantity': ['mean', 'std', 'sum', 'count'],
            'Revenue': ['mean', 'sum'],
            'Profit': ['mean', 'sum'],
            'Unit_Cost': 'mean',
            'Unit_Price': 'mean'
        }).round(2)
        
        # Flatten column names
        inventory_metrics.columns = ['_'.join(col).strip() for col in inventory_metrics.columns]
        inventory_metrics = inventory_metrics.reset_index()
        
        # Calculate demand velocity and variability
        inventory_metrics['Demand_Velocity'] = inventory_metrics['Order_Quantity_sum'] / inventory_metrics['Order_Quantity_count']
        inventory_metrics['Demand_Variability'] = inventory_metrics['Order_Quantity_std'] / inventory_metrics['Order_Quantity_mean']
        inventory_metrics['Demand_Variability'] = inventory_metrics['Demand_Variability'].fillna(0)
        
        # Calculate ABC classification based on revenue
        inventory_metrics['Revenue_Cumsum'] = inventory_metrics['Revenue_sum'].cumsum()
        total_revenue = inventory_metrics['Revenue_sum'].sum()
        inventory_metrics['Revenue_Percentage'] = (inventory_metrics['Revenue_Cumsum'] / total_revenue * 100).round(2)
        
        # ABC Classification
        def classify_abc(percentage):
            if percentage <= 80:
                return 'A'
            elif percentage <= 95:
                return 'B'
            else:
                return 'C'
        
        inventory_metrics['ABC_Classification'] = inventory_metrics['Revenue_Percentage'].apply(classify_abc)
        
        # Calculate optimal stock levels
        # Safety stock = Z-score * std * sqrt(lead_time)
        # Assuming lead time of 7 days and 95% service level (Z=1.645)
        Z_SCORE = 1.645
        LEAD_TIME = 7
        
        inventory_metrics['Safety_Stock'] = (Z_SCORE * inventory_metrics['Order_Quantity_std'] * np.sqrt(LEAD_TIME)).round(0)
        inventory_metrics['Reorder_Point'] = (inventory_metrics['Demand_Velocity'] * LEAD_TIME + inventory_metrics['Safety_Stock']).round(0)
        inventory_metrics['Optimal_Stock_Level'] = (inventory_metrics['Reorder_Point'] * 2).round(0)
        
        # Calculate economic order quantity (EOQ)
        # EOQ = sqrt(2 * D * S / H)
        # Assuming ordering cost = $50, holding cost = 25% of unit cost
        ORDERING_COST = 50
        HOLDING_COST_RATE = 0.25
        
        inventory_metrics['Holding_Cost'] = inventory_metrics['Unit_Cost_mean'] * HOLDING_COST_RATE
        inventory_metrics['EOQ'] = np.sqrt(2 * inventory_metrics['Order_Quantity_sum'] * ORDERING_COST / inventory_metrics['Holding_Cost']).round(0)
        
        # Priority scoring for stocking decisions
        inventory_metrics['Priority_Score'] = (
            inventory_metrics['Profit_sum'] / inventory_metrics['Profit_sum'].max() * 0.4 +
            inventory_metrics['Revenue_sum'] / inventory_metrics['Revenue_sum'].max() * 0.3 +
            inventory_metrics['Demand_Velocity'] / inventory_metrics['Demand_Velocity'].max() * 0.3
        ).round(3)
        
        # Sort by priority score
        inventory_recommendations = inventory_metrics.sort_values('Priority_Score', ascending=False)
        
        # Save recommendations to CSV
        inventory_recommendations.to_csv(os.path.join(REPORTS_DIR, 'inventory_recommendations.csv'), index=False)
        
        print("[OK] Inventory optimization model complete")
        print(f"  - Generated recommendations for {len(inventory_recommendations)} products")
        print("  - Recommendations saved to 'inventory_recommendations.csv'")
        
        return inventory_recommendations
    
    def product_performance_analysis(self):
        """
        Analyze product performance for stocking prioritization
        
        Returns:
            pandas.DataFrame: Product performance classification
        """
        print("\n=== PRODUCT PERFORMANCE ANALYSIS ===")
        print("Analyzing product performance for stocking decisions...")
        
        # Calculate product performance metrics
        performance_metrics = self.data.groupby(['Product_Category', 'Sub_Category', 'Product']).agg({
            'Order_Quantity': ['sum', 'mean', 'count'],
            'Revenue': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Profit_Margin': 'mean'
        }).round(2)
        
        # Flatten column names
        performance_metrics.columns = ['_'.join(col).strip() for col in performance_metrics.columns]
        performance_metrics = performance_metrics.reset_index()
        
        # Calculate performance scores
        performance_metrics['Volume_Score'] = (performance_metrics['Order_Quantity_sum'] / performance_metrics['Order_Quantity_sum'].max()).round(3)
        performance_metrics['Revenue_Score'] = (performance_metrics['Revenue_sum'] / performance_metrics['Revenue_sum'].max()).round(3)
        performance_metrics['Profit_Score'] = (performance_metrics['Profit_sum'] / performance_metrics['Profit_sum'].max()).round(3)
        performance_metrics['Frequency_Score'] = (performance_metrics['Order_Quantity_count'] / performance_metrics['Order_Quantity_count'].max()).round(3)
        
        # Overall performance score
        performance_metrics['Overall_Score'] = (
            performance_metrics['Volume_Score'] * 0.25 +
            performance_metrics['Revenue_Score'] * 0.30 +
            performance_metrics['Profit_Score'] * 0.30 +
            performance_metrics['Frequency_Score'] * 0.15
        ).round(3)
        
        # Classify performance levels
        def classify_performance(score):
            if score >= 0.7:
                return 'High'
            elif score >= 0.3:
                return 'Medium'
            else:
                return 'Low'
        
        performance_metrics['Performance_Level'] = performance_metrics['Overall_Score'].apply(classify_performance)
        
        # Sort by overall score
        performance_metrics = performance_metrics.sort_values('Overall_Score', ascending=False)
        
        print(f"Performance classification:")
        print(performance_metrics['Performance_Level'].value_counts())
        
        return performance_metrics
    
    def seasonal_pattern_analysis(self):
        """
        Analyze seasonal patterns for timing inventory decisions
        
        Returns:
            dict: Seasonal analysis results
        """
        print("\n=== SEASONAL PATTERN ANALYSIS ===")
        print("Analyzing seasonal patterns for inventory timing...")
        
        # Monthly patterns
        monthly_patterns = self.data.groupby(['Month', 'Product_Category']).agg({
            'Order_Quantity': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        monthly_pivot = monthly_patterns.pivot(index='Month', columns='Product_Category', values='Order_Quantity')
        
        # Calculate seasonal indices
        seasonal_indices = {}
        for category in monthly_pivot.columns:
            monthly_avg = monthly_pivot[category].mean()
            seasonal_indices[category] = (monthly_pivot[category] / monthly_avg).round(2).to_dict()
        
        # Day of week patterns
        dow_patterns = self.data.groupby(['DayOfWeek', 'Product_Category']).agg({
            'Order_Quantity': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        # Quarter patterns
        quarter_patterns = self.data.groupby(['Quarter', 'Product_Category']).agg({
            'Order_Quantity': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        seasonal_results = {
            'monthly_patterns': monthly_patterns,
            'seasonal_indices': seasonal_indices,
            'dow_patterns': dow_patterns,
            'quarter_patterns': quarter_patterns
        }
        
        # Create seasonal visualizations
        self.visualize_seasonal_patterns(seasonal_results)
        
        return seasonal_results
    
    def customer_lifetime_value_model(self):
        """
        Build customer lifetime value model for targeted inventory decisions
        
        Returns:
            pandas.DataFrame: Customer CLV predictions
        """
        print("\n=== CUSTOMER LIFETIME VALUE MODEL ===")
        print("Building CLV model for targeted inventory decisions...")
        
        # Calculate customer-level metrics
        customer_metrics = self.data.groupby(['Customer_Age', 'Age_Group', 'Customer_Gender']).agg({
            'Order_Quantity': ['sum', 'mean', 'count'],
            'Revenue': ['sum', 'mean'],
            'Profit': ['sum', 'mean'],
            'Date': ['min', 'max']
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
        customer_metrics = customer_metrics.reset_index()
        
        # Calculate customer lifetime metrics
        customer_metrics['Customer_Lifetime_Days'] = (customer_metrics['Date_max'] - customer_metrics['Date_min']).dt.days
        customer_metrics['Purchase_Frequency'] = customer_metrics['Order_Quantity_count'] / (customer_metrics['Customer_Lifetime_Days'] / 365.25)
        customer_metrics['Average_Order_Value'] = customer_metrics['Revenue_sum'] / customer_metrics['Order_Quantity_count']
        
        # Predict CLV (simplified model)
        customer_metrics['Predicted_CLV'] = (
            customer_metrics['Average_Order_Value'] * 
            customer_metrics['Purchase_Frequency'] * 
            2  # Assuming 2 year prediction horizon
        ).round(2)
        
        # Segment customers by CLV
        customer_metrics['CLV_Percentile'] = customer_metrics['Predicted_CLV'].rank(pct=True)
        
        def classify_clv(percentile):
            if percentile >= 0.8:
                return 'High_Value'
            elif percentile >= 0.5:
                return 'Medium_Value'
            else:
                return 'Low_Value'
        
        customer_metrics['CLV_Segment'] = customer_metrics['CLV_Percentile'].apply(classify_clv)
        
        print(f"CLV segments:")
        print(customer_metrics['CLV_Segment'].value_counts())
        
        return customer_metrics
    
    def visualize_demand_forecast(self, demand_pivot, demand_results):
        """Create demand forecasting visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot demand trends for each category
        for i, category in enumerate(demand_pivot.columns):
            row = i // 2
            col = i % 2
            if row < 2 and col < 2:
                demand_pivot[category].plot(ax=axes[row, col], title=f'{category} Demand Trend')
                axes[row, col].set_xlabel('Date')
                axes[row, col].set_ylabel('Order Quantity')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'demand_forecasting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance plot
        fig, ax = plt.subplots(figsize=(12, 8))
        importance_data = []
        for category, results in demand_results.items():
            for feature, importance in results['Feature_Importance'].items():
                importance_data.append({
                    'Category': category,
                    'Feature': feature,
                    'Importance': importance
                })
        
        importance_df = pd.DataFrame(importance_data)
        importance_pivot = importance_df.pivot(index='Feature', columns='Category', values='Importance')
        importance_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Feature Importance in Demand Forecasting')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'demand_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_seasonal_patterns(self, seasonal_results):
        """Create seasonal pattern visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Monthly patterns
        monthly_pivot = seasonal_results['monthly_patterns'].pivot(index='Month', columns='Product_Category', values='Order_Quantity')
        monthly_pivot.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Monthly Demand Patterns by Category')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Order Quantity')
        
        # Day of week patterns
        dow_pivot = seasonal_results['dow_patterns'].pivot(index='DayOfWeek', columns='Product_Category', values='Order_Quantity')
        dow_pivot.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Day of Week Patterns by Category')
        axes[0, 1].set_xlabel('Day of Week (0=Monday)')
        axes[0, 1].set_ylabel('Order Quantity')
        
        # Quarter patterns
        quarter_pivot = seasonal_results['quarter_patterns'].pivot(index='Quarter', columns='Product_Category', values='Order_Quantity')
        quarter_pivot.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Quarterly Patterns by Category')
        axes[1, 0].set_xlabel('Quarter')
        axes[1, 0].set_ylabel('Order Quantity')
        
        # Seasonal indices heatmap
        seasonal_df = pd.DataFrame(seasonal_results['seasonal_indices'])
        sns.heatmap(seasonal_df, annot=True, cmap='RdYlBu_r', ax=axes[1, 1])
        axes[1, 1].set_title('Seasonal Indices by Month and Category')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'seasonal_patterns_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_inventory_insights(self, inventory_recommendations, performance_metrics, seasonal_results, customer_metrics):
        """
        Generate comprehensive inventory management insights
        
        Returns:
            dict: Comprehensive business insights and recommendations
        """
        print("\n=== COMPREHENSIVE INVENTORY INSIGHTS ===")
        print("Generating actionable inventory management recommendations...")
        
        insights = {
            'executive_summary': {},
            'stocking_recommendations': {},
            'seasonal_insights': {},
            'customer_insights': {},
            'risk_analysis': {}
        }
        
        # Executive Summary
        insights['executive_summary'] = {
            'total_products': len(inventory_recommendations),
            'high_priority_products': len(inventory_recommendations[inventory_recommendations['Priority_Score'] >= 0.7]),
            'total_investment_needed': inventory_recommendations['Optimal_Stock_Level'].sum() * inventory_recommendations['Unit_Cost_mean'].mean(),
            'expected_revenue_potential': inventory_recommendations['Revenue_sum'].sum(),
            'top_performing_category': performance_metrics.groupby('Product_Category')['Overall_Score'].mean().idxmax()
        }
        
        # Stocking Recommendations
        high_priority = inventory_recommendations[inventory_recommendations['Priority_Score'] >= 0.7]
        insights['stocking_recommendations'] = {
            'must_stock_products': high_priority[['Product_Category', 'Sub_Category', 'Product', 'Optimal_Stock_Level', 'Priority_Score']].to_dict('records'),
            'reorder_alerts': inventory_recommendations[inventory_recommendations['Reorder_Point'] > 0][['Product_Category', 'Sub_Category', 'Product', 'Reorder_Point']].to_dict('records'),
            'abc_distribution': inventory_recommendations['ABC_Classification'].value_counts().to_dict()
        }
        
        # Seasonal Insights
        seasonal_indices = seasonal_results['seasonal_indices']
        insights['seasonal_insights'] = {
            'peak_months': {},
            'low_months': {},
            'seasonal_recommendations': []
        }
        
        for category, indices in seasonal_indices.items():
            peak_month = max(indices, key=indices.get)
            low_month = min(indices, key=indices.get)
            insights['seasonal_insights']['peak_months'][category] = peak_month
            insights['seasonal_insights']['low_months'][category] = low_month
            
            insights['seasonal_insights']['seasonal_recommendations'].append({
                'category': category,
                'recommendation': f"Stock up on {category} before {peak_month}, reduce inventory after {low_month}",
                'peak_index': indices[peak_month],
                'low_index': indices[low_month]
            })
        
        # Customer Insights
        high_value_customers = customer_metrics[customer_metrics['CLV_Segment'] == 'High_Value']
        insights['customer_insights'] = {
            'high_value_customer_preferences': high_value_customers.groupby('Age_Group')['Predicted_CLV'].mean().to_dict(),
            'target_demographics': high_value_customers['Age_Group'].mode().iloc[0] if not high_value_customers.empty else 'Adults (35-64)',
            'clv_segments': customer_metrics['CLV_Segment'].value_counts().to_dict()
        }
        
        # Risk Analysis
        high_variability_products = inventory_recommendations[inventory_recommendations['Demand_Variability'] > 1]
        insights['risk_analysis'] = {
            'high_variability_products': len(high_variability_products),
            'inventory_investment_risk': high_variability_products['Optimal_Stock_Level'].sum() * high_variability_products['Unit_Cost_mean'].mean(),
            'recommended_safety_stock': inventory_recommendations['Safety_Stock'].sum()
        }
        
        return insights
    
    def run_complete_analysis(self):
        """
        Run the complete ML analysis for inventory management
        
        Returns:
            dict: Complete analysis results
        """
        print("=== LAHN INC. INVENTORY MANAGEMENT ML ANALYSIS ===")
        print("Running complete ML analysis for inventory optimization...")
        
        # Load data
        self.load_data()
        
        # Run all ML models
        demand_results = self.demand_forecasting_model()
        inventory_recommendations = self.inventory_optimization_model()
        performance_metrics = self.product_performance_analysis()
        seasonal_results = self.seasonal_pattern_analysis()
        customer_metrics = self.customer_lifetime_value_model()
        
        # Generate comprehensive insights
        insights = self.generate_inventory_insights(
            inventory_recommendations, 
            performance_metrics, 
            seasonal_results, 
            customer_metrics
        )
        
        # Print executive summary
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY - INVENTORY MANAGEMENT INSIGHTS")
        print("="*60)
        print(f"Total Products Analyzed: {insights['executive_summary']['total_products']}")
        print(f"High Priority Products: {insights['executive_summary']['high_priority_products']}")
        print(f"Total Investment Needed: ${insights['executive_summary']['total_investment_needed']:,.2f}")
        print(f"Expected Revenue Potential: ${insights['executive_summary']['expected_revenue_potential']:,.2f}")
        print(f"Top Performing Category: {insights['executive_summary']['top_performing_category']}")
        
        # Print top recommendations
        print("\n" + "="*60)
        print("TOP INVENTORY RECOMMENDATIONS")
        print("="*60)
        for i, product in enumerate(insights['stocking_recommendations']['must_stock_products'][:5]):
            print(f"{i+1}. {product['Product_Category']} - {product['Sub_Category']}")
            print(f"   Optimal Stock Level: {product['Optimal_Stock_Level']} units")
            print(f"   Priority Score: {product['Priority_Score']}")
        
        # Print seasonal insights
        print("\n" + "="*60)
        print("SEASONAL STOCKING INSIGHTS")
        print("="*60)
        for rec in insights['seasonal_insights']['seasonal_recommendations']:
            print(f"• {rec['recommendation']}")
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Generated files:")
        print("- inventory_recommendations.csv: Detailed product recommendations")
        print("- demand_forecasting_analysis.png: Demand forecasting visualizations")
        print("- demand_feature_importance.png: Feature importance analysis")
        print("- seasonal_patterns_analysis.png: Seasonal pattern visualizations")
        
        return {
            'demand_results': demand_results,
            'inventory_recommendations': inventory_recommendations,
            'performance_metrics': performance_metrics,
            'seasonal_results': seasonal_results,
            'customer_metrics': customer_metrics,
            'insights': insights
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Initialize the ML suite
    ml_suite = InventoryMLSuite()
    
    # Run complete analysis
    results = ml_suite.run_complete_analysis()
    
    print("\nLAHN INC. INVENTORY MANAGEMENT RECOMMENDATIONS:")
    print("Use the generated insights to make data-driven inventory decisions!")
    print("Review the CSV file and visualizations for detailed recommendations.") 