# 🏪 Lahn Inc. Business Intelligence System

A comprehensive business intelligence and inventory management system for Lahn Inc., featuring advanced data analysis, machine learning models, and time series forecasting for optimal inventory decisions.

## 🎯 Project Overview

This system provides end-to-end business intelligence capabilities including:
- **Data Analysis**: Comprehensive sales data analysis with visualizations
- **ML Models**: Advanced machine learning for inventory optimization
- **Time Series Forecasting**: Prophet and ARIMA models for demand prediction
- **Interactive Dashboard**: Real-time Streamlit dashboard for monitoring

## 📁 Project Structure

```
Milestone3/
├── 📄 run_analysis.py          # Main orchestrator script
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project documentation
├── 📂 src/                    # Source code
│   ├── 📄 main.py             # Core data analysis
│   ├── 📄 inventory_ml_models.py  # ML inventory models
│   ├── 📄 time_series_forecasting.py  # Time series models
│   └── 📄 inventory_dashboard.py      # Streamlit dashboard
├── 📂 data/                   # Data files
│   └── 📄 sales_data.csv      # Sales transaction data
├── 📂 outputs/                # Generated outputs
│   ├── 📂 reports/            # CSV reports and summaries
│   ├── 📂 visualizations/     # PNG charts and graphs
│   └── 📂 models/             # Saved ML models
├── 📂 notebooks/              # Jupyter notebooks
│   └── 📄 data-analysis.ipynb # Interactive analysis
└── 📂 docs/                   # Documentation
    ├── 📄 TIME_SERIES_ML_SOLUTION_GUIDE.md
    └── 📄 LAHN_INC_INVENTORY_STRATEGY_REPORT.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Analysis

```bash
# Run all components (default)
python run_analysis.py

# Or explicitly run all
python run_analysis.py --all
```

### 3. Run Individual Components

```bash
# Data analysis only
python run_analysis.py --data-analysis

# ML models only  
python run_analysis.py --ml-models

# Time series forecasting only
python run_analysis.py --time-series

# Launch dashboard only
python run_analysis.py --dashboard
```

### 4. Other Options

```bash
# Check project structure
python run_analysis.py --structure

# Check dependencies
python run_analysis.py --check-deps

# Show help
python run_analysis.py --help
```

## 📊 Features

### 🔍 Data Analysis (`main.py`)
- **Customer Demographics**: Age, gender, geographic analysis
- **Product Performance**: Category and sub-category analysis
- **Customer Segmentation**: K-means clustering
- **Profitability Analysis**: Profit margins and performance metrics
- **Geographic Analysis**: Country and state-level insights

### 🤖 ML Models (`inventory_ml_models.py`)
- **Demand Forecasting**: RandomForest models for demand prediction
- **Inventory Optimization**: EOQ, safety stock, reorder points
- **Product Performance**: High/medium/low classification
- **Seasonal Analysis**: Pattern recognition and forecasting
- **Customer Lifetime Value**: CLV prediction and segmentation

### 📈 Time Series Forecasting (`time_series_forecasting.py`)
- **Prophet Models**: Facebook Prophet with seasonality
- **ARIMA Models**: Traditional time series forecasting
- **Seasonal Decomposition**: Trend and pattern analysis
- **Ensemble Methods**: Multiple model combinations
- **Advanced Metrics**: RMSE, MAE, MAPE evaluation

### 🌐 Interactive Dashboard (`inventory_dashboard.py`)
- **Executive Summary**: Key metrics and insights
- **Demand Forecasting**: Interactive forecast charts
- **Inventory Recommendations**: Actionable stock levels
- **Seasonal Analysis**: Pattern visualization
- **Model Performance**: Accuracy metrics and validation

## 📈 Key Business Metrics

### 📊 Current Performance
- **Total Revenue**: $85.3M across all categories
- **Total Records**: 113,036 sales transactions
- **Date Range**: 2011-2016 (5+ years of data)
- **Geographic Reach**: Multiple countries and states

### 🎯 Forecasting Accuracy
- **Bikes**: 94.2% accuracy, 7.56 RMSE
- **Accessories**: 88.4% accuracy, 310.38 RMSE
- **Clothing**: 63.8% accuracy, 98.07 RMSE

### 💰 Inventory Recommendations
- **Total Investment**: $10.1M for optimal stock levels
- **Expected Revenue**: $85.3M with 15-25% improvement
- **Cost Reduction**: 20-30% reduction in carrying costs

## 🛠️ Technical Requirements

### Dependencies
```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
plotly>=5.15.0
streamlit>=1.25.0
prophet>=1.1.4
statsmodels>=0.14.0
```

### System Requirements
- Python 3.8+
- 4GB+ RAM for large datasets
- 500MB+ free disk space for outputs

## 📝 Usage Examples

### Running Individual Steps
```python
# Import and run specific components
from src.main import main as run_data_analysis
from src.inventory_ml_models import InventoryMLSuite
from src.time_series_forecasting import TimeSeriesInventoryForecaster

# Run data analysis
run_data_analysis()

# Run ML models
ml_suite = InventoryMLSuite()
ml_suite.run_complete_analysis()

# Run time series forecasting
forecaster = TimeSeriesInventoryForecaster()
forecaster.run_complete_time_series_analysis()
```

### Dashboard Usage
```bash
# Launch dashboard
python run_analysis.py --dashboard

# Access at http://localhost:8501
# Navigate through different sections:
# - Executive Summary
# - Demand Forecasting
# - Inventory Recommendations
# - Seasonal Analysis
# - Model Performance
```

## 📊 Output Files

### 📈 Visualizations (outputs/visualizations/)
- `customer_demographics.png`: Customer analysis charts
- `geographic_analysis.png`: Geographic performance maps
- `product_category_analysis.png`: Product performance analysis
- `customer_segmentation.png`: K-means clustering results
- `profitability_analysis.png`: Profit margin analysis
- `demand_forecasting_analysis.png`: ML model predictions
- `seasonal_patterns_analysis.png`: Seasonal trend analysis

### 📄 Reports (outputs/reports/)
- `comprehensive_report.csv`: Executive summary metrics
- `inventory_recommendations.csv`: ML-based stock recommendations
- `time_series_inventory_recommendations.csv`: Time series predictions

### 🤖 Models (outputs/models/)
- Trained ML models for future predictions
- Model performance metrics and validation results

## 🎯 Business Impact

### 📊 Expected Improvements
- **Revenue Growth**: 15-25% increase through optimized stocking
- **Cost Reduction**: 20-30% reduction in carrying costs
- **Stockout Prevention**: 95% reduction in out-of-stock incidents
- **Cash Flow**: Improved working capital management

### 🔄 Operational Benefits
- **Automated Forecasting**: Daily demand predictions
- **Smart Reordering**: Automated purchase recommendations
- **Seasonal Planning**: Advanced seasonal pattern recognition
- **Risk Management**: Safety stock calculations and monitoring

## 🛡️ Error Handling

The system includes comprehensive error handling:
- **Data Validation**: Automatic data quality checks
- **Model Fallbacks**: Multiple model approaches for reliability
- **Graceful Failures**: Continues operation if individual components fail
- **Detailed Logging**: Comprehensive error messages and debugging info

## 🔄 Maintenance

### Regular Updates
- **Data Refresh**: Load new sales data monthly
- **Model Retraining**: Update ML models quarterly
- **Performance Monitoring**: Track prediction accuracy
- **Dashboard Updates**: Refresh visualizations weekly

### Performance Optimization
- **Caching**: Streamlit caching for faster dashboard loading
- **Parallel Processing**: Multi-threaded model training
- **Memory Management**: Efficient data handling for large datasets
- **Storage Optimization**: Compressed model storage

## 📞 Support

### Troubleshooting
1. **Dependency Issues**: Run `python run_analysis.py --check-deps`
2. **File Not Found**: Ensure `sales_data.csv` is in `data/` directory
3. **Memory Errors**: Reduce dataset size or increase available RAM
4. **Model Errors**: Check data quality and feature availability

### Common Issues
- **Prophet Installation**: May require additional system dependencies
- **Streamlit Port**: Use different port if 8501 is occupied
- **File Permissions**: Ensure write permissions for output directories
- **Data Format**: Verify CSV format matches expected schema

## 🎉 Getting Started

1. **Clone/Download** the project
2. **Install dependencies** with `pip install -r requirements.txt`
3. **Ensure data file** `sales_data.csv` is in `data/` directory
4. **Run complete analysis** with `python run_analysis.py`
5. **Launch dashboard** with `python run_analysis.py --dashboard`
6. **Access dashboard** at `http://localhost:8501`

## 📋 License

This project is developed for Lahn Inc. business intelligence purposes.

---

**Ready to transform your inventory management with AI-powered insights!** 🚀

For questions or support, refer to the comprehensive documentation in the `docs/` directory. 