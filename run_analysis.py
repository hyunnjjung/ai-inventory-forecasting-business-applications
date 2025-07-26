#!/usr/bin/env python3
"""
LAHN INC. - COMPLETE BUSINESS INTELLIGENCE SYSTEM
==================================================

Main orchestrator script to run all analysis components.
This script provides a unified interface to execute:
- Data Analysis
- ML Models  
- Time Series Forecasting
- Interactive Dashboard

Project Structure:
- src/: Source code files
- data/: Data files
- outputs/: Generated reports and visualizations
- notebooks/: Jupyter notebooks
- docs/: Documentation

Usage:
    python run_analysis.py [option]
    
Options:
    --data-analysis    : Run comprehensive data analysis
    --ml-models        : Run ML inventory models
    --time-series      : Run time series forecasting
    --dashboard        : Launch interactive dashboard
    --all              : Run all components (default)
    --help             : Show this help message

Author: AI Assistant
Date: 2025
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class LahnIncAnalysisOrchestrator:
    """Main orchestrator class for all analysis components"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.src_dir = self.base_dir / "src"
        self.data_dir = self.base_dir / "data"
        self.outputs_dir = self.base_dir / "outputs"
        
        # Ensure output directories exist
        (self.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "models").mkdir(parents=True, exist_ok=True)
        
        print("🚀 LAHN INC. BUSINESS INTELLIGENCE SYSTEM")
        print("="*50)
        print(f"📁 Project Directory: {self.base_dir}")
        print(f"📊 Data Directory: {self.data_dir}")
        print(f"📈 Outputs Directory: {self.outputs_dir}")
        print()
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("Checking dependencies...")
        
        # Package name mapping: pip_name -> import_name
        required_packages = {
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'scikit-learn': 'sklearn',
            'plotly': 'plotly',
            'streamlit': 'streamlit',
            'prophet': 'prophet',
            'statsmodels': 'statsmodels'
        }
        
        missing_packages = []
        available_packages = []
        
        for pip_name, import_name in required_packages.items():
            try:
                # Use a clean import test
                import subprocess
                result = subprocess.run([
                    'python', '-c', f'import {import_name}; print("OK")'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    available_packages.append(pip_name)
                    print(f"  [OK] {pip_name}")
                else:
                    missing_packages.append(pip_name)
                    print(f"  [MISSING] {pip_name}")
            except Exception as e:
                # If subprocess fails, try a simpler approach
                try:
                    __import__(import_name)
                    available_packages.append(pip_name)
                    print(f"  [OK] {pip_name}")
                except ImportError:
                    missing_packages.append(pip_name)
                    print(f"  [MISSING] {pip_name}")
        
        if missing_packages:
            print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
            print("Attempting to install missing packages...")
            
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("[SUCCESS] Dependencies installed successfully!")
                    print("Please restart the script to use the newly installed packages.")
                    return False
                else:
                    print("[ERROR] Failed to install dependencies automatically.")
                    print("Error output:")
                    print(result.stderr)
                    print("\n[SOLUTION] Please try the following:")
                    print("1. Run manually: pip install -r requirements.txt")
                    print("2. If using a virtual environment, deactivate it and try again:")
                    print("   - Windows: deactivate")
                    print("   - Then run: pip install -r requirements.txt")
                    print("3. If still having issues, try installing packages globally:")
                    print("   - pip install --user -r requirements.txt")
                    return False
                    
            except Exception as e:
                print(f"[ERROR] Could not install dependencies: {str(e)}")
                print("\n[SOLUTION] Please install dependencies manually:")
                print("pip install -r requirements.txt")
                return False
        
        print("[SUCCESS] All dependencies satisfied!")
        return True
    
    def run_data_analysis(self):
        """Run comprehensive data analysis"""
        print("\n📊 RUNNING DATA ANALYSIS")
        print("="*30)
        
        try:
            # Run main analysis as subprocess to avoid import issues
            import subprocess
            result = subprocess.run([
                sys.executable, "-c", 
                f"""
import sys
import os
sys.path.insert(0, r"{self.src_dir}")
os.chdir(r"{self.src_dir}")
from main import main as run_main_analysis
run_main_analysis()
"""
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                print("✅ Data analysis completed successfully!")
                print(f"📁 Check outputs in: {self.outputs_dir}")
                if result.stdout:
                    print("\n📋 Output:")
                    print(result.stdout)
            else:
                print(f"❌ Error in data analysis:")
                if result.stderr:
                    print(result.stderr)
                return False
            
        except Exception as e:
            print(f"❌ Error in data analysis: {str(e)}")
            return False
        
        return True
    
    def run_ml_models(self):
        """Run ML inventory optimization models"""
        print("\n🤖 RUNNING ML MODELS")
        print("="*25)
        
        try:
            # Run ML models as subprocess
            import subprocess
            result = subprocess.run([
                sys.executable, "-c", 
                f"""
import sys
import os
sys.path.insert(0, r"{self.src_dir}")
os.chdir(r"{self.src_dir}")
from inventory_ml_models import InventoryMLSuite
suite = InventoryMLSuite()
suite.run_complete_analysis()
"""
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                print("✅ ML models completed successfully!")
                print(f"📁 Check outputs in: {self.outputs_dir}")
                if result.stdout:
                    print("\n📋 Output:")
                    print(result.stdout[-2000:])  # Show last 2000 chars to avoid too much output
            else:
                print(f"❌ Error in ML models:")
                if result.stderr:
                    print(result.stderr)
                return False
            
        except Exception as e:
            print(f"❌ Error in ML models: {str(e)}")
            return False
        
        return True
    
    def run_time_series(self):
        """Run time series forecasting"""
        print("\n📈 RUNNING TIME SERIES FORECASTING")
        print("="*35)
        
        try:
            # Run time series as subprocess
            import subprocess
            result = subprocess.run([
                sys.executable, "-c", 
                f"""
import sys
import os
sys.path.insert(0, r"{self.src_dir}")
os.chdir(r"{self.src_dir}")
from time_series_forecasting import TimeSeriesInventoryForecaster
forecaster = TimeSeriesInventoryForecaster()
forecaster.run_complete_time_series_analysis()
"""
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                print("✅ Time series forecasting completed successfully!")
                print(f"📁 Check outputs in: {self.outputs_dir}")
                if result.stdout:
                    print("\n📋 Output:")
                    print(result.stdout[-2000:])  # Show last 2000 chars to avoid too much output
            else:
                print(f"❌ Error in time series forecasting:")
                if result.stderr:
                    print(result.stderr)
                return False
            
        except Exception as e:
            print(f"❌ Error in time series forecasting: {str(e)}")
            return False
        
        return True
    
    def launch_dashboard(self):
        """Launch interactive Streamlit dashboard"""
        print("\n🌐 LAUNCHING INTERACTIVE DASHBOARD")
        print("="*35)
        
        try:
            dashboard_path = self.src_dir / "inventory_dashboard.py"
            
            if not dashboard_path.exists():
                print(f"❌ Dashboard file not found: {dashboard_path}")
                return False
            
            print("🚀 Starting Streamlit dashboard...")
            print("📍 Dashboard will be available at: http://localhost:8501")
            print("🔄 Press Ctrl+C to stop the dashboard")
            
            # Launch Streamlit dashboard
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path), "--server.port=8501"
            ])
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Error launching dashboard: {str(e)}")
            return False
        
        return True
    
    def run_all_components(self):
        """Run all analysis components in sequence"""
        print("\n🔄 RUNNING ALL COMPONENTS")
        print("="*30)
        
        success_count = 0
        
        # Run data analysis
        if self.run_data_analysis():
            success_count += 1
        
        # Run ML models
        if self.run_ml_models():
            success_count += 1
        
        # Run time series forecasting
        if self.run_time_series():
            success_count += 1
        
        print(f"\n📊 EXECUTION SUMMARY")
        print("="*20)
        print(f"✅ Successfully completed: {success_count}/3 components")
        
        if success_count == 3:
            print("🎉 All components executed successfully!")
            
            # Ask if user wants to launch dashboard
            response = input("\n🌐 Launch interactive dashboard? (y/n): ").lower()
            if response in ['y', 'yes']:
                self.launch_dashboard()
        else:
            print("⚠️  Some components failed. Check error messages above.")
    
    def show_project_structure(self):
        """Display the organized project structure"""
        print("\n📁 PROJECT STRUCTURE")
        print("="*20)
        
        def print_tree(directory, prefix="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            items = list(directory.iterdir())
            items.sort(key=lambda x: (x.is_file(), x.name))
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and not item.name.startswith('.') and current_depth < max_depth:
                    extension = "    " if is_last else "│   "
                    print_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        print_tree(self.base_dir)
    
    def show_help(self):
        """Display help information"""
        print(__doc__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LAHN INC. Business Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-analysis', action='store_true',
                        help='Run comprehensive data analysis')
    parser.add_argument('--ml-models', action='store_true',
                        help='Run ML inventory models')
    parser.add_argument('--time-series', action='store_true',
                        help='Run time series forecasting')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch interactive dashboard')
    parser.add_argument('--all', action='store_true',
                        help='Run all components (default)')
    parser.add_argument('--structure', action='store_true',
                        help='Show project structure')
    parser.add_argument('--check-deps', action='store_true',
                        help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Create orchestrator instance
    orchestrator = LahnIncAnalysisOrchestrator()
    
    # Show project structure if requested
    if args.structure:
        orchestrator.show_project_structure()
        return
    
    # Check dependencies if requested
    if args.check_deps:
        orchestrator.check_dependencies()
        return
    
    # Check dependencies before running any analysis
    if not orchestrator.check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        return
    
    # Execute requested components
    if args.data_analysis:
        orchestrator.run_data_analysis()
    elif args.ml_models:
        orchestrator.run_ml_models()
    elif args.time_series:
        orchestrator.run_time_series()
    elif args.dashboard:
        orchestrator.launch_dashboard()
    elif args.all:
        orchestrator.run_all_components()
    else:
        # Default: run all components
        orchestrator.run_all_components()

if __name__ == "__main__":
    main() 