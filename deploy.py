#!/usr/bin/env python3
"""
Deployment Helper Script for Indonesia Super League Football Analytics Dashboard
This script checks if your project is ready for deployment to Streamlit Cloud.
"""

import os
import sys
import pandas as pd
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - MISSING")
        return False

def check_imports():
    """Check if all required packages can be imported"""
    required_packages = ['streamlit', 'pandas', 'plotly', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ Package {package} imported successfully")
        except ImportError:
            print(f"❌ Package {package} not found")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_data_file():
    """Check if the data file exists and is readable"""
    data_path = Path("data/football_stats.csv")
    
    if not data_path.exists():
        print("❌ Data file not found: data/football_stats.csv")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Data file loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for required columns
        if 'TEAM' not in df.columns:
            print("❌ Required column 'TEAM' not found in data file")
            return False
        
        print(f"✅ Required column 'TEAM' found with {len(df['TEAM'].unique())} unique teams")
        return True
        
    except Exception as e:
        print(f"❌ Error reading data file: {e}")
        return False

def check_app_file():
    """Check if the main app file exists and is valid Python"""
    app_path = Path("app.py")
    
    if not app_path.exists():
        print("❌ Main app file not found: app.py")
        return False
    
    try:
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, app_path, 'exec')
        print("✅ app.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in app.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def main():
    """Main deployment check function"""
    print("🚀 Streamlit Cloud Deployment Check")
    print("=" * 50)
    
    # Check essential files
    files_to_check = [
        ("app.py", "Main Streamlit application"),
        ("requirements.txt", "Python dependencies"),
        ("data/football_stats.csv", "Football statistics data"),
        (".streamlit/config.toml", "Streamlit configuration"),
        ("README.md", "Project documentation"),
        ("DEPLOYMENT.md", "Deployment guide")
    ]
    
    file_checks = []
    for file_path, description in files_to_check:
        file_checks.append(check_file_exists(file_path, description))
    
    print("\n" + "=" * 50)
    
    # Check imports
    print("\n📦 Package Dependencies:")
    import_check = check_imports()
    
    print("\n" + "=" * 50)
    
    # Check data file
    print("\n📊 Data File Check:")
    data_check = check_data_file()
    
    print("\n" + "=" * 50)
    
    # Check app file
    print("\n🐍 App File Check:")
    app_check = check_app_file()
    
    print("\n" + "=" * 50)
    
    # Summary
    print("\n📋 DEPLOYMENT READINESS SUMMARY:")
    print("=" * 50)
    
    all_checks = file_checks + [import_check, data_check, app_check]
    
    if all(all_checks):
        print("🎉 ALL CHECKS PASSED! Your project is ready for deployment.")
        print("\n📝 Next Steps:")
        print("1. Initialize Git repository: git init")
        print("2. Add all files: git add .")
        print("3. Commit changes: git commit -m 'Initial commit'")
        print("4. Create GitHub repository and push your code")
        print("5. Deploy to Streamlit Cloud at: https://share.streamlit.io")
        print("\n📖 See DEPLOYMENT.md for detailed instructions")
    else:
        print("⚠️  SOME CHECKS FAILED. Please fix the issues above before deploying.")
        print("\n🔧 Common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Ensure data file exists in data/football_stats.csv")
        print("- Check file paths and permissions")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
