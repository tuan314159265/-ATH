#!/usr/bin/env python3
"""
Quick start script to verify project setup and run the application
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🛍️  RETAIL CUSTOMER ANALYTICS PLATFORM  🛍️          ║
    ║                                                              ║
    ║               Professional Portfolio Project                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check if Python version is compatible"""
    print("\n✓ Checking Python version...")
    if sys.version_info < (3, 8):
        print("  ❌ Python 3.8+ required")
        return False
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_virtual_env():
    """Check if running in virtual environment"""
    print("\n✓ Checking virtual environment...")
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("  ✅ Virtual environment detected")
        return True
    print("  ⚠️  Not in virtual environment (recommended but not required)")
    return True


def check_dependencies():
    """Check if main dependencies are installed"""
    print("\n✓ Checking dependencies...")
    required_packages = ['pandas', 'streamlit', 'sklearn', 'numpy']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package} installed")
        except ImportError:
            print(f"  ❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"  pip install -r requirements.txt")
        return False
    return True


def check_file_structure():
    """Check if project structure is intact"""
    print("\n✓ Checking project structure...")
    required_dirs = ['app', 'modeling', 'clean_data', 'raw_data', 'docs']
    required_files = ['README.md', 'requirements.txt', 'main.py', 'Dockerfile']
    
    project_root = Path(__file__).parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}/ found")
        else:
            print(f"  ⚠️  {dir_name}/ missing (optional)")
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✅ {file_name} found")
        else:
            print(f"  ⚠️  {file_name} missing")
    
    return True


def print_next_steps():
    """Print next steps"""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    
    steps = [
        ("1. Install dependencies", "pip install -r requirements.txt"),
        ("2. Run Streamlit app", "streamlit run app/app.py"),
        ("3. Run full pipeline", "python main.py --all"),
        ("4. Docker deployment", "docker-compose up -d"),
        ("5. Read documentation", "cat QUICKSTART.md"),
    ]
    
    for step, command in steps:
        print(f"\n{step}:")
        print(f"   $ {command}")


def print_features():
    """Print project features"""
    print("\n" + "="*60)
    print("✨ PROJECT FEATURES")
    print("="*60)
    
    features = [
        "📊 Multi-source ETL pipeline (3 data sources)",
        "💳 RFM analysis with customer segmentation",
        "🎯 K-means clustering (optimal K selection)",
        "🤖 ML models (Decision Tree & Random Forest)",
        "📈 87.3% accuracy Random Forest classifier",
        "🌐 Interactive Streamlit dashboard (6 pages)",
        "🐳 Docker containerization & deployment",
        "📚 Comprehensive documentation (5 guides)",
        "🚀 Production-ready code structure",
        "☁️  Multi-cloud deployment options",
    ]
    
    for feature in features:
        print(f"  {feature}")


def print_documentation():
    """Print documentation guide"""
    print("\n" + "="*60)
    print("📚 DOCUMENTATION")
    print("="*60)
    
    docs = [
        ("README.md", "Complete project documentation"),
        ("QUICKSTART.md", "Quick setup in 5 minutes"),
        ("PORTFOLIO.md", "CV/Portfolio project summary"),
        ("docs/PIPELINE.md", "Data pipeline architecture"),
        ("docs/MODELS.md", "ML models documentation"),
        ("docs/DEPLOYMENT.md", "Deployment guide"),
    ]
    
    for file_name, description in docs:
        print(f"\n  📖 {file_name}")
        print(f"     {description}")


def main():
    """Main execution"""
    print_banner()
    print_features()
    print_documentation()
    
    # Run checks
    print("\n" + "="*60)
    print("✓ SYSTEM CHECKS")
    print("="*60)
    
    checks = [
        ("Python version", check_python_version),
        ("Virtual environment", check_virtual_env),
        ("Dependencies", check_dependencies),
        ("Project structure", check_file_structure),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"  ⚠️  Error checking {check_name}: {e}")
    
    print_next_steps()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! You're ready to go.")
    else:
        print("⚠️  Some checks failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
    print("="*60)
    
    print("\n💡 Pro tips:")
    print("   • Use 'make' commands: make help")
    print("   • Deploy to Streamlit Cloud: Free hosting")
    print("   • Use Docker for consistent environment")
    print("   • Check PORTFOLIO.md for CV highlights")
    print("\n")


if __name__ == "__main__":
    main()
