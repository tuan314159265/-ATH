"""
Setup configuration for Retail Analytics Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retail-analytics-platform",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive retail customer analytics platform with ETL, RFM analysis, clustering, and ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retail-analytics-platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "sqlalchemy>=1.4.0",
        "psycopg2-binary>=2.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "streamlit>=1.28.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "retail-analytics=main:run_pipeline",
        ],
    },
)
