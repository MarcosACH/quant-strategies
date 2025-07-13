from setuptools import setup, find_packages

setup(
    name="quantitative-trading-strategies",
    version="1.0.0",
    author="ActiveQuants",
    description="A comprehensive framework for developing and backtesting quantitative trading strategies",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "vectorbt>=0.25.0",
        "scipy>=1.9.0",
        "yfinance>=0.2.0",
        "plotly>=5.10.0",
        "scikit-learn>=1.1.0",
        "joblib>=1.1.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "loguru>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "data": [
            "ccxt>=4.0.0",
            "alpha-vantage>=2.3.0",
            "sqlalchemy>=1.4.0",
            "h5py>=3.7.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "backtest-strategy=scripts.backtesting.run_backtest:main",
            "optimize-strategy=scripts.optimization.run_optimization:main",
        ],
    },
)
