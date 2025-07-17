import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
import json

from config.settings import settings

Base = declarative_base()


class MarketData(Base):
    """Market data table schema."""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    source = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QuestDBManager:
    """
    Database abstraction layer for market data and backtesting results.

    This class provides methods for storing and retrieving market data,
    backtest results, and other trading-related information.
    """

    def __init__(self, database_url: str = None):
        """
        Initialize database connection.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or settings.DATABASE_URL
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)

    def store_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        source: str = "unknown",
        if_exists: str = "append"
    ) -> None:
        """
        Store market data in the database.

        Args:
            data: DataFrame with OHLCV data
            symbol: Asset symbol
            timeframe: Data timeframe (e.g., '1D', '1H')
            source: Data source identifier
            if_exists: How to behave if table exists ('append', 'replace', 'fail')
        """
        # Prepare data for storage
        data_copy = data.copy()
        data_copy['symbol'] = symbol
        data_copy['timeframe'] = timeframe
        data_copy['source'] = source
        data_copy['created_at'] = datetime.utcnow()

        # Ensure timestamp column exists
        if 'timestamp' not in data_copy.columns:
            data_copy['timestamp'] = data_copy.index

        # Store in database
        data_copy.to_sql(
            'market_data',
            self.engine,
            if_exists=if_exists,
            index=False,
            method='multi'
        )

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve market data from the database.

        Args:
            symbol: Asset symbol
            timeframe: Data timeframe
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            source: Data source filter

        Returns:
            DataFrame with market data
        """
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM market_data
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """

        if start_date:
            query += f" AND timestamp >= '{start_date}'"

        if end_date:
            query += f" AND timestamp <= '{end_date}'"

        if source:
            query += f" AND source = '{source}'"

        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, self.engine, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    def store_backtest_results(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        parameters: Dict[str, Any],
        results: Dict[str, float]
    ) -> None:
        """
        Store backtest results in the database.

        Args:
            strategy_name: Name of the strategy
            symbol: Asset symbol
            timeframe: Data timeframe
            start_date: Backtest start date
            end_date: Backtest end date
            parameters: Strategy parameters
            results: Backtest performance metrics
        """
        with self.SessionLocal() as session:
            backtest_result = BacktestResults(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                parameters=json.dumps(parameters),
                total_return=results.get('total_return'),
                sharpe_ratio=results.get('sharpe_ratio'),
                max_drawdown=results.get('max_drawdown'),
                total_trades=results.get('total_trades'),
                win_rate=results.get('win_rate'),
                profit_factor=results.get('profit_factor')
            )

            session.add(backtest_result)
            session.commit()

    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve backtest results from the database.

        Args:
            strategy_name: Filter by strategy name
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            limit: Maximum number of results

        Returns:
            DataFrame with backtest results
        """
        query = "SELECT * FROM backtest_results WHERE 1=1"

        if strategy_name:
            query += f" AND strategy_name = '{strategy_name}'"

        if symbol:
            query += f" AND symbol = '{symbol}'"

        if timeframe:
            query += f" AND timeframe = '{timeframe}'"

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, self.engine, parse_dates=[
                               'start_date', 'end_date', 'created_at'])

        # Parse parameters JSON
        if not df.empty:
            df['parameters'] = df['parameters'].apply(json.loads)

        return df

    def get_data_quality_stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get data quality statistics for a symbol.

        Args:
            symbol: Asset symbol
            timeframe: Data timeframe

        Returns:
            Dictionary with quality statistics
        """
        query = f"""
        SELECT 
            COUNT(*) as total_records,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_count,
            AVG(volume) as avg_volume,
            MIN(low) as min_price,
            MAX(high) as max_price
        FROM market_data
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """

        result = pd.read_sql_query(query, self.engine)
        return result.iloc[0].to_dict() if not result.empty else {}

    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """
        Clean up old data beyond the specified retention period.

        Args:
            days_to_keep: Number of days to retain
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        with self.SessionLocal() as session:
            # Clean up old market data
            session.execute(
                f"DELETE FROM market_data WHERE created_at < '{cutoff_date}'"
            )

            # Clean up old backtest results
            session.execute(
                f"DELETE FROM backtest_results WHERE created_at < '{cutoff_date}'"
            )

            session.commit()

    def get_available_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols in the database.

        Args:
            timeframe: Filter by timeframe

        Returns:
            List of available symbols
        """
        query = "SELECT DISTINCT symbol FROM market_data"

        if timeframe:
            query += f" WHERE timeframe = '{timeframe}'"

        result = pd.read_sql_query(query, self.engine)
        return result['symbol'].tolist()

    def close(self):
        """Close database connections."""
        self.engine.dispose()
