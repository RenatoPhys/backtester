"""
Módulo de estratégias para o forex_backtester.
"""

from .indicators import (
    sma_crossover,
    ema_crossover,
    bollinger_bands,
    rsi_strategy,
    macd_strategy
)

__all__ = [
    "sma_crossover",
    "ema_crossover",
    "bollinger_bands",
    "rsi_strategy",
    "macd_strategy"
]