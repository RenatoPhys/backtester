"""
Funções de estratégias de trading para o forex_backtester.
"""

import numpy as np
import pandas as pd


def sma_crossover(df, fast_period=5, slow_period=20):
    """ 
    Estratégia de cruzamento de médias móveis simples.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        fast_period (int): Período da média móvel rápida.
        slow_period (int): Período da média móvel lenta.
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()  # Para evitar SettingWithCopyWarning
    df['sma_fast'] = df['close'].rolling(window=fast_period).mean()
    df['sma_slow'] = df['close'].rolling(window=slow_period).mean()
    
    # Gera sinais: 1 quando rápida cruza acima da lenta, -1 ao contrário
    df['signal'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
    
    return df['signal']


def ema_crossover(df, fast_period=12, slow_period=26):
    """
    Estratégia de cruzamento de médias móveis exponenciais.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        fast_period (int): Período da média móvel exponencial rápida.
        slow_period (int): Período da média móvel exponencial lenta.
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Gera sinais: 1 quando rápida cruza acima da lenta, -1 ao contrário
    df['signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    return df['signal']


def bollinger_bands(df, period=20, std_dev=2):
    """
    Estratégia baseada em Bandas de Bollinger.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        period (int): Período para cálculo da média e desvio padrão.
        std_dev (float): Número de desvios padrão para as bandas.
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()
    df['sma'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['upper_band'] = df['sma'] + (rolling_std * std_dev)
    df['lower_band'] = df['sma'] - (rolling_std * std_dev)
    
    # Gera sinais: compra quando preço toca banda inferior, vende quando toca banda superior
    signals = np.zeros(len(df))
    signals[df['close'] <= df['lower_band']] = 1  # Compra
    signals[df['close'] >= df['upper_band']] = -1  # Venda
    
    # Preenche os zeros com o último sinal não-zero
    # (mantém a posição até um novo sinal)
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1]
            
    return pd.Series(signals, index=df.index)


def rsi_strategy(df, period=14, overbought=70, oversold=30):
    """
    Estratégia baseada no indicador RSI (Relative Strength Index).
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        period (int): Período para cálculo do RSI.
        overbought (int): Nível de sobrecompra.
        oversold (int): Nível de sobrevenda.
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()
    
    # Calcula o RSI
    delta = df['close'].diff().dropna()
    gains = delta.copy().clip(lower=0)
    losses = -1 * delta.copy().clip(upper=0)
    
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    
    rsi = np.zeros(len(df))
    
    for i in range(period, len(df)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    df['rsi'] = rsi
    
    # Generate signals
    signals = np.zeros(len(df))
    signals[df['rsi'] <= oversold] = 1  # Compra em sobrevenda
    signals[df['rsi'] >= overbought] = -1  # Venda em sobrecompra
    
    # Mantém a posição até haver sinal contrário
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1]
            
    return pd.Series(signals, index=df.index)


def macd_strategy(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Estratégia baseada no indicador MACD (Moving Average Convergence Divergence).
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        fast_period (int): Período da EMA rápida para MACD.
        slow_period (int): Período da EMA lenta para MACD.
        signal_period (int): Período da linha de sinal.
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """
    df = df.copy()
    
    # Calcula o MACD
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['signal_line'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal_line']
    
    # Gera sinais quando o MACD cruza a linha de sinal
    signals = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if df['macd'].iloc[i] > df['signal_line'].iloc[i] and df['macd'].iloc[i-1] <= df['signal_line'].iloc[i-1]:
            signals[i] = 1  # Compra quando MACD cruza acima da linha de sinal
        elif df['macd'].iloc[i] < df['signal_line'].iloc[i] and df['macd'].iloc[i-1] >= df['signal_line'].iloc[i-1]:
            signals[i] = -1  # Venda quando MACD cruza abaixo da linha de sinal
        else:
            signals[i] = signals[i-1]  # Mantém a posição
            
    return pd.Series(signals, index=df.index)