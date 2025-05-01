"""
Funções de estratégias de trading para o forex_backtester.
"""

import numpy as np
import pandas as pd


def bollinger_bands(df, bb_length, std, allowed_hours=None):

    """
    Estratégia baseada em Bandas de Bollinger.
    
    Args:
        df (pandas.DataFrame): DataFrame com dados OHLC.
        bb_length (int): Período para cálculo da média e desvio padrão.
        std (float): Número de desvios padrão para as bandas.
        allowed_hours (list): horas que vamos executar a estratégia
        
    Returns:
        pandas.Series: Posições (-1=short, 0=neutro, 1=long)
    """

    df = df.copy()  # Para evitar SettingWithCopyWarning
    
    aux = df.ta.bbands(length=bb_length, std = std)
    df[f"BBL_{bb_length}_{std}"] = aux[f"BBL_{bb_length}_{std}"]
    df[f"BBU_{bb_length}_{std}"] = aux[f"BBU_{bb_length}_{std}"]
    df[f"BBM_{bb_length}_{std}"] = aux[f"BBM_{bb_length}_{std}"]    
    

    # Calculando entradas (buy/sell)
    cond1 = (df.close < df[f"BBL_{bb_length}_{std}"]) & (df.close.shift(+1) >= df[f"BBL_{bb_length}_{std}"].shift(+1))
    cond2 = (df.close > df[f"BBU_{bb_length}_{std}"]) & (df.close.shift(+1) <= df[f"BBU_{bb_length}_{std}"].shift(+1))

    df['position'] = 0
    df.loc[cond1, "position"] = +1
    df.loc[cond2, "position"] = -1
    
    
    # Restrição de horários
    if allowed_hours is not None:
        # Zera posição fora dos horários permitidos
        current_hours = df.index.to_series().dt.hour
        df.loc[~current_hours.isin(allowed_hours), 'position'] = 0  
   
    return df['position']


