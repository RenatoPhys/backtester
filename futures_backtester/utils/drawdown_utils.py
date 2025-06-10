"""
Módulo contendo funções utilitárias para cálculo de drawdown e métricas relacionadas.
"""

import numpy as np
import pandas as pd

def calc_dd(series):
    """
    Calcula o drawdown e métricas relacionadas para uma série de resultados.
    
    Args:
        series (pandas.Series): Série com os resultados acumulados.
        
    Returns:
        pandas.DataFrame: DataFrame com as seguintes colunas:
            - cummax: Máximo acumulado
            - drawdown: Drawdown em valor
            - drawdown_pct: Drawdown em percentual
            - underwater: Flag indicando se está em drawdown (1) ou não (0)
            - time_uw: Tempo em drawdown (contagem de períodos)
    """
    # Cria um DataFrame para armazenar os resultados
    df = pd.DataFrame({'equity': series})
    
    # Calcula o máximo acumulado
    df['cummax'] = df['equity'].cummax()
    
    # Calcula o drawdown em valor
    df['drawdown'] = df['equity'] - df['cummax']
    
    # Calcula o drawdown em percentual
    df['drawdown_pct'] = df['drawdown'] / df['cummax'].replace(0, np.nan)
    
    # Identifica períodos underwater (abaixo do pico)
    df['underwater'] = 0
    df.loc[df['equity'] < df['cummax'], 'underwater'] = 1
    
    # Calcula mudanças no estado underwater
    df['change_uw'] = df['underwater'].diff().fillna(0)
    
    # Enumera cada segmento de underwater
    df.loc[df['change_uw'] != 0, 'n_change_uw'] = np.arange(0, df[df['change_uw'] != 0].shape[0])
    df['n_change_uw'] = df['n_change_uw'].ffill()
    
    # Inicializa o tempo underwater para 0 quando há mudança
    df.loc[df['change_uw'] != 0, 'time_uw'] = 0
    
    # Calcula o tempo underwater para cada segmento
    g = df.groupby('n_change_uw')
    df['time_uw'] = g['time_uw'].ffill() + g.cumcount()
    
    # Zera o tempo underwater quando não está em drawdown
    df.loc[df['underwater'] == 0, 'time_uw'] = 0
    
    # Remove colunas auxiliares
    df = df.drop(['change_uw', 'n_change_uw'], axis=1)
    
    return df