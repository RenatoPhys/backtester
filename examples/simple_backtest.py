"""
Exemplo de uso do Forex Backtester com estratégias simples.
"""

import pandas as pd
from forex_backtester import Backtester
from forex_backtester.strategies import sma_crossover, bollinger_bands
import pandas_ta as ta
import talib

# Configurar o backtester
sym = 'USDJPY'
bt = Backtester(
    symbol = sym,
    timeframe = 't5',
    data_ini = '2019-01-01',
    data_fim = '2025-12-10',
    tp = params['tp'],
    sl = params['sl'],
    slippage = 0,
    tc = dict_custos[sym], # $ per lot
    lote = 0.01,
    valor_lote= dict_valor_lot[sym],
    initial_cash = 30000,
    path_base= dict_path[sym],
    daytrade = True
)

# Parâmetros da estratégia RSI
BB_LENGTH = params['BB_LENGTH']
STD = params['STD']

# Executa o backtest com a estratégia RSI
results, metrics = bt.run(
    signal_function=entrada, 
    signal_args={
        "bb_length": BB_LENGTH,
        "std": STD,
        'allowed_hours': [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
        #'allowed_hours': [16]
    }
)


# Print de métricas
bt.print_metrics(metrics)