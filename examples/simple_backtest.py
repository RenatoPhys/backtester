"""
Exemplo de uso do Forex Backtester com estratégias simples.
"""

import pandas as pd
from forex_backtester import Backtester
from forex_backtester.strategies import sma_crossover, bollinger_bands

# Parâmetros de backtesting
SYMBOL = "EURUSD"
TIMEFRAME = "H1"
DATA_INI = "2022-01-01"
DATA_FIM = "2022-12-31"
TP = 50  # Take Profit em pontos
SL = 30  # Stop Loss em pontos
PATH_BASE = "./data/"

# Inicializa o backtester
bt = Backtester(
    symbol=SYMBOL,
    timeframe=TIMEFRAME,
    data_ini=DATA_INI,
    data_fim=DATA_FIM,
    tp=TP,
    sl=SL,
    slippage=1,
    tc=0.5,
    lote=1.0,
    path_base=PATH_BASE
)

# Exemplo 1: Backtest com estratégia de cruzamento de médias móveis
print("Executando backtest com estratégia SMA Crossover...")
results, metrics = bt.run(signal_function=sma_crossover, signal_args={"fast_period": 5, "slow_period": 20})
bt.print_metrics(metrics)

# Exemplo 2: Backtest com estratégia de Bandas de Bollinger
print("\nExecutando backtest com estratégia Bollinger Bands...")
results, metrics = bt.run(signal_function=bollinger_bands, signal_args={"period": 20, "std_dev": 2})
bt.print_metrics(metrics)

# Salva os resultados do último backtest
results.to_csv("resultados_bollinger.csv")

# Plota a curva de equity
plt = bt.plot_equity_curve()
plt.savefig("equity_curve_bollinger.png")
plt.close()

print("\nBacktests concluídos! Resultados salvos.")