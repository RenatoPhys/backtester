# Forex Backtester

Framework de backtesting para estratégias sistemáticas em mercados FOREX.

## Características

- Backtesting rápido e eficiente para dados FOREX
- Suporte a múltiplos timeframes
- Implementação de Take Profit e Stop Loss
- Cálculo de métricas de desempenho detalhadas
- Visualização de resultados

## Instalação

```bash
pip install forex-backtester
```

## Uso Básico

```python
from forex_backtester import Backtester
from forex_backtester.strategies import SimpleMovingAverageCrossover

# Configurar o backtester
bt = Backtester(
    symbol="EURUSD",
    timeframe="H1",
    data_ini="2020-01-01",
    data_fim="2020-12-31",
    tp=50,
    sl=30,
    slippage=1,
    tc=0.5,
    lote=1.0,
    path_base="./data/"
)

# Definir estratégia
strategy = SimpleMovingAverageCrossover(fast_period=5, slow_period=20)

# Executar backtest
results, metrics = bt.run(signal_function=strategy.generate_signals)

# Imprimir métricas
bt.print_metrics(metrics)
```

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.