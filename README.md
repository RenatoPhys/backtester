# Futures Backtester - Professional Backtesting Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Um framework Python profissional e de alta performance para backtesting de estratégias sistemáticas em mercados futuros, forex, commodities e criptomoedas.

## 🚀 Características Principais

- ⚡ **Alta Performance**: Otimizado com Numba para execução ultrarrápida
- 📊 **Análise Completa**: Métricas detalhadas incluindo Sharpe, Sortino, Calmar e mais
- 🎯 **Sistema de Stops**: Take Profit e Stop Loss automáticos com slippage
- 📈 **Visualizações Avançadas**: Gráficos de equity, drawdown e análise por hora
- 🔧 **Otimização Bayesiana**: Otimize estratégias com Optuna usando múltiplos núcleos
- 🕐 **Trading por Horário**: Configure parâmetros específicos para cada hora do dia
- 📁 **Suporte Multi-Timeframe**: De 1 minuto até timeframes diários
- 🏭 **Strategy Generator**: Sistema completo para criar e otimizar estratégias

## 📋 Índice

- [Instalação](#-instalação)
- [Quickstart](#-quickstart)
- [Uso Avançado](#-uso-avançado)
- [Strategy Optimizer](#-strategy-optimizer)
- [Estratégias Disponíveis](#-estratégias-disponíveis)
- [Métricas e Análises](#-métricas-e-análises)
- [Visualizações](#-visualizações)
- [API Reference](#-api-reference)
- [Exemplos](#-exemplos)
- [Performance](#-performance)

## 💻 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- NumPy, Pandas, Matplotlib
- Numba (para otimização de performance)
- Optuna (para otimização de estratégias)

### Instalação via pip (recomendado)

```bash
pip install futures-backtester
```

### Instalação via GitHub (desenvolvimento)

```bash
# Clone o repositório
git clone https://github.com/seu_usuario/futures-backtester.git
cd futures-backtester

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale em modo desenvolvimento
pip install -e .
```

## 🎯 Quickstart

### Exemplo Básico

```python
from futures_backtester import Backtester
from futures_backtester.strategies import bollinger_bands

# Configurar o backtester
bt = Backtester(
    symbol="EURUSD",
    timeframe="H1",
    data_ini="2023-01-01",
    data_fim="2023-12-31",
    tp=50,        # Take profit em pontos
    sl=30,        # Stop loss em pontos
    slippage=1,   # Slippage em pontos
    tc=0.5,       # Custo por transação
    lote=1.0,     # Tamanho do lote
    initial_cash=10000,
    path_base="./data/"
)

# Executar backtest com estratégia Bollinger Bands
results, metrics = bt.run(
    signal_function=bollinger_bands,
    signal_args={"bb_length": 20, "std": 2.0}
)

# Visualizar resultados
bt.print_metrics(metrics)
bt.plot_equity_curve(include_drawdown=True)
```

### Resultado Esperado

```
==================================================
RELATÓRIO DE DESEMPENHO
==================================================
Símbolo: EURUSD
Timeframe: H1
Período: 2023-01-01 a 2023-12-31

--- RESULTADOS ---
Saldo Inicial: $10,000.00
Saldo Final: $12,845.50
Retorno Total: $2,845.50 (28.46%)
Retorno Anualizado: 28.46%

--- RATIOS ---
Sharpe Ratio: 1.823
Sortino Ratio: 2.451
Calmar Ratio: 3.102
Profit Factor: 1.753
```

## 🔧 Uso Avançado

### Criando Estratégias Customizadas

```python
import numpy as np
import pandas as pd

def my_custom_strategy(df, fast_period=10, slow_period=30, allowed_hours=None):
    """
    Estratégia customizada de cruzamento de médias móveis.
    
    Args:
        df: DataFrame com dados OHLC
        fast_period: Período da média rápida
        slow_period: Período da média lenta
        allowed_hours: Lista de horas permitidas para trading
        
    Returns:
        pd.Series: Sinais de posição (-1=short, 0=neutro, 1=long)
    """
    # Calcular médias móveis
    fast_ma = df['close'].rolling(fast_period).mean()
    slow_ma = df['close'].rolling(slow_period).mean()
    
    # Gerar sinais
    position = pd.Series(0, index=df.index)
    position[fast_ma > slow_ma] = 1
    position[fast_ma < slow_ma] = -1
    
    # Aplicar filtro de horário se especificado
    if allowed_hours is not None:
        mask = df.index.hour.isin(allowed_hours)
        position[~mask] = 0
    
    return position
```

### Análise Multi-Timeframe

```python
# Analisar múltiplos timeframes
timeframes = ['M5', 'M15', 'H1', 'H4']
results = {}

for tf in timeframes:
    bt = Backtester(
        symbol="EURUSD",
        timeframe=tf,
        data_ini="2023-01-01",
        data_fim="2023-12-31",
        tp=50, sl=30,
        slippage=1, tc=0.5,
        lote=1.0,
        path_base="./data/"
    )
    
    _, metrics = bt.run(signal_function=my_custom_strategy)
    results[tf] = metrics

# Comparar resultados
for tf, metrics in results.items():
    print(f"{tf}: Sharpe={metrics['sharpe_ratio']:.3f}, "
          f"Return={metrics['total_return_pct']:.2f}%")
```

## 🏆 Strategy Optimizer

O módulo `StrategyOptimizer` permite otimização avançada de estratégias usando Optuna:

```python
from futures_backtester.strategy_generator import StrategyOptimizer
from futures_backtester.strategies import bollinger_bands

# Criar otimizador
optimizer = StrategyOptimizer(
    symbol="EURUSD",
    timeframe="H1",
    data_ini="2023-01-01",
    data_fim="2023-12-31",
    initial_cash=10000,
    num_trials=100,        # Tentativas por hora
    max_workers=4,         # Processamento paralelo
    optimize_metric='sortino_ratio',  # Métrica para otimizar
    direction='maximize'   # Maximizar ou minimizar
)

# Definir ranges de parâmetros para otimização
param_ranges = {
    'tp': (20, 100),       # Take profit entre 20 e 100 pontos
    'sl': (10, 50),        # Stop loss entre 10 e 50 pontos
    'bb_length': (10, 30), # Período BB entre 10 e 30
    'std': (1.5, 3.0, 0.1) # Desvio padrão entre 1.5 e 3.0 (step 0.1)
}

# Executar otimização completa
results_dir = optimizer.run(
    signal_function=bollinger_bands,
    param_ranges=param_ranges,
    hours_to_optimize=[9, 10, 11, 14, 15, 16],  # Horários para otimizar
    min_threshold=0.5      # Threshold mínimo para Sortino
)

print(f"Resultados salvos em: {results_dir}")
```

### Métricas Disponíveis para Otimização

- `sortino_ratio`: Melhor para estratégias que focam em minimizar perdas
- `sharpe_ratio`: Clássico risk-adjusted return
- `calmar_ratio`: Foco em drawdown
- `profit_factor`: Relação ganhos/perdas
- `win_rate`: Taxa de acerto
- `total_return`: Retorno absoluto

## 📊 Estratégias Disponíveis

### 1. Bollinger Bands
```python
from futures_backtester.strategies import bollinger_bands

signal_args = {
    "bb_length": 20,  # Período das bandas
    "std": 2.0,       # Desvio padrão
    "allowed_hours": [9, 10, 11, 14, 15]  # Horários permitidos
}
```

### 2. RSI Strategy
```python
from futures_backtester.strategies import rsi_strategy

signal_args = {
    "rsi_period": 14,
    "oversold": 30,
    "overbought": 70
}
```

### 3. MACD Strategy
```python
from futures_backtester.strategies import macd_strategy

signal_args = {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
}
```

## 📈 Métricas e Análises

### Métricas Calculadas Automaticamente

**Métricas de Retorno:**
- Retorno Total ($ e %)
- Retorno Anualizado
- Volatilidade Anualizada

**Métricas de Risco:**
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Time Underwater

**Métricas de Trading:**
- Total de Trades
- Taxa de Acerto (Win Rate)
- Profit Factor
- Média de Ganhos/Perdas
- Expectancy

**Análise de Saídas:**
- % Saídas por Take Profit
- % Saídas por Stop Loss
- % Saídas por Tempo

## 🎨 Visualizações

### 1. Curva de Equity com Drawdown

```python
bt.plot_equity_curve(figsize=(12, 8), include_drawdown=True)
plt.show()
```

### 2. Análise por Tipo de Posição

```python
bt.plot_by_position(figsize=(12, 6))
plt.show()
```

### 3. Performance por Hora do Dia

```python
bt.plot_profit_by_hour(figsize=(14, 8))
plt.show()
```

### 4. Evolução Acumulada por Hora

```python
bt.plot_cumulative_by_hour(figsize=(14, 10))
plt.show()
```

## 📚 API Reference

### Classe Backtester

```python
Backtester(
    symbol: str,           # Símbolo do ativo
    timeframe: str,        # Timeframe (M1, M5, H1, etc)
    data_ini: str,         # Data inicial (YYYY-MM-DD)
    data_fim: str,         # Data final (YYYY-MM-DD)
    tp: float,             # Take profit em pontos
    sl: float,             # Stop loss em pontos
    slippage: float,       # Slippage em pontos
    tc: float,             # Custo por transação
    lote: float,           # Tamanho do lote
    path_base: str,        # Caminho dos dados
    initial_cash: float,   # Capital inicial
    valor_lote: float,     # Valor do lote
    daytrade: bool         # Fechar posições no fim do dia
)
```

### Métodos Principais

- `run()`: Executa o backtest
- `print_metrics()`: Imprime relatório de métricas
- `plot_equity_curve()`: Plota curva de equity
- `plot_by_position()`: Plota performance por tipo de posição
- `plot_profit_by_hour()`: Plota lucro por hora
- `calculate_metrics()`: Calcula todas as métricas

## 💡 Exemplos

### Exemplo 1: Backtest Simples com Parâmetros Fixos

```python
# Exemplo com estratégia de médias móveis
from futures_backtester import Backtester

def sma_crossover(df, fast=20, slow=50):
    fast_ma = df['close'].rolling(fast).mean()
    slow_ma = df['close'].rolling(slow).mean()
    
    position = pd.Series(0, index=df.index)
    position[fast_ma > slow_ma] = 1
    position[fast_ma < slow_ma] = -1
    
    return position

# Executar backtest
bt = Backtester("EURUSD", "H1", "2023-01-01", "2023-12-31",
                tp=50, sl=30, slippage=1, tc=0.5, lote=1.0,
                path_base="./data/")

results, metrics = bt.run(signal_function=sma_crossover,
                         signal_args={"fast": 20, "slow": 50})
```

### Exemplo 2: Otimização com Filtro de Horário

```python
# Otimizar apenas horários de maior liquidez
optimizer = StrategyOptimizer(
    symbol="EURUSD",
    timeframe="M15",
    data_ini="2023-01-01",
    data_fim="2023-12-31",
    optimize_metric='sharpe_ratio'
)

# Otimizar apenas horário europeu e americano
optimizer.run(
    signal_function=bollinger_bands,
    param_ranges={
        'tp': (30, 100),
        'sl': (20, 60),
        'bb_length': (15, 25),
        'std': (1.8, 2.5, 0.1)
    },
    hours_to_optimize=[8, 9, 10, 11, 14, 15, 16, 17]
)
```

## ⚡ Performance

O framework é otimizado para alta performance:

- **Numba JIT**: Cálculos de stop loss/take profit até 100x mais rápidos
- **Vectorização**: Operações pandas otimizadas
- **Multiprocessing**: Otimização paralela de estratégias
- **Memory Efficient**: Processamento eficiente de grandes datasets

### Benchmarks

| Dataset Size | Backtest Time | Optimization Time (100 trials) |
|-------------|---------------|--------------------------------|
| 1 ano (M5)  | ~0.5s        | ~5 min                         |
| 5 anos (M5) | ~2.5s        | ~25 min                        |
| 10 anos (M5)| ~5s          | ~50 min                        |

## 🛠️ Desenvolvimento

### Estrutura do Projeto

```
futures_backtester/
├── futures_backtester/
│   ├── __init__.py
│   ├── backtester.py         # Classe principal
│   ├── strategy_generator.py # Otimizador de estratégias
│   ├── strategies/           # Estratégias built-in
│   │   ├── __init__.py
│   │   └── indicators.py
│   └── utils/                # Utilitários
│       ├── __init__.py
│       ├── stop_utils.py     # Cálculo de stops (Numba)
│       ├── drawdown_utils.py # Análise de drawdown
│       └── metrics_utils.py  # Cálculo de métricas
├── examples/                 # Exemplos de uso
├── tests/                    # Testes unitários
├── data/                     # Dados de exemplo
├── requirements.txt
├── setup.py
└── README.md
```

### Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Rodando Testes

```bash
# Instalar dependências de desenvolvimento
pip install -r requirements-dev.txt

# Rodar testes
pytest tests/

# Rodar com coverage
pytest --cov=futures_backtester tests/
```

## ⚠️ Avisos Importantes

- **DADOS HISTÓRICOS**: Certifique-se de ter dados de qualidade em formato Parquet
- **RISCO**: Sempre faça paper trading antes de usar estratégias em conta real
- **OVERFITTING**: Cuidado com otimização excessiva - use validação out-of-sample
- **CUSTOS**: Considere spreads e custos reais de sua corretora

## 🔮 Roadmap

- [ ] Suporte para múltiplos ativos simultâneos
- [ ] Machine Learning integrado
- [ ] Walk-forward optimization
- [ ] Risk management avançado
- [ ] Integração com brokers via API
- [ ] Dashboard web interativo

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Autores

- **Renato Critelli** - *Desenvolvimento inicial* - [GitHub](https://github.com/renatophys)

## 📞 Suporte

- 📧 Email: renato.critelli.ifusp@gmail.com
- 💬 Discord: [Link do servidor](https://discord.gg/seu_link)
- 📖 Documentação: [futures-backtester.readthedocs.io](https://futures-backtester.readthedocs.io)

---

<p align="center">
  Feito com ❤️ para a comunidade de trading algorítmico
</p>