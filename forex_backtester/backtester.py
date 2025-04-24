"""
Módulo principal para backtesting de estratégias sistemáticas em FOREX.
"""

import os
import numpy as np
import pandas as pd
from .utils.stop_utils import compute_stop_loss_take_profit


class Backtester:
    """Classe para backtesting de estratégias sistemáticas em FOREX.

    Args:
        symbol (str): Símbolo do par de moedas (ex: 'EURUSD').
        timeframe (str): Timeframe dos dados (ex: 'M1', 'H1').
        data_ini (str): Data inicial do backtest (formato 'YYYY-MM-DD').
        data_fim (str): Data final do backtest (formato 'YYYY-MM-DD').
        tp (float): Take-profit em pontos.
        sl (float): Stop-loss em pontos.
        slippage (float): Slippage em pontos.
        tc (float): Custo de transação por trade.
        lote (float): Tamanho do lote.
        path_base (str): Caminho para os arquivos de dados históricos.
        valor_lote (float, optional): Valor do lote em unidades da moeda base. Padrão 10^5.
        trading_hours (int, optional): Horas de trading por dia. Padrão 23.
    """
    
    def __init__(self, symbol, timeframe, data_ini, data_fim, tp, sl, slippage, tc, lote, path_base, valor_lote=10**5, trading_hours=23):
        """Inicializa o backtester com parâmetros configuráveis."""
        if not os.path.exists(path_base):
            raise ValueError(f"Diretório {path_base} não existe.")
        if not isinstance(tp, (int, float)) or tp <= 0:
            raise ValueError("Take-profit deve ser um número positivo.")
        # Outras validações...
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_ini = data_ini
        self.data_fim = data_fim
        self.tp = tp
        self.sl = sl
        self.slippage = slippage
        self.tc = tc
        self.lote = lote
        self.valor_lote = valor_lote
        self.trading_hours = trading_hours
        self.path_base = path_base
        self.df = None
        

    def load_data(self):
        """Carrega os dados históricos do arquivo Parquet."""
        try:
            tabela = f'{self.symbol}_{self.timeframe}'
            self.df = pd.read_parquet(f'{self.path_base}{tabela}.parquet')
            self.df['safra'] = self.df['time'].dt.date.astype(str)
            self.df = self.df.set_index('time')[self.data_ini:self.data_fim]
            if self.df.empty:
                raise ValueError("Nenhum dado encontrado no intervalo especificado.")
        except Exception as e:
            raise Exception(f"Erro ao carregar dados: {str(e)}")
            

    def define_positions(self, signal_function=None, signal_args=None):
        """
        Define as posições com base em uma função de sinal ou estratégia padrão.
        
        Args:
            signal_function (callable, optional): Função que retorna uma série com valores [-1, 0, 1]
                representando posições (short, neutro, long). A função deve aceitar o dataframe como
                primeiro argumento.
            signal_args (dict, optional): Argumentos adicionais para passar para a função de sinal.
        """
        
        self.df['tempo_final'] = pd.to_datetime(self.df['safra']) + pd.to_timedelta(self.trading_hours, unit='h')
        self.df['pct_change'] = self.df['close'].pct_change().fillna(0)
        
        if signal_function:
            args = signal_args or {}
            self.df['position'] = signal_function(self.df, **args)
        else:
            # Estratégia padrão simples
            self.df['position'] = np.where(self.df['pct_change'] > 0, 1, -1)

            
    def add_indices(self):
        """Adiciona índices e valores finais por safra."""
        self.df['idx'] = np.arange(self.df.shape[0])
        self.df['close_final'] = self.df.groupby('safra')['close'].transform('last')
        self.df['idx_final'] = self.df.groupby('safra')['idx'].transform('last')

        
    def calculate_stops(self):
        """Calcula os stops (TP/SL) usando função externa."""
        status_trade, sl_idx, tp_idx = compute_stop_loss_take_profit(
            self.df['position'].values,
            self.df['high'].values,
            self.df['low'].values,
            self.df['close'].values,
            self.sl, self.tp, self.slippage,
            self.df['idx'].values,
            self.df['idx_final'].values
        )
        self.df['status_trade'] = status_trade
        self.df['sl_idx'] = sl_idx
        self.df['tp_idx'] = tp_idx

        
    def calculate_results(self):
        """Calcula os resultados financeiros da estratégia."""
        self.df['pts_final'] = 0
        self.df.loc[self.df['status_trade'] == 1, 'pts_final'] = self.tp
        self.df.loc[self.df['status_trade'] == -1, 'pts_final'] = -self.sl

        filtro_buy = (self.df['status_trade'] == 0) & (self.df['position'] == 1)
        self.df.loc[filtro_buy, 'pts_final'] = self.df.loc[filtro_buy, 'close_final'] - self.df.loc[filtro_buy, 'close']

        filtro_sell = (self.df['status_trade'] == 0) & (self.df['position'] == -1)
        self.df.loc[filtro_sell, 'pts_final'] = self.df.loc[filtro_sell, 'close'] - self.df.loc[filtro_sell, 'close_final']

        self.df['strategy'] = self.valor_lote * self.lote * (self.df['pts_final']) - 2*self.valor_lote * self.lote * self.tc
        self.df['cstrategy'] = self.df['strategy'].cumsum()

        
    def calculate_metrics(self):
        """Calcula métricas detalhadas de desempenho da estratégia."""
        # Resultados básicos
        total_return = self.df['cstrategy'].iloc[-1]

        # Cálculo de drawdown
        drawdown_series = (self.df['cstrategy'].cummax() - self.df['cstrategy']) / self.df['cstrategy'].cummax().replace(0, np.nan)
        max_drawdown = drawdown_series.max() if not drawdown_series.empty else 0

        # Análise de trades
        # Identifica início de cada trade (quando position muda de valor)
        self.df['trade_start'] = self.df['position'].diff().ne(0) & (self.df['position'] != 0)

        # Estatísticas de trades
        total_trades = self.df['trade_start'].sum()
        if total_trades > 0:
            win_trades = (self.df[self.df['trade_start']]['strategy'] > 0).sum()
            loss_trades = (self.df[self.df['trade_start']]['strategy'] < 0).sum()
            win_rate = win_trades / total_trades if total_trades > 0 else 0

            # Análise de razão de saídas
            tp_hits = (self.df[self.df['trade_start']]['status_trade'] == 1).sum()
            sl_hits = (self.df[self.df['trade_start']]['status_trade'] == -1).sum()
            time_exits = (self.df[self.df['trade_start']]['status_trade'] == 0).sum()

            tp_rate = tp_hits / total_trades if total_trades > 0 else 0
            sl_rate = sl_hits / total_trades if total_trades > 0 else 0
            time_exit_rate = time_exits / total_trades if total_trades > 0 else 0

            # Profit factor (ganhos brutos / perdas brutas)
            gross_profits = self.df.loc[self.df['strategy'] > 0, 'strategy'].sum()
            gross_losses = abs(self.df.loc[self.df['strategy'] < 0, 'strategy'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')

            # Métricas de risco
            daily_returns = self.df['strategy'].resample('D').sum()
            annual_return = daily_returns.mean() * 252
            annual_volatility = daily_returns.std() * np.sqrt(252)

            # Diversos ratios
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            sortino_ratio = annual_return / daily_returns[daily_returns < 0].std() * np.sqrt(252) if len(daily_returns[daily_returns < 0]) > 0 else 0
            calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0

            # Média de ganhos e perdas
            avg_win = gross_profits / win_trades if win_trades > 0 else 0
            avg_loss = gross_losses / loss_trades if loss_trades > 0 else 0
            win_loss_ratio = avg_win / abs(avg_loss) if loss_trades > 0 else float('inf')

            # Expectativa matemática (Expectancy)
            expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if (win_trades > 0 and loss_trades > 0) else 0

        else:
            # Valores padrão se não houver trades
            win_rate = 0
            tp_rate = sl_rate = time_exit_rate = 0
            profit_factor = 0
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            avg_win = avg_loss = win_loss_ratio = expectancy = 0
            win_trades = loss_trades = 0
    
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'tp_rate': tp_rate,  # % de trades que atingiram take profit
            'sl_rate': sl_rate,  # % de trades que atingiram stop loss
            'time_exit_rate': time_exit_rate,  # % de trades que fecharam por tempo
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy
        }

    def print_metrics(self, metrics=None):
        """Imprime as métricas de desempenho da estratégia de forma organizada."""
        if metrics is None:
            metrics = self.calculate_metrics()

        print("\n===== RELATÓRIO DE DESEMPENHO =====")
        print(f"Símbolo: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"Período: {self.data_ini} a {self.data_fim}")
        print("\n--- RESULTADOS ---")
        print(f"Retorno Total: ${metrics['total_return']:.2f}")
        print(f"Drawdown Máximo: {metrics['max_drawdown']:.2%}")

        print("\n--- TRADES ---")
        print(f"Total de Trades: {metrics['total_trades']}")
        print(f"Trades Vencedores: {metrics['win_trades']} ({metrics['win_rate']:.2%})")
        print(f"Trades Perdedores: {metrics['loss_trades']} ({1-metrics['win_rate']:.2%})")
        print(f"Saídas por TP: {metrics['tp_rate']:.2%}")
        print(f"Saídas por SL: {metrics['sl_rate']:.2%}")
        print(f"Saídas por Tempo: {metrics['time_exit_rate']:.2%}")

        print("\n--- RATIOS ---")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"Profit Factor: {metrics['profit_factor']:.3f}")

        print("\n--- ANÁLISE DE GANHOS/PERDAS ---")
        print(f"Ganho Médio: ${metrics['avg_win']:.2f}")
        print(f"Perda Média: ${metrics['avg_loss']:.2f}")
        print(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
        print(f"Expectancy: ${metrics['expectancy']:.2f}")
        print("===================================\n")

    
    def run(self, signal_function=None, signal_args=None):
        """
        Executa o backtest completo.
        
        Args:
            signal_function (callable, optional): Função de sinal para definir posições.
            signal_args (dict, optional): Argumentos adicionais para a função de sinal.
            
        Returns:
            tuple: (DataFrame com resultados, dicionário de métricas)
        """
        
        self.load_data()
        self.define_positions(signal_function, signal_args)
        self.add_indices()
        self.calculate_stops()
        self.calculate_results()
        return self.df, self.calculate_metrics()
    
    
    def plot_equity_curve(self, figsize=(12, 6)):
        """
        Plota a curva de equity da estratégia.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        plt.plot(self.df.index, self.df['cstrategy'])
        plt.title(f'Curva de Equity - {self.symbol} ({self.timeframe})')
        plt.xlabel('Data')
        plt.ylabel('Resultado ($)')
        plt.grid(True)
        plt.tight_layout()
        return plt