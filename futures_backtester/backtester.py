"""
Módulo principal para backtesting de estratégias sistemáticas em FOREX.
"""

import os
import numpy as np
import pandas as pd
from .utils.stop_utils import compute_stop_loss_take_profit
from .utils.drawdown_utils import calc_dd  # Importa a nova função


class Backtester:
    """Classe para backtesting de estratégias sistemáticas 

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
        initial_cash (float, optional): Saldo inicial em unidades monetárias. Padrão 10000.0.
        valor_lote (float, optional): Valor do lote em unidades da moeda base. Padrão 10^5.
        daytrade (bool, optional): Flag para saber se fechamos a posição no fim do pregao diário.
    """
    
    def __init__(self, symbol, timeframe, data_ini, data_fim, tp, sl, slippage, tc, lote, path_base, 
                 initial_cash=10000.0, valor_lote=10**5, daytrade=True):
        """Inicializa o backtester com parâmetros configuráveis."""
        if not os.path.exists(path_base):
            raise ValueError(f"Diretório {path_base} não existe.")
        if not isinstance(tp, (int, float)) or tp <= 0:
            raise ValueError("Take-profit deve ser um número positivo.")
        if not isinstance(initial_cash, (int, float)) or initial_cash <= 0:
            raise ValueError("Saldo inicial deve ser um número positivo.")
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
        self.initial_cash = initial_cash
        self.valor_lote = valor_lote
        self.daytrade = daytrade
        self.path_base = path_base
        self.df = None
        self.dd_metrics = None  # Adiciona atributo para métricas de drawdown
        

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
        
              
        if signal_function:
            args = signal_args or {}
            self.df['position'] = signal_function(self.df, **args)
        else:
            # Estratégia padrão simples
            self.df['pct_change'] = self.df['close'].pct_change().fillna(0)
            self.df['position'] = np.where(self.df['pct_change'] > 0, 1, -1)

            
    def add_indices(self):
        """Adiciona índices e valores finais por safra."""
        self.df['idx'] = np.arange(self.df.shape[0])        

        # Se for daytrading, paramos no fim do dia
        if self.daytrade:  
            self.df['close_final'] = self.df.groupby('safra')['close'].transform('last')
            self.df['idx_final'] = self.df.groupby('safra')['idx'].transform('last')
        else:
            self.df['close_final'] = self.df['close'].iloc[-1]
            self.df['idx_final'] = self.df['idx'].iloc[-1]

        
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
        self.df['pts_final'] = 0.0
        self.df.loc[self.df['status_trade'] == 1, 'pts_final'] = self.tp
        self.df.loc[self.df['status_trade'] == -1, 'pts_final'] = -self.sl

        filtro_buy = (self.df['status_trade'] == 0) & (self.df['position'] == 1)
        self.df.loc[filtro_buy, 'pts_final'] = self.df.loc[filtro_buy, 'close_final'] - self.df.loc[filtro_buy, 'close']

        filtro_sell = (self.df['status_trade'] == 0) & (self.df['position'] == -1)
        self.df.loc[filtro_sell, 'pts_final'] = self.df.loc[filtro_sell, 'close'] - self.df.loc[filtro_sell, 'close_final']

        #self.df['strategy'] = self.valor_lote * self.lote * (self.df['pts_final']) - 2*self.valor_lote * self.lote * self.tc
        self.df['strategy'] = 0.0
        self.df.loc[self.df['position']!=0,'strategy'] = self.valor_lote * self.lote * (self.df[self.df['position']!=0]['pts_final']) - 2*self.lote * self.tc
        self.df['cstrategy'] = self.df['strategy'].cumsum()

        # Adicionar coluna de equidade (equity) considerando saldo inicial
        self.df['equity'] = self.initial_cash + self.df['cstrategy']
        
        # Calcula retornos diários em percentual sobre o capital
        self.df['daily_returns_pct'] = self.df['strategy'] / self.df['equity'].shift(1).fillna(self.initial_cash)
        
        # Calcula drawdown e time underwater
        dd_df = calc_dd(self.df['equity'])
        
        # Adiciona as colunas de drawdown ao dataframe principal
        self.df['cummax'] = dd_df['cummax']
        self.df['drawdown'] = dd_df['drawdown']
        self.df['drawdown_pct'] = dd_df['drawdown_pct']
        self.df['underwater'] = dd_df['underwater']
        self.df['time_uw'] = dd_df['time_uw']

        
    def calculate_metrics(self, risk_free_rate=0.0):
        """
        Calcula métricas detalhadas de desempenho da estratégia.
        
        Args:
            risk_free_rate (float): Taxa livre de risco anual em decimal (ex: 0.02 = 2% ao ano)
        """
        
        from .utils.metrics_utils import calculate_comprehensive_metrics
    
        return calculate_comprehensive_metrics(
            df=self.df,
            initial_cash=self.initial_cash,
            risk_free_rate=risk_free_rate
        )

    def print_metrics(self, metrics=None):
        """Imprime as métricas de desempenho."""
        from .utils.metrics_utils import format_metrics_report
        
        if metrics is None:
            metrics = self.calculate_metrics()
        
        report = format_metrics_report(
            metrics=metrics,
            symbol=self.symbol,
            timeframe=self.timeframe,
            period=f"{self.data_ini} a {self.data_fim}"
        )
        
        print(report)


    def run(self, signal_function=None, signal_args=None):
        """
        Executa o backtest completo.
        
        Args:
            signal_function (callable, optional): Função de sinal para definir posições.
            signal_args (dict, optional): Argumentos adicionais para a função de sinal.
            
        Returns:
            tuple: (DataFrame com resultados, dicionário de métricas)
        """
        
        # Armazenar informações da estratégia
        self.signal_function_name = signal_function.__name__ if signal_function else "default"
        self.signal_args = signal_args or {}
        
        self.load_data()
        self.define_positions(signal_function, signal_args)
        self.add_indices()
        self.calculate_stops()
        self.calculate_results()
        return self.df, self.calculate_metrics()
        
    
    def plot_equity_curve(self, figsize=(12, 6), include_drawdown=True):
        """
        Plota a curva de equity da estratégia e opcionalmente o drawdown.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
            include_drawdown (bool): Se True, inclui um gráfico de drawdown
        """
        import matplotlib.pyplot as plt
        
        # Criar string com os parâmetros da estratégia
        strategy_params = f"Estratégia: {getattr(self, 'signal_function_name', 'N/A')}"
        
        # Adicionar parâmetros da estratégia arredondados
        if hasattr(self, 'signal_args') and self.signal_args:
            rounded_params = {}
            for k, v in self.signal_args.items():
                # Arredondar valores numéricos para 2 casas decimais
                if isinstance(v, (int, float)):
                    rounded_params[k] = round(v, 2)
                else:
                    rounded_params[k] = v
            
            params_str = ', '.join([f"{k}={v}" for k, v in rounded_params.items()])
            strategy_params += f" ({params_str})"
        
        # Adicionar SL e TP ao título
        sl_tp_info = f"SL: {round(self.sl, 2)}, TP: {round(self.tp, 2)}"
        
        # Configurações básicas de título
        base_title = f'Curva de Equity - {self.symbol} ({self.timeframe})'
        full_title = f'{base_title}\n{strategy_params}\n{sl_tp_info}'
        
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.5), 
                                            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            
            # Curva de equity
            ax1.plot(self.df.index, self.df['equity'])
            ax1.set_title(full_title)
            ax1.set_ylabel('Resultado ($)')
            ax1.grid(True)
            
            # Gráfico de drawdown
            ax2.fill_between(self.df.index, self.df['drawdown'], 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown ($)')
            ax2.set_ylabel('Drawdown ($)')
            ax2.set_xlabel('Data')
            ax2.grid(True)
            
            plt.tight_layout()
        else:
            plt.figure(figsize=figsize)
            plt.plot(self.df.index, self.df['equity'])
            plt.title(full_title)
            plt.xlabel('Data')
            plt.ylabel('Resultado ($)')
            plt.grid(True)
            plt.tight_layout()
            
        return plt

    def plot_drawdown(self, figsize=(12, 6)):
        """
        Plota apenas o drawdown da estratégia.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        plt.fill_between(self.df.index, self.df['drawdown_pct']*100, 0, color='red', alpha=0.3)
        plt.plot(self.df.index, self.df['time_uw'], color='blue', alpha=0.6, label='Tempo em Drawdown')
        plt.title(f'Análise de Drawdown - {self.symbol} ({self.timeframe})')
        plt.xlabel('Data')
        plt.ylabel('Drawdown (%) / Tempo')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
            
        return plt
    
    def plot_by_position(self, figsize=(12, 6)):
        """
        Plota a curva de equity separada por posições de compra e venda.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
            
        Returns:
            matplotlib.pyplot: Objeto pyplot com o gráfico
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        
        # Calcular resultados separados por posição (compra/venda)
        df_analysis = self.df.copy()
        df_analysis['profit_buy'] = 0.0
        df_analysis['profit_sell'] = 0.0
        
        # Atribuir lucros às respectivas posições
        df_analysis.loc[df_analysis['position'] == 1, 'profit_buy'] = df_analysis.loc[df_analysis['position'] == 1, 'strategy']
        df_analysis.loc[df_analysis['position'] == -1, 'profit_sell'] = df_analysis.loc[df_analysis['position'] == -1, 'strategy']
        
        # Calcular acumulados
        df_analysis['cstrategy_buy'] = df_analysis['profit_buy'].cumsum()
        df_analysis['cstrategy_sell'] = df_analysis['profit_sell'].cumsum()
        
        # Criar o gráfico
        plt.figure(figsize=figsize)
        plt.plot(df_analysis.index, df_analysis['cstrategy'], label='Total', linewidth=2)
        plt.plot(df_analysis.index, df_analysis['cstrategy_buy'], label='Compras', linewidth=1.5)
        plt.plot(df_analysis.index, df_analysis['cstrategy_sell'], label='Vendas', linewidth=1.5)
        
        # Formatação do gráfico
        plt.title(f'Retorno Acumulado por Tipo de Posição - {self.symbol} ({self.timeframe})')
        plt.xlabel('Data')
        plt.ylabel('Resultado ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt

    def plot_profit_by_hour(self, figsize=(14, 8)):
        """
        Plota o lucro acumulado por hora do dia, separado por posições de compra e venda.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
            
        Returns:
            matplotlib.pyplot: Objeto pyplot com o gráfico
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Extrair hora do dia
        df_analysis = self.df.copy()
        df_analysis['hour'] = df_analysis.index.hour
        
        # Preparar DataFrame para análise por hora
        hourly_results = {'hour': [], 'total': [], 'buy': [], 'sell': []}
        
        # Analisar todas as horas do dia (0-23)
        hours = sorted(df_analysis['hour'].unique())
        
        for hour in hours:
            # Filtrar dados para a hora específica
            hour_data = df_analysis[df_analysis['hour'] == hour]
            
            # Calcular resultados para essa hora
            total_profit = hour_data['strategy'].sum()
            buy_profit = hour_data.loc[hour_data['position'] == 1, 'strategy'].sum()
            sell_profit = hour_data.loc[hour_data['position'] == -1, 'strategy'].sum()
            
            # Adicionar resultados ao dicionário
            hourly_results['hour'].append(hour)
            hourly_results['total'].append(total_profit)
            hourly_results['buy'].append(buy_profit)
            hourly_results['sell'].append(sell_profit)
        
        # Criar DataFrame de resultados por hora
        hourly_df = pd.DataFrame(hourly_results)
        
        # Criar gráfico de barras
        plt.figure(figsize=figsize)
        
        bar_width = 0.25
        x = np.arange(len(hours))
        
        plt.bar(x - bar_width, hourly_df['total'], width=bar_width, label='Total', color='blue', alpha=0.7)
        plt.bar(x, hourly_df['buy'], width=bar_width, label='Compras', color='green', alpha=0.7)
        plt.bar(x + bar_width, hourly_df['sell'], width=bar_width, label='Vendas', color='red', alpha=0.7)
        
        # Formatação do gráfico
        plt.title(f'Lucro Total por Hora do Dia - {self.symbol} ({self.timeframe})')
        plt.xlabel('Hora do Dia')
        plt.ylabel('Resultado ($)')
        plt.xticks(x, hourly_df['hour'])
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt

    def plot_cumulative_by_hour(self, figsize=(14, 10)):
        """
        Plota o lucro acumulado ao longo do tempo por hora do dia,
        mostrando a evolução de cada hora separadamente.
        
        Args:
            figsize (tuple): Dimensões da figura (largura, altura)
            
        Returns:
            matplotlib.pyplot: Objeto pyplot com o gráfico
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import matplotlib.cm as cm
        
        # Cópia do DataFrame original
        df_analysis = self.df.copy()
        df_analysis['hour'] = df_analysis.index.hour
        df_analysis['date'] = df_analysis.index.date

        hours = sorted(df_analysis['hour'].unique())
        daily_index = pd.DatetimeIndex(df_analysis['date'].unique())
        daily_df = pd.DataFrame(index=daily_index)

        # Criar colunas de lucro por hora e tipo de posição
        for hour in hours:
            hour_key = f'profit_hour_{hour}'
            buy_key = f'profit_hour_buy_{hour}'
            sell_key = f'profit_hour_sell_{hour}'

            hour_mask = df_analysis['hour'] == hour
            buy_mask = hour_mask & (df_analysis['position'] == 1)
            sell_mask = hour_mask & (df_analysis['position'] == -1)

            df_analysis[hour_key] = 0.0
            df_analysis[buy_key] = 0.0
            df_analysis[sell_key] = 0.0

            df_analysis.loc[hour_mask, hour_key] = df_analysis.loc[hour_mask, 'strategy']
            df_analysis.loc[buy_mask, buy_key] = df_analysis.loc[buy_mask, 'strategy']
            df_analysis.loc[sell_mask, sell_key] = df_analysis.loc[sell_mask, 'strategy']

        # Gerar colunas de lucro diário agregadas
        daily_columns = []

        for hour in hours:
            hour_key = f'profit_hour_{hour}'
            buy_key = f'profit_hour_buy_{hour}'
            sell_key = f'profit_hour_sell_{hour}'

            grouped = df_analysis.groupby('date')[[hour_key, buy_key, sell_key]].sum()
            grouped.index = pd.to_datetime(grouped.index)
            daily_columns.append(grouped)

        daily_df = pd.concat([daily_df] + daily_columns, axis=1)

        # Calcular acumulados de forma vetorizada
        cum_columns = {}

        for hour in hours:
            hour_key = f'profit_hour_{hour}'
            buy_key = f'profit_hour_buy_{hour}'
            sell_key = f'profit_hour_sell_{hour}'

            cum_columns[f'cum_hour_{hour}'] = daily_df[hour_key].cumsum().fillna(0)
            cum_columns[f'cum_hour_buy_{hour}'] = daily_df[buy_key].cumsum().fillna(0)
            cum_columns[f'cum_hour_sell_{hour}'] = daily_df[sell_key].cumsum().fillna(0)

        daily_df = pd.concat([daily_df, pd.DataFrame(cum_columns, index=daily_df.index)], axis=1)
        
        # Criar gráficos
        # Criar paleta de cores para todas as horas
        hours = sorted([int(col.split('_')[-1]) for col in daily_df.columns if col.startswith('cum_hour_') and not 'buy' in col and not 'sell' in col])
        colors = cm.get_cmap('tab20', len(hours))
        
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # --- Gráfico 1: Total ---
        for i, hour in enumerate(hours):
            cum_hour_key = f'cum_hour_{hour}'
            axs[0].plot(daily_df.index, daily_df[cum_hour_key], label=f'{hour:02d}h', color=colors(i))
        axs[0].set_title(f'Lucro Acumulado por Hora do Dia - {self.symbol} ({self.timeframe})')
        axs[0].set_ylabel('Resultado ($)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, ncol=2)

        # --- Gráfico 2: Compras ---
        for i, hour in enumerate(hours):
            cum_buy_key = f'cum_hour_buy_{hour}'
            axs[1].plot(daily_df.index, daily_df[cum_buy_key], label=f'{hour:02d}h', color=colors(i))
        axs[1].set_title('Lucro Acumulado por Hora - Apenas Compras')
        axs[1].set_ylabel('Resultado ($)')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, ncol=2)

        # --- Gráfico 3: Vendas ---
        for i, hour in enumerate(hours):
            cum_sell_key = f'cum_hour_sell_{hour}'
            axs[2].plot(daily_df.index, daily_df[cum_sell_key], label=f'{hour:02d}h', color=colors(i))
        axs[2].set_title('Lucro Acumulado por Hora - Apenas Vendas')
        axs[2].set_xlabel('Data')
        axs[2].set_ylabel('Resultado ($)')
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, ncol=2)

        fig.suptitle('Análise de Lucro Acumulado por Hora', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        return plt