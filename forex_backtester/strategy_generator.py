"""
Forex Strategy Optimizer

Este módulo implementa uma estrutura OO para otimização de estratégias de trading FOREX
usando o framework forex_backtester e otimização bayesiana via Optuna.

Características:
- Otimização por horário do dia
- Exportação de resultados e parâmetros ótimos
- Visualização dos resultados por horário
"""

import os
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from .backtester import Backtester


class StrategyOptimizer:
    """
    Classe principal para otimização de estratégias FOREX.
    
    Esta classe encapsula todo o processo de otimização, validação e teste
    de estratégias de trading, com foco na otimização por horário do dia.
    """
    
    def __init__(self, 
                 symbol, 
                 timeframe, 
                 data_ini, 
                 data_fim, 
                 initial_cash=30000, 
                 lote=0.01, 
                 slippage=0, 
                 daytrade=True, 
                 path_base='./data/', 
                 num_trials=50, 
                 max_workers=4, 
                 export_dir='./resultados',
                 tc=0.5,
                 valor_lote=100000):
        """
        Inicializa o otimizador de estratégias.
        
        Args:
            symbol (str): Símbolo a ser negociado (ex: 'USDJPY')
            timeframe (str): Timeframe para análise (ex: 't5')
            data_ini (str): Data inicial para backtest (formato: 'YYYY-MM-DD')
            data_fim (str): Data final para backtest (formato: 'YYYY-MM-DD')
            initial_cash (float): Capital inicial para backtest
            lote (float): Tamanho do lote para operações
            slippage (float): Slippage em pontos
            daytrade (bool): Se True, fecha posições no final do dia
            path_base (str): Caminho para os arquivos de dados históricos
            num_trials (int): Número de tentativas por otimização
            max_workers (int): Número de processos paralelos
            export_dir (str): Diretório para exportação de resultados
            tc (float): Custo de transação por lote
            valor_lote (float): Valor do lote em unidades da moeda base
        """
        # Armazenar configurações
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_ini = data_ini
        self.data_fim = data_fim
        self.initial_cash = initial_cash
        self.lote = lote
        self.slippage = slippage
        self.daytrade = daytrade
        self.path_base = path_base
        self.num_trials = num_trials
        self.max_workers = max_workers
        self.export_dir = export_dir
        self.tc = tc
        self.valor_lote = valor_lote
        
        # Estado interno
        self.run_dir = None
        self.all_results = []
        self.validation_results = []
        self.combined_strategy = None
        self.param_ranges = {}
        self.strategy_function = None
        self.fixed_params = {}
        
    def setup_dirs(self):
        """
        Configura diretórios necessários para resultados e logs.
        
        Returns:
            str: Caminho para o diretório da execução atual
        """
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
        
        # Criar diretório específico para essa execução com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.export_dir, f"run_{timestamp}")
        os.makedirs(run_dir)
        
        # Armazenar diretório da execução
        self.run_dir = run_dir
        
        # Salvar configuração desta execução
        self._save_config()
        
        return run_dir
    
    def _save_config(self):
        """Salva as configurações da execução atual."""
        config = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'data_ini': self.data_ini,
            'data_fim': self.data_fim,
            'initial_cash': self.initial_cash,
            'lote': self.lote,
            'slippage': self.slippage,
            'daytrade': self.daytrade,
            'path_base': self.path_base,
            'num_trials': self.num_trials,
            'max_workers': self.max_workers,
            'export_dir': self.export_dir,
            'tc': self.tc,
            'valor_lote': self.valor_lote,
            'param_ranges': self.param_ranges,
            'fixed_params': self.fixed_params,
            'strategy': self.strategy_function.__name__ if self.strategy_function else None
        }
        
        with open(os.path.join(self.run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
            
    def set_param_ranges(self, param_ranges):
        """
        Define os ranges de parâmetros a serem otimizados.
        
        Args:
            param_ranges (dict): Dicionário com os ranges de parâmetros.
                Ex: {'tp': (10, 100), 'sl': (5, 50), 'bb_length': (5, 30), 'std': (1.0, 3.0, 0.1)}
                Cada range pode ser:
                    - Tupla (min, max) para int ou float
                    - Tupla (min, max, step) para float com step
                    - Lista de valores para categoricos
        """
        self.param_ranges = param_ranges
        return self
        
    def set_fixed_params(self, fixed_params):
        """
        Define parâmetros fixos (não otimizados).
        
        Args:
            fixed_params (dict): Dicionário com os parâmetros fixos.
                Ex: {'is_async': True}
        """
        self.fixed_params = fixed_params
        return self
        
    def _objective_hour(self, trial, hour):
        """
        Função objetivo para otimização com Optuna, específica para um horário.
        
        Args:
            trial: Objeto trial do Optuna
            hour: Hora específica para otimização
            
        Returns:
            float: Valor da métrica a ser otimizada
        """
        # Gerar parâmetros a partir dos ranges definidos
        params = {}
        
        for param_name, param_range in self.param_ranges.items():
            if isinstance(param_range, list):
                # Parâmetro categórico
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif len(param_range) == 2:
                # Range simples (min, max)
                min_val, max_val = param_range
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            elif len(param_range) == 3:
                # Range com step (min, max, step)
                min_val, max_val, step = param_range
                params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)
        
        # Adicionar parâmetros fixos
        params.update(self.fixed_params)
        
        # Adicionar allowed_hours
        signal_args = params.copy()
        signal_args['allowed_hours'] = [hour]
        
        # Extrair tp e sl (se existirem) para o backtester
        tp = params.pop('tp', 30)  # Default
        sl = params.pop('sl', 20)  # Default
        
        # Lógica de validação TP > SL (opcional)
        if 'tp' in self.param_ranges and 'sl' in self.param_ranges:
            if tp <= sl:
                return float('-inf')  # Penalizar estas combinações
        
        # Configurar o backtester
        bt = Backtester(
            symbol=self.symbol,
            timeframe=self.timeframe,
            data_ini=self.data_ini,
            data_fim=self.data_fim,
            tp=tp,
            sl=sl,
            slippage=self.slippage,
            tc=self.tc,
            lote=self.lote,
            valor_lote=self.valor_lote,
            initial_cash=self.initial_cash,
            path_base=self.path_base,
            daytrade=self.daytrade
        )
        
        # Executar backtest focando apenas na hora específica
        try:
            _, metrics = bt.run(
                signal_function=self.strategy_function,
                signal_args=signal_args
            )
            
            # Podemos usar diferentes métricas, dependendo do objetivo
            metric_value = metrics['sortino_ratio']
            
            # Adicionar métricas adicionais para o Optuna armazenar
            trial.set_user_attr('profit_factor', metrics['profit_factor'])
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('total_return', metrics['total_return'])
            trial.set_user_attr('trades', metrics['total_trades'])
            
            return metric_value
        
        except Exception as e:
            print(f"Erro no backtest para hora {hour}: {str(e)}")
            return float('-inf')
            
    def optimize_hour(self, hour):
        """
        Otimiza estratégia para uma hora específica.
        
        Args:
            hour (int): Hora para otimização (0-23)
            
        Returns:
            dict: Resultados da otimização
        """
        print(f"\n{'='*50}")
        print(f"Otimizando estratégia para hora: {hour:02d}:00")
        print(f"{'='*50}")
        
        # Criar função objetivo específica para a hora
        objective = partial(self._objective_hour, hour=hour)
        
        # Configurar e executar o estudo Optuna
        study_name = f"hora_{hour:02d}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name
        )
        
        study.optimize(objective, n_trials=self.num_trials)
        
        # Extrair melhores parâmetros e resultados
        best_params = study.best_trial.params
        best_value = study.best_value
        best_metrics = {
            'profit_factor': study.best_trial.user_attrs.get('profit_factor', 0),
            'win_rate': study.best_trial.user_attrs.get('win_rate', 0),
            'max_drawdown': study.best_trial.user_attrs.get('max_drawdown', 0),
            'total_return': study.best_trial.user_attrs.get('total_return', 0),
            'trades': study.best_trial.user_attrs.get('trades', 0)
        }
        
        # Adicionar parâmetros fixos nos resultados
        best_params.update(self.fixed_params)
        
        # Salvar resultados
        results = {
            'hour': hour,
            'best_params': best_params,
            'best_value': best_value,
            'metrics': best_metrics
        }
        
        # Salvar em arquivo JSON
        with open(os.path.join(self.run_dir, f"results_hour_{hour:02d}.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nMelhores parâmetros para hora {hour:02d}:")
        print(f"Sortino Ratio: {best_value:.4f}")
        print(f"Parâmetros: {best_params}")
        print(f"Profit Factor: {best_metrics['profit_factor']:.4f}")
        print(f"Win Rate: {best_metrics['win_rate']:.2%}")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.2%}")
        print(f"Total Return: ${best_metrics['total_return']:.2f}")
        print(f"Total Trades: {best_metrics['trades']}")
        
        return results
    
    def optimize_all_hours(self, hours_to_optimize=None):
        """
        Otimiza estratégia para todas as horas especificadas.
        
        Args:
            hours_to_optimize (list): Lista de horas para otimizar (default: todas)
            
        Returns:
            list: Resultados da otimização para todas as horas
        """
        if hours_to_optimize is None:
            hours_to_optimize = list(range(24))  # 0-23
        
        # Garantir que o diretório de execução foi criado
        if self.run_dir is None:
            self.setup_dirs()
            
        # Otimizar cada hora (em paralelo se possível)
        self.all_results = []
        
        if self.max_workers > 1:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.optimize_hour, hour) for hour in hours_to_optimize]
                for future in futures:
                    self.all_results.append(future.result())
        else:
            for hour in hours_to_optimize:
                result = self.optimize_hour(hour)
                self.all_results.append(result)
                
        return self.all_results
    
    def validate_best_params(self, validation_period=None):
        """
        Valida os melhores parâmetros encontrados em um período separado (opcional).
        
        Args:
            validation_period (tuple): Tupla (data_ini, data_fim) para validação
        
        Returns:
            list: Resultados da validação
        """
        if not self.all_results:
            raise ValueError("Nenhum resultado de otimização encontrado. Execute optimize_all_hours primeiro.")
            
        if validation_period is None:
            # Se não for especificado, usamos o mesmo período (subótimo, mas aceitável)
            validation_period = (self.data_ini, self.data_fim)
        
        # Criar um dataframe com os resultados por hora
        self.validation_results = []
        
        for res in self.all_results:
            hour = res['hour']
            params = res['best_params']
            
            # Extrair tp e sl (se existirem) para o backtester
            tp = params.get('tp', 30)  # Default
            sl = params.get('sl', 20)  # Default
            
            # Configurar o backtester para o período de validação
            bt = Backtester(
                symbol=self.symbol,
                timeframe=self.timeframe,
                data_ini=validation_period[0],
                data_fim=validation_period[1],
                tp=tp,
                sl=sl,
                slippage=self.slippage,
                tc=self.tc,
                lote=self.lote,
                valor_lote=self.valor_lote,
                initial_cash=self.initial_cash,
                path_base=self.path_base,
                daytrade=self.daytrade
            )
            
            # Preparar signal_args (sem tp/sl)
            signal_args = {k: v for k, v in params.items() if k not in ['tp', 'sl']}
            signal_args['allowed_hours'] = [hour]
            
            # Executar backtest com os melhores parâmetros
            df, metrics = bt.run(
                signal_function=self.strategy_function,
                signal_args=signal_args
            )
            
            # Salvar gráfico de equity para essa hora
            plt.figure(figsize=(12, 6))
            plt.plot(df['equity'])
            plt.title(f"Curva de Equity - Hora {hour:02d}")
            plt.grid(True)
            plt.savefig(os.path.join(self.run_dir, f"equity_hour_{hour:02d}.png"))
            plt.close()
            
            # Adicionar resultados da validação
            self.validation_results.append({
                'hour': hour,
                'params': params,
                'metrics': {
                    'sortino_ratio': metrics['sortino_ratio'],
                    'profit_factor': metrics['profit_factor'],
                    'win_rate': metrics['win_rate'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_return': metrics['total_return'],
                    'total_trades': metrics['total_trades'],
                    'annual_return': metrics['annual_return']
                }
            })
        
        # Salvar resultados da validação
        with open(os.path.join(self.run_dir, "validation_results.json"), 'w') as f:
            json.dump(self.validation_results, f, indent=4)
        
        return self.validation_results
    
    def create_combined_strategy(self, min_threshold=0.5):
        """
        Cria uma estratégia combinada usando apenas as horas com bons resultados.
        
        Args:
            min_threshold (float): Threshold mínimo para o sortino ratio
            
        Returns:
            dict: Configuração da estratégia combinada
        """
        if not self.validation_results:
            raise ValueError("Nenhum resultado de validação encontrado. Execute validate_best_params primeiro.")
            
        # Filtrar apenas horas com bom desempenho
        good_hours = []
        good_params = {}
        
        for res in self.validation_results:
            if res['metrics']['sortino_ratio'] > min_threshold:
                hour = res['hour']
                good_hours.append(hour)
                good_params[str(hour)] = res['params']  # Usar str como chave para serialização JSON
        
        # Verificar se temos horas suficientes
        if not good_hours:
            print("Nenhuma hora passou pelo threshold mínimo!")
            return None
        
        # Ordenar horas
        good_hours.sort()
        
        # Criar configuração da estratégia combinada
        self.combined_strategy = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': self.strategy_function.__name__,
            'hours': good_hours,
            'hour_params': good_params,
            'tc': self.tc,
            'valor_lote': self.valor_lote
        }
        
        # Salvar configuração da estratégia combinada
        with open(os.path.join(self.run_dir, "combined_strategy.json"), 'w') as f:
            json.dump(self.combined_strategy, f, indent=4)
        
        print(f"\nEstratégia combinada criada com {len(good_hours)} horas:")
        print(f"Horas selecionadas: {good_hours}")
        
        return self.combined_strategy
    
    def test_combined_strategy(self):
        """
        Testa a estratégia combinada.
        
        Returns:
            tuple: DataFrame de resultados e métricas
        """
        if self.combined_strategy is None:
            raise ValueError("Nenhuma estratégia combinada encontrada. Execute create_combined_strategy primeiro.")
        
        # Criamos um backtester para a estratégia combinada com valores default
        bt = Backtester(
            symbol=self.symbol,
            timeframe=self.timeframe,
            data_ini=self.data_ini,
            data_fim=self.data_fim,
            tp=30,  # Não importa, será sobrescrito
            sl=20,  # Não importa, será sobrescrito
            slippage=self.slippage,
            tc=self.tc,
            lote=self.lote,
            valor_lote=self.valor_lote,
            initial_cash=self.initial_cash,
            path_base=self.path_base,
            daytrade=self.daytrade
        )
        
        # Carregar dados para poder modificá-los
        bt.load_data()
        
        # Criar uma coluna para armazenar as posições
        bt.df['position'] = 0
        
        # Para cada hora, aplicar a estratégia com os parâmetros otimizados
        for hour in self.combined_strategy['hours']:
            str_hour = str(hour)  # Converter para string para acessar o dicionário
            params = self.combined_strategy['hour_params'][str_hour]
            
            # Extrair tp e sl dos parâmetros (se existirem)
            signal_args = {k: v for k, v in params.items() if k not in ['tp', 'sl']}
            
            # Criar uma máscara para a hora atual
            hour_mask = bt.df.index.hour == hour
            
            # Calcular sinais apenas para essa hora
            signals = self.strategy_function(
                bt.df[hour_mask], 
                **signal_args
            )
            
            # Atribuir sinais ao dataframe principal
            bt.df.loc[hour_mask, 'position'] = signals
        
        # Continuar com o processo normal de backtest
        bt.add_indices()
        bt.calculate_stops()
        bt.calculate_results()
        metrics = bt.calculate_metrics()
        
        # Salvar gráfico da curva de equity
        plt.figure(figsize=(12, 6))
        plt.plot(bt.df['equity'])
        plt.title(f"Curva de Equity - Estratégia Combinada")
        plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "equity_combined.png"))
        plt.close()
        
        # Salvar outros gráficos úteis
        bt.plot_equity_curve(figsize=(14, 10), include_drawdown=True)
        plt.savefig(os.path.join(self.run_dir, "equity_with_dd.png"))
        plt.close()
        
        bt.plot_by_position(figsize=(14, 6))
        plt.savefig(os.path.join(self.run_dir, "equity_by_position.png"))
        plt.close()
        
        bt.plot_profit_by_hour(figsize=(14, 8))
        plt.savefig(os.path.join(self.run_dir, "profit_by_hour.png"))
        plt.close()
        
        # Salvar métricas da estratégia combinada
        with open(os.path.join(self.run_dir, "combined_metrics.json"), 'w') as f:
            # Converter valores numpy para Python nativos
            clean_metrics = {k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}
            json.dump(clean_metrics, f, indent=4)
        
        # Imprimir métricas principais
        print("\n===== RESULTADOS DA ESTRATÉGIA COMBINADA =====")
        print(f"Retorno Total: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
        print(f"Retorno Anualizado: {metrics['annual_return']:.2f}%")
        print(f"Drawdown Máximo: {metrics['max_drawdown']:.2%}")
        print(f"Trades Totais: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        
        return bt.df, metrics
    
    def create_performance_summary(self):
        """
        Cria um resumo visual do desempenho por hora.
        
        Returns:
            DataFrame: DataFrame com o resumo de desempenho
        """
        if not self.all_results:
            raise ValueError("Nenhum resultado de otimização encontrado. Execute optimize_all_hours primeiro.")
            
        # Extrair dados para o dataframe
        data = []
        param_names = set()
        
        # Primeiro, identifique todos os nomes de parâmetros
        for res in self.all_results:
            param_names.update(res['best_params'].keys())
        
        # Adicione colunas para cada resultado
        for res in self.all_results:
            row = {
                'hour': res['hour'],
                'sortino': res['best_value'],
                'profit_factor': res['metrics']['profit_factor'],
                'win_rate': res['metrics']['win_rate'],
                'max_drawdown': res['metrics']['max_drawdown'],
                'total_return': res['metrics']['total_return'],
                'trades': res['metrics']['trades']
            }
            
            # Adicione todos os parâmetros (com valores default se não existirem)
            for param in param_names:
                row[param] = res['best_params'].get(param, None)
                
            data.append(row)
        
        # Criar dataframe
        df = pd.DataFrame(data)
        df = df.sort_values('hour')
        
        # Salvar como CSV
        df.to_csv(os.path.join(self.run_dir, "performance_by_hour.csv"), index=False)
        
        # Criar visualizações de métricas
        plt.figure(figsize=(14, 10))
        
        # Gráfico de barras do Sortino Ratio por hora
        plt.subplot(2, 2, 1)
        plt.bar(df['hour'], df['sortino'], color='blue', alpha=0.7)
        plt.title('Sortino Ratio por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Sortino Ratio')
        plt.grid(True, alpha=0.3)
        plt.xticks(df['hour'])
        
        # Gráfico de Profit Factor por hora
        plt.subplot(2, 2, 2)
        plt.bar(df['hour'], df['profit_factor'], color='green', alpha=0.7)
        plt.title('Profit Factor por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Profit Factor')
        plt.grid(True, alpha=0.3)
        plt.xticks(df['hour'])
        
        # Gráfico de Win Rate por hora
        plt.subplot(2, 2, 3)
        plt.bar(df['hour'], df['win_rate'], color='purple', alpha=0.7)
        plt.title('Win Rate por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Win Rate')
        plt.grid(True, alpha=0.3)
        plt.xticks(df['hour'])
        
        # Gráfico de número de trades por hora
        plt.subplot(2, 2, 4)
        plt.bar(df['hour'], df['trades'], color='orange', alpha=0.7)
        plt.title('Número de Trades por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Trades')
        plt.grid(True, alpha=0.3)
        plt.xticks(df['hour'])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, "performance_summary.png"))
        plt.close()
        
        # Criar visualizações de parâmetros (até 4 parâmetros)
        param_cols = [col for col in df.columns if col not in ['hour', 'sortino', 'profit_factor', 'win_rate', 'max_drawdown', 'total_return', 'trades']]
        
        if param_cols:
            plt.figure(figsize=(14, 10))
            
            for i, param in enumerate(param_cols[:4]):  # Limitar a 4 parâmetros
                if i < 4:  # Apenas 4 subplots
                    plt.subplot(2, 2, i+1)
                    plt.bar(df['hour'], df[param], color=plt.cm.tab10(i), alpha=0.7)
                    plt.title(f'{param} Ótimo por Hora')
                    plt.xlabel('Hora')
                    plt.ylabel(param)
                    plt.grid(True, alpha=0.3)
                    plt.xticks(df['hour'])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, "params_summary.png"))
            plt.close()
        
        return df
    
    def run(self, signal_function, param_ranges, fixed_params=None, hours_to_optimize=None, min_threshold=0.5):
        """
        Executa o pipeline completo de otimização de estratégias.
        
        Args:
            signal_function (callable): Função de sinal para definir posições
            param_ranges (dict): Dicionário com os ranges de parâmetros a otimizar
            fixed_params (dict, optional): Parâmetros fixos (não otimizados)
            hours_to_optimize (list): Lista de horas para otimizar (default: todas)
            min_threshold (float): Threshold mínimo para o sortino ratio
            
        Returns:
            str: Caminho para o diretório com os resultados
        """
        self.strategy_function = signal_function
        self.set_param_ranges(param_ranges)
        
        if fixed_params:
            self.set_fixed_params(fixed_params)
        
        print(f"{'='*80}")
        print(f"INICIANDO PIPELINE DE OTIMIZAÇÃO DE ESTRATÉGIAS FOREX")
        print(f"{'='*80}")
        print(f"Símbolo: {self.symbol}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Período: {self.data_ini} a {self.data_fim}")
        print(f"Estratégia base: {self.strategy_function.__name__}")
        print(f"Parâmetros a otimizar: {self.param_ranges}")
        print(f"Parâmetros fixos: {self.fixed_params}")
        print(f"{'='*80}\n")

        # Configurar diretórios
        self.setup_dirs()
        print(f"Resultados serão salvos em: {self.run_dir}")
        
        # Definir horas para otimização
        if hours_to_optimize is None:
            hours_to_optimize = list(range(24))  # 0-23
        
        # Etapa 1: Otimizar cada hora
        print("\nEtapa 1: Otimizando estratégias por hora...")
        self.optimize_all_hours(hours_to_optimize)
        
        # Etapa 2: Criar resumo de desempenho
        print("\nEtapa 2: Criando resumo de desempenho...")
        self.create_performance_summary()
        
        # Etapa 3: Validar os melhores parâmetros
        print("\nEtapa 3: Validando os melhores parâmetros...")
        self.validate_best_params()
        
        # Etapa 4: Criar estratégia combinada
        print("\nEtapa 4: Criando estratégia combinada...")
        self.create_combined_strategy(min_threshold)
        
        # Etapa 5: Testar estratégia combinada
        if self.combined_strategy:
            print("\nEtapa 5: Testando estratégia combinada...")
            df, metrics = self.test_combined_strategy()
            
        print(f"\nPipeline concluído! Resultados salvos em: {self.run_dir}")
        return self.run_dir