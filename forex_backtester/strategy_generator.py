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
#import json
import simplejson as json
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from .backtester import Backtester


# Adicione estas definições no nível global do arquivo
class NumpyEncoder(json.JSONEncoder):
    """
    Codificador JSON personalizado que lida com tipos NumPy.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
    

def save_json(data, filepath):
    """
    Função auxiliar para salvar dados como JSON com suporte a tipos NumPy.
    
    Args:
        data: Dados a serem salvos
        filepath: Caminho do arquivo onde os dados serão salvos
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=4)

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
                 valor_lote=100000,
                 optimize_metric='sortino_ratio',
                 direction='maximize'):
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
            optimize_metric (str): Métrica a ser otimizada. Opções: 'sortino_ratio', 'sharpe_ratio', 
                                   'calmar_ratio', 'profit_factor', 'total_return', 'win_rate', etc.
            direction (str): Direção da otimização: 'maximize' ou 'minimize'
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
        self.optimize_metric = optimize_metric
        self.direction = direction
        
        # Validar a métrica e direção de otimização
        self._validate_optimization_params()
        
        # Estado interno
        self.run_dir = None
        self.all_results = []
        self.validation_results = []
        self.combined_strategy = None
        self.param_ranges = {}
        self.strategy_function = None
        self.fixed_params = {}
    
    def _validate_optimization_params(self):
        """
        Valida os parâmetros de otimização.
        
        Raises:
            ValueError: Se a métrica de otimização não for válida.
            ValueError: Se a direção de otimização não for válida.
        """
        valid_metrics = [
            'sortino_ratio', 'sharpe_ratio', 'calmar_ratio', 'profit_factor',
            'total_return', 'total_return_pct', 'annual_return', 'win_rate',
            'expectancy', 'win_loss_ratio', 'max_drawdown'
        ]
        
        valid_directions = ['maximize', 'minimize']
        
        if self.optimize_metric not in valid_metrics:
            raise ValueError(f"Métrica de otimização inválida: {self.optimize_metric}. "
                            f"Opções válidas: {', '.join(valid_metrics)}")
        
        if self.direction not in valid_directions:
            raise ValueError(f"Direção de otimização inválida: {self.direction}. "
                            f"Opções válidas: {', '.join(valid_directions)}")
        
        # Para algumas métricas, a direção de otimização é óbvia e deve ser respeitada
        inverse_metrics = ['max_drawdown']  # Métricas que geralmente são minimizadas
        
        if self.optimize_metric in inverse_metrics and self.direction == 'maximize':
            print(f"AVISO: A métrica '{self.optimize_metric}' geralmente é minimizada, "
                  f"mas foi selecionada para maximização.")
            
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
        run_dir = os.path.join(self.export_dir, f"run_{self.symbol}_{timestamp}")
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
            'strategy': self.strategy_function.__name__ if self.strategy_function else None,
            'optimize_metric': self.optimize_metric,
            'direction': self.direction
        }
        
        save_json(config, os.path.join(self.run_dir, "config.json"))
            
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
        try:
            # Gerar parâmetros a partir dos ranges definidos
            strategy_params = {}
            
            # Inicializar com valores padrão
            tp = 0.15  # Valor padrão adequado para sua escala
            sl = 0.15  # Valor padrão adequado para sua escala
            
            for param_name, param_range in self.param_ranges.items():
                # Processar tp e sl preservando valores decimais
                if param_name == 'tp':
                    if isinstance(param_range, list):
                        tp = trial.suggest_categorical(param_name, param_range)
                    elif len(param_range) == 2:
                        min_val, max_val = param_range
                        tp = trial.suggest_float(param_name, min_val, max_val)
                    elif len(param_range) == 3:
                        min_val, max_val, step = param_range
                        tp = trial.suggest_float(param_name, min_val, max_val, step=step)
                elif param_name == 'sl':
                    if isinstance(param_range, list):
                        sl = trial.suggest_categorical(param_name, param_range)
                    elif len(param_range) == 2:
                        min_val, max_val = param_range
                        sl = trial.suggest_float(param_name, min_val, max_val)
                    elif len(param_range) == 3:
                        min_val, max_val, step = param_range
                        sl = trial.suggest_float(param_name, min_val, max_val, step=step)
                else:
                    # Parâmetros para a função de estratégia
                    if isinstance(param_range, list):
                        strategy_params[param_name] = trial.suggest_categorical(param_name, param_range)
                    elif len(param_range) == 2:
                        min_val, max_val = param_range
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            strategy_params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                        else:
                            strategy_params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                    elif len(param_range) == 3:
                        min_val, max_val, step = param_range
                        strategy_params[param_name] = trial.suggest_float(param_name, min_val, max_val, step=step)
            
            # Adicionar parâmetros fixos à estratégia
            for k, v in self.fixed_params.items():
                if k not in ['tp', 'sl']:
                    strategy_params[k] = v
            
            # Print para debug - use {:.5f} para formatar os valores com 5 casas decimais
            print(f"Trial {trial.number} para hora {hour}: Testando estratégia com parâmetros {strategy_params}")
            print(f"Trial {trial.number} para hora {hour}: Usando backtester com TP={tp:.5f}, SL={sl:.5f}")            
        
            # Adicionar allowed_hours à estratégia
            strategy_params['allowed_hours'] = [hour]
            
            # Configurar o backtester com valores decimais
            bt = Backtester(
                symbol=self.symbol,
                timeframe=self.timeframe,
                data_ini=self.data_ini,
                data_fim=self.data_fim,
                tp=tp,  # Valor decimal
                sl=sl,  # Valor decimal
                slippage=self.slippage,
                tc=self.tc,
                lote=self.lote,
                valor_lote=self.valor_lote,
                initial_cash=self.initial_cash,
                path_base=self.path_base,
                daytrade=self.daytrade
            )
            
            # Executar backtest focando apenas na hora específica
            print(f"Executando backtest para trial {trial.number}, hora {hour}...")
            _, metrics = bt.run(
                signal_function=self.strategy_function,
                signal_args=strategy_params
            )
            
            # Obter o valor da métrica selecionada
            metric_value = metrics.get(self.optimize_metric, 0)
            
            # Ajuste para métricas que normalmente são minimizadas, quando a direção é 'minimize'
            if self.direction == 'minimize':
                if self.optimize_metric == 'max_drawdown':
                    # max_drawdown já é positivo, então não precisa inverter
                    pass
                else:
                    # Para outras métricas que queremos minimizar, invertemos o sinal
                    metric_value = -metric_value
            
            print(f"Trial {trial.number} concluído: {self.optimize_metric} = {metric_value:.4f}")
            
            # Adicionar métricas adicionais para o Optuna armazenar
            trial.set_user_attr('sortino_ratio', metrics['sortino_ratio'])
            trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
            trial.set_user_attr('calmar_ratio', metrics['calmar_ratio'])
            trial.set_user_attr('profit_factor', metrics['profit_factor'])
            trial.set_user_attr('win_rate', metrics['win_rate'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('total_return', metrics['total_return'])
            trial.set_user_attr('trades', metrics['total_trades'])
            
            # Adicionar tp e sl aos resultados
            all_params = strategy_params.copy()
            all_params['tp'] = tp
            all_params['sl'] = sl
            
            # Store all parameters for retrieval in optimize_hour
            for param_name, param_value in all_params.items():
                trial.set_user_attr(f"param_{param_name}", param_value)
            
            return metric_value
        
        except Exception as e:
            import traceback
            print(f"Erro no trial {trial.number} para hora {hour}:")
            print(traceback.format_exc())
            return float('-inf') if self.direction == 'maximize' else float('inf')


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
        print(f"Métrica: {self.optimize_metric} - Direção: {self.direction}")
        print(f"{'='*50}")
        
        try:
            # Criar função objetivo específica para a hora
            objective = partial(self._objective_hour, hour=hour)
            
            # Configurar e executar o estudo Optuna
            study_name = f"hora_{hour:02d}"
            study = optuna.create_study(
                direction=self.direction,
                study_name=study_name
            )
            
            print(f"Iniciando otimização com {self.num_trials} trials para hora {hour}...")
            study.optimize(objective, n_trials=self.num_trials)
            print(f"Otimização concluída para hora {hour}.")
            
            # Extrair o melhor valor
            best_value = study.best_value
            
            # Ajustar o valor da métrica para exibição se a direção for minimizar
            display_value = best_value
            if self.direction == 'minimize' and self.optimize_metric != 'max_drawdown':
                display_value = -best_value
            
            # Extrair os melhores parâmetros dos user_attrs
            best_params = {}
            for key, value in study.best_trial.user_attrs.items():
                if key.startswith("param_"):
                    param_name = key[6:]  # Remove "param_" prefix
                    best_params[param_name] = value
            
            # Extrair métricas
            best_metrics = {
                'sortino_ratio': study.best_trial.user_attrs.get('sortino_ratio', 0),
                'sharpe_ratio': study.best_trial.user_attrs.get('sharpe_ratio', 0),
                'calmar_ratio': study.best_trial.user_attrs.get('calmar_ratio', 0),
                'profit_factor': study.best_trial.user_attrs.get('profit_factor', 0),
                'win_rate': study.best_trial.user_attrs.get('win_rate', 0),
                'max_drawdown': study.best_trial.user_attrs.get('max_drawdown', 0),
                'total_return': study.best_trial.user_attrs.get('total_return', 0),
                'trades': study.best_trial.user_attrs.get('trades', 0)
            }
            
            # Salvar resultados
            results = {
                'hour': hour,
                'best_params': best_params,
                'best_value': display_value,  # Usar o valor para exibição
                'raw_best_value': best_value,  # Salvar o valor bruto da otimização
                'optimize_metric': self.optimize_metric,
                'direction': self.direction,
                'metrics': best_metrics
            }

           
            # Salvar em arquivo JSON
            if self.run_dir:
                save_json(results, os.path.join(self.run_dir, f"results_hour_{hour:02d}.json"))
            
            print(f"\nMelhores parâmetros para hora {hour:02d}:")
            print(f"{self.optimize_metric}: {display_value:.4f}")
            print(f"Parâmetros: {best_params}")
            print(f"Profit Factor: {best_metrics['profit_factor']:.4f}")
            print(f"Win Rate: {best_metrics['win_rate']:.2%}")
            print(f"Max Drawdown: {best_metrics['max_drawdown']:.2%}")
            print(f"Total Return: ${best_metrics['total_return']:.2f}")
            print(f"Total Trades: {best_metrics['trades']}")
            
            return results
        
        except Exception as e:
            import traceback
            print(f"Erro detalhado na otimização para hora {hour}:")
            print(traceback.format_exc())
            return None
    
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
            
        # Otimizar cada hora usando execução serial
        self.all_results = []
        
        print(f"Iniciando otimização serial para {len(hours_to_optimize)} horas...")
        for hour in hours_to_optimize:
            try:
                result = self.optimize_hour(hour)
                if result:  # Only add non-None results
                    self.all_results.append(result)
            except Exception as e:
                import traceback
                print(f"Erro na otimização da hora {hour}:")
                print(traceback.format_exc())
        
        print(f"Otimização concluída para {len(self.all_results)} de {len(hours_to_optimize)} horas.")
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
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'calmar_ratio': metrics['calmar_ratio'],
                    'profit_factor': metrics['profit_factor'],
                    'win_rate': metrics['win_rate'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_return': metrics['total_return'],
                    'total_trades': metrics['total_trades'],
                    'annual_return': metrics['annual_return']
                }
            })
        
        # Salvar resultados da validação
        save_json(self.validation_results, os.path.join(self.run_dir, "validation_results.json"))
        
        return self.validation_results
    
    def create_combined_strategy(self, min_threshold=0.5):
        """
        Cria uma estratégia combinada usando apenas as horas com bons resultados.
        
        Args:
            min_threshold (float): Threshold mínimo para a métrica de otimização
            
        Returns:
            dict: Configuração da estratégia combinada
        """
        if not self.validation_results:
            raise ValueError("Nenhum resultado de validação encontrado. Execute validate_best_params primeiro.")
            
        # Filtrar apenas horas com bom desempenho
        good_hours = []
        good_params = {}
        
        for res in self.validation_results:
            # Verificar o valor da métrica escolhida
            metric_value = res['metrics'].get(self.optimize_metric, 0)
            
            # Para métricas que queremos minimizar (como drawdown) a condição é invertida
            threshold_condition = True
            if self.direction == 'maximize':
                threshold_condition = metric_value > min_threshold
            else:
                # Para minimização, o valor deve ser menor que o threshold
                threshold_condition = metric_value < min_threshold
            
            if threshold_condition:
                hour = res['hour']
                good_hours.append(hour)
                good_params[str(hour)] = res['params']  # Usar str como chave para serialização JSON
        
        # Verificar se temos horas suficientes
        if not good_hours:
            print(f"Nenhuma hora passou pelo threshold de {min_threshold} para a métrica {self.optimize_metric}!")
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
            'valor_lote': self.valor_lote,
            'optimize_metric': self.optimize_metric,
            'direction': self.direction
        }
        
        # Salvar configuração da estratégia combinada
        save_json(self.combined_strategy, os.path.join(self.run_dir, "combined_strategy.json"))
        
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
        save_json(metrics, os.path.join(self.run_dir, "combined_metrics.json"))
        
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
        print(f"Métrica otimizada ({self.optimize_metric}): {metrics[self.optimize_metric]:.4f}")
        
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
            n_params = len(param_cols)
            
            # Calcular layout automático
            if n_params <= 2:
                rows, cols = 1, n_params
            elif n_params <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 2  # Para 5-6 parâmetros
                if n_params > 6:
                    print(f"Atenção: Apenas os primeiros 6 parâmetros serão visualizados de um total de {n_params}.")
                    param_cols = param_cols[:6]
                    
            plt.figure(figsize=(14, 5 * rows))
            
            for i, param in enumerate(param_cols):
                plt.subplot(rows, cols, i+1)
                plt.bar(df['hour'], df[param], color=plt.cm.tab10(i % 10), alpha=0.7)
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