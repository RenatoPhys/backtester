"""
Módulo contendo funções utilitárias para cálculo de métricas de performance de estratégias de trading.
"""

import numpy as np
import pandas as pd


def calculate_basic_metrics(equity_series, initial_cash):
    """
    Calcula métricas básicas de retorno e equity.
    
    Args:
        equity_series (pd.Series): Série temporal da equity
        initial_cash (float): Valor inicial do capital
        
    Returns:
        dict: Dicionário com métricas básicas
    """
    final_equity = equity_series.iloc[-1]
    total_return = final_equity - initial_cash
    total_return_pct = (final_equity / initial_cash - 1) * 100
    
    return {
        'initial_cash': initial_cash,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return_pct
    }


def calculate_drawdown_metrics(equity_series):
    """
    Calcula métricas relacionadas ao drawdown.
    
    Args:
        equity_series (pd.Series): Série temporal da equity
        
    Returns:
        dict: Dicionário com métricas de drawdown
    """
    from .drawdown_utils import calc_dd
    
    dd_df = calc_dd(equity_series)
    
    max_drawdown = abs(dd_df['drawdown_pct'].min()) if len(dd_df) > 0 else 0
    max_drawdown_value = abs(dd_df['drawdown'].min()) if len(dd_df) > 0 else 0
    max_time_underwater = dd_df['time_uw'].max() if len(dd_df) > 0 else 0
    
    total_periods = len(dd_df)
    underwater_periods = dd_df['underwater'].sum()
    underwater_rate = underwater_periods / total_periods if total_periods > 0 else 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_value': max_drawdown_value,
        'max_time_underwater': max_time_underwater,
        'underwater_rate': underwater_rate
    }


def calculate_trade_metrics(df, strategy_col='strategy', position_col='position', status_col='status_trade'):
    """
    Calcula métricas relacionadas aos trades executados.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados do backtest
        strategy_col (str): Nome da coluna com resultados da estratégia
        position_col (str): Nome da coluna com as posições
        status_col (str): Nome da coluna com status dos trades
        
    Returns:
        dict: Dicionário com métricas de trades
    """
    # Identificar início de trades
    df_copy = df.copy()
    df_copy['trade_start'] = df_copy[position_col].diff().ne(0) & (df_copy[position_col] != 0)
    total_trades = df_copy['trade_start'].sum()
    
    if total_trades == 0:
        return {
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'win_rate': 0,
            'tp_rate': 0,
            'sl_rate': 0,
            'time_exit_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_loss_ratio': 0,
            'expectancy': 0
        }
    
    # Analisar trades vencedores e perdedores
    trade_starts = df_copy[df_copy['trade_start']]
    win_trades = (trade_starts[strategy_col] > 0).sum()
    loss_trades = (trade_starts[strategy_col] < 0).sum()
    win_rate = win_trades / total_trades
    
    # Analisar tipos de saída
    tp_hits = (trade_starts[status_col] == 1).sum()
    sl_hits = (trade_starts[status_col] == -1).sum()
    time_exits = (trade_starts[status_col] == 0).sum()
    
    tp_rate = tp_hits / total_trades
    sl_rate = sl_hits / total_trades
    time_exit_rate = time_exits / total_trades
    
    # Profit factor
    gross_profits = df_copy.loc[df_copy[strategy_col] > 0, strategy_col].sum()
    gross_losses = abs(df_copy.loc[df_copy[strategy_col] < 0, strategy_col].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # Médias de ganhos e perdas
    avg_win = gross_profits / win_trades if win_trades > 0 else 0
    avg_loss = gross_losses / loss_trades if loss_trades > 0 else 0
    win_loss_ratio = avg_win / abs(avg_loss) if loss_trades > 0 else float('inf')
    
    # Expectância
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if (win_trades > 0 and loss_trades > 0) else 0
    
    return {
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': win_rate,
        'tp_rate': tp_rate,
        'sl_rate': sl_rate,
        'time_exit_rate': time_exit_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'expectancy': expectancy
    }


def calculate_risk_metrics(equity_series, initial_cash, risk_free_rate=0.0):
    """
    Calcula métricas de risco ajustado (Sharpe, Sortino, Calmar).
    
    Args:
        equity_series (pd.Series): Série temporal da equity
        initial_cash (float): Valor inicial do capital
        risk_free_rate (float): Taxa livre de risco anual em decimal
        
    Returns:
        dict: Dicionário com métricas de risco
    """
    # Calcular retornos diários
    equity_returns = equity_series.pct_change().fillna(0)
    
    # Agregar retornos por dia (composição correta para múltiplas operações no mesmo dia)
    daily_returns = equity_returns.resample('D').apply(
        lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
    )
    
    # Filtrar apenas dias com dados reais
    daily_returns_filtered = daily_returns[daily_returns != 0]
    
    # Calcular métricas anualizadas
    trading_days = len(daily_returns_filtered)
    years = trading_days / 252  # 252 dias úteis por ano
    
    if years > 0 and len(daily_returns_filtered) > 0:
        # Retorno anualizado
        final_equity = equity_series.iloc[-1]
        total_return_decimal = (final_equity / initial_cash) - 1
        annual_return = (1 + total_return_decimal) ** (1/years) - 1
        
        # Volatilidade anualizada
        annual_volatility = daily_returns_filtered.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Sortino Ratio
        downside_returns = daily_returns_filtered[daily_returns_filtered < 0]
        if len(downside_returns) > 0:
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar Ratio
        from .drawdown_utils import calc_dd
        dd_df = calc_dd(equity_series)
        max_drawdown = abs(dd_df['drawdown_pct'].min()) if len(dd_df) > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0
        
        # Converter para percentual para compatibilidade
        annual_return_pct = annual_return * 100
        annual_volatility_pct = annual_volatility * 100
        
    else:
        annual_return = annual_return_pct = 0
        annual_volatility = annual_volatility_pct = 0
        sharpe_ratio = sortino_ratio = calmar_ratio = 0
    
    return {
        'annual_return': annual_return_pct,
        'annual_volatility': annual_volatility_pct,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'risk_free_rate': risk_free_rate
    }


def calculate_comprehensive_metrics(df, initial_cash, risk_free_rate=0.0, 
                                  equity_col='equity', strategy_col='strategy', 
                                  position_col='position', status_col='status_trade'):
    """
    Calcula todas as métricas de performance de uma vez.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados do backtest
        initial_cash (float): Valor inicial do capital
        risk_free_rate (float): Taxa livre de risco anual em decimal
        equity_col (str): Nome da coluna com a equity
        strategy_col (str): Nome da coluna com resultados da estratégia
        position_col (str): Nome da coluna com as posições
        status_col (str): Nome da coluna com status dos trades
        
    Returns:
        dict: Dicionário completo com todas as métricas
    """
    equity_series = df[equity_col]
    
    # Calcular cada grupo de métricas
    basic_metrics = calculate_basic_metrics(equity_series, initial_cash)
    drawdown_metrics = calculate_drawdown_metrics(equity_series)
    trade_metrics = calculate_trade_metrics(df, strategy_col, position_col, status_col)
    risk_metrics = calculate_risk_metrics(equity_series, initial_cash, risk_free_rate)
    
    # Combinar todos os resultados
    all_metrics = {}
    all_metrics.update(basic_metrics)
    all_metrics.update(drawdown_metrics)
    all_metrics.update(trade_metrics)
    all_metrics.update(risk_metrics)
    
    return all_metrics


def format_metrics_report(metrics, symbol=None, timeframe=None, period=None):
    """
    Formata as métricas em um relatório legível.
    
    Args:
        metrics (dict): Dicionário com as métricas calculadas
        symbol (str, optional): Símbolo negociado
        timeframe (str, optional): Timeframe usado
        period (str, optional): Período analisado
        
    Returns:
        str: Relatório formatado
    """
    report = []
    
    # Cabeçalho
    report.append("=" * 50)
    report.append("RELATÓRIO DE DESEMPENHO")
    report.append("=" * 50)
    
    if symbol or timeframe or period:
        if symbol:
            report.append(f"Símbolo: {symbol}")
        if timeframe:
            report.append(f"Timeframe: {timeframe}")
        if period:
            report.append(f"Período: {period}")
        report.append("")
    
    # Resultados básicos
    report.append("--- RESULTADOS ---")
    report.append(f"Saldo Inicial: ${metrics['initial_cash']:.2f}")
    report.append(f"Saldo Final: ${metrics['final_equity']:.2f}")
    report.append(f"Retorno Total: ${metrics['total_return']:.2f} ({metrics['total_return_pct']:.2f}%)")
    report.append(f"Retorno Anualizado: {metrics['annual_return']:.2f}%")
    report.append(f"Volatilidade Anualizada: {metrics['annual_volatility']:.2f}%")
    report.append("")
    
    # Drawdown
    report.append("--- DRAWDOWN ---")
    report.append(f"Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    report.append(f"Drawdown Máximo (Valor): ${metrics['max_drawdown_value']:.2f}")
    report.append(f"Tempo Máximo em Drawdown: {metrics['max_time_underwater']} períodos")
    report.append(f"Tempo em Drawdown: {metrics['underwater_rate']:.2%} do total")
    report.append("")
    
    # Trades
    report.append("--- TRADES ---")
    report.append(f"Total de Trades: {metrics['total_trades']}")
    report.append(f"Trades Vencedores: {metrics['win_trades']} ({metrics['win_rate']:.2%})")
    report.append(f"Trades Perdedores: {metrics['loss_trades']} ({1-metrics['win_rate']:.2%})")
    report.append(f"Saídas por TP: {metrics['tp_rate']:.2%}")
    report.append(f"Saídas por SL: {metrics['sl_rate']:.2%}")
    report.append(f"Saídas por Tempo: {metrics['time_exit_rate']:.2%}")
    report.append("")
    
    # Ratios
    report.append("--- RATIOS ---")
    report.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    report.append(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    report.append(f"Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    report.append(f"Profit Factor: {metrics['profit_factor']:.3f}")
    report.append("")
    
    # Análise de ganhos/perdas
    report.append("--- ANÁLISE DE GANHOS/PERDAS ---")
    report.append(f"Ganho Médio: ${metrics['avg_win']:.2f}")
    report.append(f"Perda Média: ${metrics['avg_loss']:.2f}")
    report.append(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}")
    report.append(f"Expectancy: ${metrics['expectancy']:.2f}")
    
    report.append("=" * 50)
    
    return "\n".join(report)


# Funções de conveniência para métricas específicas
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calcula o Sharpe Ratio para uma série de retornos."""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std() if returns.std() != 0 else 0


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calcula o Sortino Ratio para uma série de retornos."""
    excess_returns = returns - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    return excess_returns.mean() / downside_std if downside_std != 0 else 0


def calculate_calmar_ratio(annual_return, max_drawdown):
    """Calcula o Calmar Ratio."""
    return annual_return / max_drawdown if max_drawdown != 0 else 0


def calculate_profit_factor(gross_profits, gross_losses):
    """Calcula o Profit Factor."""
    return gross_profits / gross_losses if gross_losses != 0 else float('inf')