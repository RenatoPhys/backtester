"""
Módulo contendo funções utilitárias para cálculo de stop loss e take profit.
"""

from numba import jit, float64, int64
import numpy as np

@jit(nopython=True)
def compute_stop_loss_take_profit(position, high, low, close, sl, tp, slippage, idx, idx_final):
    n = position.shape[0]
    status_trade = np.zeros(n)
    sl_idx = np.zeros(n, dtype=np.int64)
    tp_idx = np.zeros(n, dtype=np.int64)

    for i in range(n):
        if position[i] == 1:
            for j in range(i+1, n):
                if low[j] - close[i] < - sl:
                    status_trade[i] = -1
                    sl_idx[i] = j
                    break
                if high[j] - close[i] >= tp + slippage:
                    status_trade[i] = 1
                    tp_idx[i] = j 
                    break
                if idx[j] == idx_final[i]:
                    break
        elif position[i] == -1:
            for j in range(i+1, n):
                if high[j] - close[i] > sl:
                    status_trade[i] = -1
                    sl_idx[i] = j
                    break
                if low[j] - close[i] <= -tp - slippage:
                    status_trade[i] = 1
                    tp_idx[i] = j 
                    break
                if idx[j] == idx_final[i]:
                    break

    return status_trade, sl_idx, tp_idx