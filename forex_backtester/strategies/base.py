
"""
Classes base para estratégias de trading.
"""

from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Classe base abstrata para todas as estratégias de trading.
    """
    
    @abstractmethod
    def generate_signals(self, df, **kwargs):
        """
        Gera sinais de trading.
        
        Args:
            df (pandas.DataFrame): DataFrame com os dados de mercado.
            **kwargs: Argumentos adicionais específicos da estratégia.
            
        Returns:
            pandas.Series: Série com os sinais de trading (-1=short, 0=neutro, 1=long).
        """
        pass
    
    @abstractmethod
    def get_parameters(self):
        """
        Retorna os parâmetros da estratégia.
        
        Returns:
            dict: Dicionário com os parâmetros da estratégia.
        """
        pass
    
    def __str__(self):
        """Representação em string da estratégia."""
        params = self.get_parameters()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str})"