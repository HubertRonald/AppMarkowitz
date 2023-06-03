#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit Test"""

from pandera.typing import DataFrame
from typing import Tuple, List
import subprocess
import numpy as np
import pandas as pd
from ..modules.get_data import get_credentials
from ..modules.markowitz import rendimiento_logaritmo
from ..utils.constants import Constants

CTE: Constants = Constants()


def get_aplaca_markets_keys()->Tuple[str, str]:
    """Acceder a Credenciales para el API de Alpaca Markets a traves de bash"""
    source: List[str] = ['cat', '../.env']
    process = subprocess.Popen(source, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    output: List[str] = output.decode('utf-8')\
        .replace(CTE.ALPACA_MARKETS_API_KEY, CTE.EMPTY_STRING)\
        .replace(CTE.ALPACA_MARKETS_SECRET_KEY, CTE.EMPTY_STRING)\
        .replace('=', CTE.EMPTY_STRING)\
        .split('\n')
    
    return tuple(output)


def test_constants():
    """"""
    assert CTE.SEED == 4042023
    assert CTE.EMPTY_STRING == ''
    assert CTE.ALPACA_MARKETS_API_KEY == 'ALPACA_MARKETS_API_KEY'
    assert CTE.ALPACA_MARKETS_SECRET_KEY == 'ALPACA_MARKETS_SECRET_KEY'


def test_get_credentials():
    """Verificar si las api keys de Alpaca Markets existen"""
    
    assert get_credentials() == get_aplaca_markets_keys()[:2]


def test_rendimiento_logaritmo():
    """"""
    np.random.seed(CTE.SEED)
    fake_stock: DataFrame = pd.DataFrame({
        'fake_close_price': np.random.uniform(100,300,3)
    })

    assert rendimiento_logaritmo(fake_stock).equals(np.log(1+fake_stock.pct_change()[1:])) == True
