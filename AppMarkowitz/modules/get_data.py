#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Get Data From Alpaca Market"""

import os
from typing import List, Tuple, Union
from datetime import datetime as dt
from pandera.typing import DataFrame
from dotenv import load_dotenv

from alpaca.common.types import RawData
from alpaca.data.models import BarSet

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import pandas as pd
from ..utils.constants import Constants

CTE: Constants = Constants()


def get_credentials()->Tuple[str, str]:
    """Conseguir credenciales de Alpaca Markets"""
    load_dotenv()
    return (
        os.getenv(CTE.ALPACA_MARKETS_API_KEY),
        os.getenv(CTE.ALPACA_MARKETS_SECRET_KEY)
    )


def get_bars(
        assets: List[str],
        end: Union[str, dt.timestamp]='2023-04-04 00:00:00',
        start: Union[str, dt.timestamp]='2022-04-04 00:00:00'
    )->Union[BarSet, RawData]:
    """Obtener de Alpaca Markets los precios diarios de n activos para un periodo dado"""
    # Alpaca Markets' Crendtials
    alpaca_markets_api_key, alpaca_markets_secret_key = get_credentials()

    # Alpaca Markets' Client
    stock_client: StockHistoricalDataClient = StockHistoricalDataClient(
        api_key=alpaca_markets_api_key,
        secret_key=alpaca_markets_secret_key
    )

    # Alpaca Markets' Request
    request_params: StockBarsRequest = StockBarsRequest(
        symbol_or_symbols=assets,
        timeframe=TimeFrame.Day,
        end=end,
        start=start
    )

    # Alpaca Markets' Response
    bars: Union[BarSet, RawData] = stock_client.get_stock_bars(
        request_params=request_params
    )

    return bars


def get_closing_price(
        assets: List[str],
        end: Union[str, dt.timestamp]='2023-04-04 00:00:00',
        start: Union[str, dt.timestamp]='2022-04-04 00:00:00',
        path_save: str=CTE.DATA_STOCK_PRICE
    )->DataFrame:
    """Obtener de Alpaca Markets, el precio de cierre de un conjunto de activos para un periodo dado"""
    # Alpaca Markets' Response
    bars = get_bars(assets=assets, end=end, start=start)
    
    # Tiny ETL
    df: DataFrame = bars.df.close.copy()\
        .reset_index() \
        .pivot_table(index='timestamp', columns='symbol', values='close')
    
    df.index = df.index.map(pd.Timestamp.date)
    df.columns = df.columns.to_list()
    df.index.names = ['date']

    # Save
    if check_path_file():
        df.to_csv(path_save)

    return df


def check_path_file(path_save: str=CTE.DATA_STOCK_PRICE)->bool:
    """Asegurarse de no sobreescribir el actual archivo"""
    override: bool = False
    check: bool = os.path.exists(path_save)
    if check:
        res: str = input(
            f'\nDesea sobreescribir el archivo "{path_save}" si/no: '
        ).lower()
        override = res in ['s','si','sí','y','yes']

    return override


if __name__ == '__main__':
    print('ALPACA_MARKETS_API_KEY:', get_credentials()[0])
    print('Save Data In:', CTE.DATA_STOCK_PRICE)
    print(check_path_file())
