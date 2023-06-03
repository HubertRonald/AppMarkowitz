#!/usr/bin/env python3
# -*- Coding: utf-8 -*-

from dataclasses import dataclass

@dataclass(frozen=True)
class Constants():
    SEED: int = 4042023
    EMPTY_STRING: str = ''
    SP500_COMPANIES: str = './data/s_and_p_500_companies.csv'
    DATA_STOCK_PRICE: str = './data/HistoricalStockPrice.csv'
    ALPACA_MARKETS_API_KEY: str = 'ALPACA_MARKETS_API_KEY'
    ALPACA_MARKETS_SECRET_KEY: str = 'ALPACA_MARKETS_SECRET_KEY'
