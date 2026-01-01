from pydantic import BaseModel
from typing import Optional, List
from datetime import date

class StockRequest(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    interval: Optional[str] = "1d"

class MLRequest(StockRequest):
    model: str  # LSTM, Tree Classifier, Prophet

class TARequest(StockRequest):
    indicator: str  # EMA, RSI, MACD, Bollinger Band
