import yfinance as yf
from datetime import datetime, timezone

class MarketLayer:
    def __init__(self, symbols, use_real_data=True):
        self.symbols = symbols
        self.use_real_data = use_real_data
        self.data = {sym: [] for sym in symbols}

        if use_real_data:
            self._fetch_initial_data()
        else:
            # synthetic mode: start with empty lists
            for sym in self.symbols:
                self.data[sym] = []

    def _fetch_initial_data(self):
        for sym in self.symbols:
            try:
                df = yf.download(sym, period="1d", interval="1m")
                candles = []
                for ts, row in df.iterrows():
                    candles.append({
                        "ts": ts.to_pydatetime().replace(tzinfo=timezone.utc),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                    })
                self.data[sym] = candles
            except Exception:
                self.data[sym] = []

    def get_history(self, symbol, limit=None):
        arr = self.data.get(symbol, [])
        if limit is None:
            return list(arr)
        return arr[-limit:]

    def get_latest(self, symbol):
        arr = self.data.get(symbol, [])
        return arr[-1] if arr else None
