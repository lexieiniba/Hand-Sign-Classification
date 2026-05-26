import json
import numpy as np
from trading_floor import TradingFloor
from market_layer import MarketLayer as Market
from tick_generator import generate_tick

def get_log_returns(candles):
    """Converts a list of candle dicts into an array of log returns."""
    closes = np.array([c["close"] for c in candles])
    # Use np.diff(np.log(x)) to get percentage changes
    return np.diff(np.log(closes)).tolist()

def generate_training_data(samples=5000, seq_len=30, future_len=5):
    print(f"Generating {samples} samples...")

    symbols = ["SIM"]
    market = Market(symbols, use_real_data=False)
    tf = TradingFloor(market, symbols, use_real_data=False)

    data = []
    # Warm up
    for _ in range(200):
        generate_tick(tf, "SIM", training=True)

    for i in range(samples):
        generate_tick(tf, "SIM", training=True)
        
        # We need seq_len + future_len worth of data
        if len(tf.buffers["SIM"]) < seq_len + future_len:
            continue

        # Get the full slice
        full_slice = tf.buffers["SIM"][-(seq_len + future_len):]
        
        # 1. Feature: Log returns of the sequence (first 30 candles)
        seq_candles = full_slice[:seq_len]
        features = get_log_returns(seq_candles)
        
        # 2. Target: Direction of the next 5 candles relative to current
        current_price = seq_candles[-1]["close"]
        future_price = full_slice[-1]["close"]
        target = 1 if future_price > current_price else 0

        data.append({
            "seq": features,
            "target": target
        })

        if i % 1000 == 0:
            print(f"{i}/{samples}...")

    with open("training_data.json", "w") as f:
        json.dump(data, f)

    print(f"Saved {len(data)} samples to training_data.json")

if __name__ == "__main__":
    generate_training_data()