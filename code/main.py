import pygame
from market_layer import MarketLayer
from trading_floor import TradingFloor

SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]


def main():
    pygame.init()
    market = MarketLayer(SYMBOLS)
    floor = TradingFloor(market, SYMBOLS)
    floor.run()


if __name__ == "__main__":
    main()
