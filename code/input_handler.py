import pygame
import sys
import numpy as np

def handle_input(floor):
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            mods = pygame.key.get_mods()
            shift = mods & pygame.KMOD_SHIFT

            # Switch symbols
            if pygame.K_1 <= event.key <= pygame.K_5:
                idx = event.key - pygame.K_1
                if idx < len(floor.symbols):
                    floor.current_index = idx
                    floor.current_symbol = floor.symbols[idx]
                    floor.shake_timer = 5

            # BUY
            if event.key == pygame.K_b and shift:
                sym = floor.current_symbol
                price = floor.current_price[sym]
                units = 2
                cost = price * units

                if floor.portfolio < cost:
                    return

                floor.portfolio -= cost

                if sym not in floor.position:
                    floor.position[sym] = {"entry": price, "units": units}
                else:
                    old_entry = floor.position[sym]["entry"]
                    old_units = floor.position[sym]["units"]
                    new_units = old_units + units
                    new_entry = (old_entry * old_units + price * units) / new_units
                    floor.position[sym]["entry"] = new_entry
                    floor.position[sym]["units"] = new_units

                floor.shake_timer = 5

            # SELL
            if event.key == pygame.K_s and shift:
                sym = floor.current_symbol
                if sym in floor.position:
                    entry = floor.position[sym]["entry"]
                    units = floor.position[sym]["units"]
                    price = floor.current_price[sym]
                    value = price * units *1.5
                    floor.portfolio += value
                    del floor.position[sym]
                    floor.shake_timer = 8

            # INSIDER TERMINAL
            if event.key == pygame.K_i and shift:
                if floor.insider_cooldown == 0 and not floor.insider_active:
                    if floor.portfolio >= floor.insider_cost:

                        # 1. Deduct cost
                        floor.portfolio -= floor.insider_cost

                        # 2. Get the data in the format the AI expects
                        # We need the last 30 log returns
                        candles = floor.buffers[floor.current_symbol][-30:]
                        closes = [c["close"] for c in candles]
                        log_returns = np.diff(np.log(closes)).tolist()

                        # 3. Get the prediction (now a probability float)
                        prob = floor.ai.predict(log_returns)
                        
                        # 4. Set the signal string
                        floor.ai_prediction = "UP" if prob > 0.5 else "DOWN"

                        # 5. Activate insider mode
                        floor.insider_active = True
                        floor.insider_timer = 180
                        floor.insider_cooldown = 300

                        print(f"INSIDER: Prediction generated. Signal: {floor.ai_prediction} (Conf: {prob:.4f})")

            # Toggle holdings
            if event.key == pygame.K_h:
                floor.show_holdings = not floor.show_holdings

        # Mouse tab switching
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if my < 40:
                tab_x = 30
                for i, sym in enumerate(floor.symbols):
                    if tab_x <= mx <= tab_x + 100:
                        floor.current_index = i
                        floor.current_symbol = sym
                        floor.shake_timer = 5
                        break
                    tab_x += 120
