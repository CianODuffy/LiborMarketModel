from scipy.optimize import minimize_scalar
import numpy as np

# Inverts black swaption formula to determine volatility from price
class BlackScholesSolver():
    def __init__(self, volatility):
        self.volatility = volatility

    def set_parameters(self, start, swap_length, price):
        self.start = start
        self.swap_length = swap_length
        self.price = price
        self.strike  = self.volatility.bootstrapping.get_forward_swap_rates(self.start,
                                                                            self.swap_length)

    def objective_function_payer(self, implied_volatility):
        price = self.volatility.get_swaption_price_t0_payer(self.start,
                                                            self.swap_length, self.strike, implied_volatility)
        return np.power(price - self.price, 2)

    def solve_and_get_implied_volatility_payer(self):
        result = minimize_scalar(self.objective_function_payer)
        return result.x

    def objective_function_receiver(self, implied_volatility):
        price = self.volatility.get_swaption_price_t0_receiver(self.start,
                                                            self.swap_length, self.strike, implied_volatility)
        return np.power(price - self.price, 2)

    def solve_and_get_implied_volatility_receiver(self):
        result = minimize_scalar(self.objective_function_receiver)
        return result.x