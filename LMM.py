import numpy as np
import Volatility as vol
import pandas as pd
import Bootstrapping as boot
import copy as copy
import BlackScholesSolver as bss
from scipy.optimize import least_squares
from math import *
from scipy.optimize import minimize_scalar

# 1. runs the libor market model and generates forward rate simulations
# 2. produces swaption prices and implied volatilities
# 3. fits an adjustment factor to adjust for drift error
class LMM():
    def __init__(self, swaption_vol_matrix_path, swap_curve_path):
        self.number_of_factors = 3
        self.use_factor_reduction = False
        self.number_of_sims  = 10000
        self.max_projection_time = 40
        self.iterations = 0
        self.bootstrapping = boot.Bootstrapping(swap_curve_path)
        self.volatility = vol.Volatility(self.number_of_factors, self.bootstrapping,
                                         self.use_factor_reduction, swaption_vol_matrix_path)
        self.time_increment = self.bootstrapping.term_increment
        self.bs_solver = bss.BlackScholesSolver(self.volatility)
        self.number_of_terms = self.bootstrapping.number_of_terms
        ##[terms,time, sim]
        self.starting_forward_curve = self.bootstrapping.forward_curve

    def get_random_numbers(self):
        if self.use_factor_reduction:
            return np.random.normal(0, 1, (self.number_of_factors, self.number_of_sims))
        return np.random.normal(0, 1, (self.number_of_terms, self.number_of_sims))

    def set_implied_volatilities_from_prices(self):
        self.implied_volatilities_model_payer = copy.deepcopy(self.swaption_prices_calibration_payer)
        self.implied_volatilities_model_receiver = copy.deepcopy(self.swaption_prices_calibration_receiver)

        for row_index, row in self.swaption_prices_calibration_payer.iterrows():
            start = row[self.volatility.term_name]
            values = row.drop([self.volatility.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_length = float(column_index)
                    self.bs_solver.set_parameters(start, swap_length, v)
                    implied_volatility_payer = self.bs_solver.solve_and_get_implied_volatility_payer()
                    self.implied_volatilities_model_payer.at[row_index, column_index] = implied_volatility_payer

        for row_index, row in self.swaption_prices_calibration_receiver.iterrows():
            start = row[self.volatility.term_name]
            values = row.drop([self.volatility.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_length = float(column_index)
                    self.bs_solver.set_parameters(start, swap_length, v)
                    implied_volatility_receiver = self.bs_solver.solve_and_get_implied_volatility_receiver()
                    self.implied_volatilities_model_receiver.at[row_index, column_index] = implied_volatility_receiver

        np.savetxt('implied_volatility_model_payer.csv', self.implied_volatilities_model_payer, delimiter=',')
        np.savetxt('implied_volatility_model_receiver.csv', self.implied_volatilities_model_receiver, delimiter=',')


    def objective_function(self, parameters):
        self.iterations += 1
        self.volatility.set_parameters_swap(parameters)
        self.volatility.instantiate_arrays()
        self.set_swaption_prices_for_atm_calibration()
        sum = np.zeros(15)
        N=0
        for row_index, row in self.volatility.swaption_prices.iterrows():
            values = row.drop([self.volatility.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    difference = v -  self.swaption_prices_calibration_payer.at[row_index, column_index]
                    sum[N] = difference
                    N += 1
        return sum

    def objective_function_line_search(self, factor):
        self.iterations += 1
        self.volatility.mc_adjustment_factor = factor
        self.volatility.instantiate_arrays()
        self.set_swaption_prices_for_atm_calibration()
        self.set_implied_volatilities_from_prices()
        sum = np.zeros(15)
        N = 0
        for row_index, row in self.volatility.vol_matrix.iterrows():
            values = row.drop([self.volatility.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    difference = v - self.implied_volatilities_model_payer.at[row_index, column_index]
                    sum[N] = difference
                    N += 1
        value = np.sum(np.power(sum,2))
        return value

    def fit_adjustment_factor(self):
        result = minimize_scalar(self.objective_function_line_search, bounds=(0.1, 0.999), method='bounded')

    def set_swaption_prices_for_atm_calibration(self):
        self.swaption_prices_calibration_payer = copy.deepcopy(self.volatility.vol_matrix)
        self.swaption_prices_calibration_receiver = copy.deepcopy(self.volatility.vol_matrix)
        self.put_call_difference = copy.deepcopy(self.volatility.vol_matrix)

        for row_index, row in self.volatility.vol_matrix.iterrows():
            number_of_time_steps_to_option_expiry = int(row[self.volatility.term_name] / self.time_increment)
            start = row[self.volatility.term_name]
            values = row.drop([self.volatility.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_length = float(column_index)
                    swap_length_steps = int(swap_length/self.time_increment)
                    beta = swap_length_steps + number_of_time_steps_to_option_expiry
                    numeraire_index = beta
                    self.run_projection(numeraire_index, number_of_time_steps_to_option_expiry)
                    forward_swap_rate = self.get_forward_swap_rate(number_of_time_steps_to_option_expiry, numeraire_index)
                    strike = self.bootstrapping.get_forward_swap_rates(start, swap_length)
                    strike_vector = np.ones(self.number_of_sims) * strike
                    sum = np.zeros(self.number_of_sims)

                    for i in range(number_of_time_steps_to_option_expiry + 1, numeraire_index + 1):
                        sum += self.time_increment * self.DF[i, number_of_time_steps_to_option_expiry, :]

                    payoff_receiver = np.maximum(strike_vector - forward_swap_rate, 0) * sum \
                                      / self.DF[numeraire_index, number_of_time_steps_to_option_expiry, :]

                    payoff_payer = np.maximum(forward_swap_rate - strike_vector, 0)*sum\
                                   /self.DF[numeraire_index, number_of_time_steps_to_option_expiry,:]

                    receiver_swaption = np.mean(payoff_receiver) * self.DF[numeraire_index, 0, 0]
                    payer_swaption = np.mean(payoff_payer) * self.DF[numeraire_index, 0,0]


                    self.swaption_prices_calibration_receiver.at[row_index, column_index] \
                        = receiver_swaption

                    self.swaption_prices_calibration_payer.at[row_index, column_index] \
                        = payer_swaption

        np.savetxt('swap_rate_price_payer_model.csv', self.swaption_prices_calibration_payer, delimiter=',')
        np.savetxt('swap_rate_price_receiver_model.csv', self.swaption_prices_calibration_receiver, delimiter=',')

    def set_forward_sims(self, numeraire_index, number_of_projection_periods):
        self.forward_sims = np.zeros((numeraire_index,
                                          number_of_projection_periods+1,
                                          self.number_of_sims))
        self.forward_sims[:, 0, :] = np.tile(self.starting_forward_curve[:numeraire_index],
                                                 (self.number_of_sims, 1)).transpose()

    def get_forward_swap_rate(self, time_index, numeraire_index):
        sum = np.zeros(self.number_of_sims)

        for i in range(time_index+1, numeraire_index + 1):
            sum += self.time_increment*self.DF[i, time_index, :]
        output = (1 - (self.DF[numeraire_index, time_index,:]))/sum
        return output

    def set_discount_factors(self, numeraire_index, number_of_projection_periods):
        self.DF = np.ones((numeraire_index+1,
                            number_of_projection_periods+1,
                                          self.number_of_sims))
        for n in range(number_of_projection_periods+1):
            for i in range(n + 1, numeraire_index + 1):
                df_prod = np.ones(self.number_of_sims)
                for k in range(n, i):
                    df_prod = df_prod / (np.ones(self.number_of_sims) + self.time_increment * self.forward_sims[k,n,:])
                self.DF[i,n,:] = df_prod


    def run_projection(self, numeraire_index, number_of_projection_periods):
        self.set_forward_sims(numeraire_index, number_of_projection_periods)

        for n in range(number_of_projection_periods):
            diffusion = self.get_diffusion()

            for i in range(n+1, numeraire_index):
                    summation = np.zeros(self.number_of_sims)

                    for k in range(i + 1, numeraire_index):
                        forward_sims = self.forward_sims[k, n, :]
                        top = forward_sims * self.time_increment
                        bottom = np.ones(self.number_of_sims) + forward_sims * self.time_increment
                        quotient = top / bottom
                        correlation = self.volatility.correlation_matrix[i, k]
                        volatility = self.volatility.working_vol_array[k]
                        summation += quotient * correlation * volatility

                    drift = summation * self.volatility.working_vol_array[i]
                    correction = np.ones(self.number_of_sims)*self.volatility.covariance[i,i]/2
                    # as diffusion is time-homogenous it is the difference between term and time that counts
                    step_diffusion = diffusion[i-n-1,:]
                    step =  self.forward_sims[i, n, :] * np.exp((-drift - correction) * self.time_increment + step_diffusion)
                    self.forward_sims[i, n+1, :] = step
        self.set_discount_factors(numeraire_index, number_of_projection_periods)

    def run_projection_predictor_corrector(self, numeraire_index, number_of_projection_periods):
        self.set_forward_sims(numeraire_index, number_of_projection_periods)
        previous_drift = np.zeros((numeraire_index, self.number_of_sims))

        for n in range(number_of_projection_periods):
            diffusion = self.get_diffusion()

            for i in range(n + 1, numeraire_index):
                summation = np.zeros(self.number_of_sims)

                for k in range(i + 1, numeraire_index):
                    forward_sims = self.forward_sims[k, n, :]
                    top = forward_sims * self.time_increment
                    bottom = np.ones(self.number_of_sims) + forward_sims * self.time_increment
                    quotient = top / bottom
                    correlation = self.volatility.correlation_matrix[i, k]
                    volatility = self.volatility.working_vol_array[k]
                    summation += quotient * correlation * volatility

                constant_drift = summation * self.volatility.working_vol_array[i]

                if n == 0:
                    working_drift = constant_drift
                else:
                    working_drift = (previous_drift[i,:] + constant_drift)/2

                correction = np.ones(self.number_of_sims) * self.volatility.covariance[i, i] / 2
                step_diffusion = diffusion[i - n - 1, :]
                step = self.forward_sims[i, n, :] * np.exp((-working_drift - correction) * self.time_increment + step_diffusion)
                self.forward_sims[i, n + 1, :] = step
                previous_drift[i,:] = working_drift
        self.set_discount_factors(numeraire_index, number_of_projection_periods)

    def get_diffusion(self):
        random = self.get_random_numbers()
        vol_matrix = self.volatility.working_chol_matrix
        output = vol_matrix.dot(random)
        return output

    # Option expiry numeraire projection code. Not used in the main project
    # But used in the change of numeraire figure 12 results
    # Does not show same drift error as the run_projection code.
    # def run_projection_option_expiry_numeraire(self, numeraire_index):
    #     self.set_forward_sims(self.number_of_terms - numeraire_index - 1, numeraire_index)
    #
    #     for n in range(self.number_of_terms - numeraire_index - 1):
    #         diffusion = self.get_diffusion()
    #
    #         for i in range(numeraire_index, self.number_of_terms):
    #             if (i > n):
    #                 summation = np.zeros(self.number_of_sims)
    #                 for k in range(i + 1, self.number_of_terms):
    #                     # forward_sims = np.exp(self.log_forward_sims[k, time_index, :])
    #                     forward_sims = self.forward_sims[k - numeraire_index, n, :]
    #                     top = forward_sims * self.time_increment
    #                     bottom = np.ones(self.number_of_sims) + forward_sims * self.time_increment
    #                     quotient = top / bottom
    #                     correlation = self.volatility.correlation_matrix[i, k]
    #                     volatility = self.volatility.working_vol_array[k - n]
    #                     summation += quotient * correlation * volatility
    #
    #                 drift = summation * self.volatility.working_vol_array[i - n]
    #                 correction = np.ones(self.number_of_sims) * self.volatility.covariance[i, i] / 2
    #                 step_diffusion = diffusion[i - n - 1, :]
    #                 step = self.forward_sims[i - numeraire_index, n, :] * np.exp(
    #                     (-drift - correction) * self.time_increment + step_diffusion)
    #                 self.forward_sims[i - numeraire_index, n + 1, :] = step











