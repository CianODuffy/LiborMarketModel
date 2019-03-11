import numpy as np
import scipy.optimize as op
from scipy.optimize import least_squares
from math import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import copy

# 1. implements Rebonato method for calibration to swaption implied volatilities
# 2. generates correlation and diffusion matrices for use in the LMM module
# 3. Calculates swaption price from swaption volatility using Black formula for swaptions
class Volatility():
    def __init__(self, number_of_factors, bootstrapping, use_factor_reduction, swaption_vol_matrix_path):
        path = swaption_vol_matrix_path
        self.use_factor_reduction = use_factor_reduction
        self.vol_matrix = pd.read_csv(path)
        self.bootstrapping = bootstrapping
        self.term_name = 'expiry'
        self.terms = np.array(self.vol_matrix[self.term_name], float)
        self.tenors = np.array(self.vol_matrix.columns)[1:].astype(np.float)
        self.a =  2.81170636
        self.b = 0.30994865
        self.c = 0.08030116
        self.d =  -2.74048478
        self.beta = 0.1
        self.number_of_factors = number_of_factors
        self.min_term = self.bootstrapping.min_term
        self.time_increment = self.min_term
        self.max_term = self.bootstrapping.max_term
        self.number_of_terms = self.bootstrapping.number_of_terms
        self.mc_adjustment_factor = 0.954816569302
        # self.mc_adjustment_factor = 1

        self.constant_increment_times = \
            np.arange(0, self.max_term, self.time_increment)
        self.instantiate_arrays()

    def instantiate_arrays(self):
        self.set_vol_array()
        self.set_correlation_and_covariance_matrix()
        self.set_diffusion_matrix()

    def set_vol_array(self):
        time_array = np.zeros(self.number_of_terms)
        term_array = np.linspace(self.time_increment, self.max_term, num=(self.number_of_terms))
        self.working_vol_array = ((self.a + self.b*(term_array - time_array))*np.exp(-self.c*(term_array - time_array)) + self.d)*self.mc_adjustment_factor
        self.forward_vol_matrix = np.diag(self.working_vol_array)

    def set_calibrated_vol_matrix(self):
        self.bootstrapping.set_Z_matrix(self.number_of_terms)
        self.calibrated_swaption_vol_matrix = copy.deepcopy(self.vol_matrix)
        start_time = 0.0

        for row_index, row in self.calibrated_swaption_vol_matrix.iterrows():
            values = row.drop([self.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_start_index = int(row[self.term_name] / self.time_increment)
                    swap_end_index = int(float(column_index) / self.time_increment + swap_start_index)
                    calculated_vol = self.get_swap_volatility(start_time, swap_start_index, swap_end_index)
                    self.calibrated_swaption_vol_matrix.at[row_index, column_index] = calculated_vol
        np.savetxt('calibrated_vols.csv', self.calibrated_swaption_vol_matrix, delimiter=',')


    def calculate_volatility(self, time, term):
        output = ((self.a + self.b*(term - time))*np.exp(-self.c*(term - time)) + self.d)
        return output

    def set_parameters_swap(self, parameters):
        self.a = parameters[0]
        self.b = parameters[1]
        self.c = parameters[2]
        self.d = parameters[3]

    def get_parameters_swap(self):
        output = np.zeros(4)
        output[0] = self.a
        output[1] = self.b
        output[2] = self.c
        output[3] = self.d
        return output

    def get_swaption_price_t0_payer(self, start, swap_length, strike, swaption_volatility):
        d1 = (np.log(self.bootstrapping.get_forward_swap_rates(start, swap_length)/strike) +
              start*np.power(swaption_volatility, 2)/2)/(swaption_volatility*sqrt(start))
        d2 = d1 - swaption_volatility*sqrt(start)
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)

        sum = 0
        start_index = int((start/self.time_increment) - 1)
        end_index = int(((start + swap_length)/self.time_increment)-1)

        for i in range(start_index + 1, end_index + 1):
            sum += self.time_increment\
                   *self.bootstrapping.zero_coupon_prices[i]\
                   *(self.bootstrapping.get_forward_swap_rates(start, swap_length)
                     *Nd1 - strike*Nd2)
        return sum

    def get_swaption_price_t0_receiver(self, start, swap_length, strike, swaption_volatility):
        d1 = (np.log(self.bootstrapping.get_forward_swap_rates(start, swap_length)/strike) +
              start*np.power(swaption_volatility, 2)/2)/(swaption_volatility*sqrt(start))
        d2 = d1 - swaption_volatility*sqrt(start)
        Nd1 = norm.cdf(-d1)
        Nd2 = norm.cdf(-d2)

        sum = 0
        start_index = int((start/self.time_increment) - 1)
        end_index = int(((start + swap_length)/self.time_increment)-1)

        for i in range(start_index + 1, end_index + 1):
            sum += self.time_increment\
                   *self.bootstrapping.zero_coupon_prices[i]\
                   *(strike*Nd2 -self.bootstrapping.get_forward_swap_rates(start, swap_length)*Nd1)
        return sum

    def set_swaption_price_matrix(self):
        self.swaption_prices = copy.deepcopy(self.vol_matrix)

        for row_index, row in self.vol_matrix.iterrows():
            values = row.drop([self.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_length = float(column_index)
                    start = row[self.term_name]
                    # if swap_length < 15.0 and start < 15.0:
                    K = self.bootstrapping.get_forward_swap_rates(start, swap_length)
                    swaption_price_t0 = self.get_swaption_price_t0_payer(start, swap_length, K, v)
                    self.swaption_prices.at[row_index, column_index] =  swaption_price_t0

    def fit_parameters(self):
        initial_parameters = self.get_parameters_swap()
        self.bootstrapping.set_Z_matrix(self.number_of_terms)
        result = least_squares(self.objective_function_swap, initial_parameters)
        return result


    def objective_function_swap(self, parameters):
        num_rows = self.vol_matrix[self.term_name].count()
        num_columns = len(self.vol_matrix.columns) - 1
        number_of_iterations = int(num_rows*num_columns)
        sum = np.zeros(number_of_iterations)
        N = 0
        start_time = 0.0
        self.set_parameters_swap(parameters)

        for row_index, row in self.vol_matrix.iterrows():
            values = row.drop([self.term_name])
            for column_index, v in values.items():
                if False == isnan(v):
                    swap_start_index = int(row[self.term_name]/self.time_increment)
                    swap_end_index = int(float(column_index)/self.time_increment + swap_start_index)
                    calculated_vol = self.get_swap_volatility(start_time, swap_start_index, swap_end_index)
                    sum[N] = calculated_vol - v
                    N += 1
        return sum

    def get_swap_volatility(self, option_start_time, swap_start_index, swap_end_index):
        total_sum = 0
        for k in range(swap_start_index, swap_end_index):
            for l in range(swap_start_index, swap_end_index):
                time_k = self.constant_increment_times[k]
                time_l = self.constant_increment_times[l]
                swap_start_time = swap_start_index*self.time_increment

                integrated_covariance_end = self.get_integrated_covariance(swap_start_time, time_k, time_l)
                integrated_covariance_start = self.get_integrated_covariance(option_start_time, time_k, time_l)

                integrated_covariance = (integrated_covariance_end - integrated_covariance_start)\
                                        /(swap_start_time - option_start_time)
                first_Z = self.bootstrapping.Z_matrix[swap_start_index, swap_end_index, k]
                second_Z = self.bootstrapping.Z_matrix[swap_start_index, swap_end_index, l]
                total_sum += first_Z*integrated_covariance*second_Z
        output = np.sqrt(total_sum)
        return output


    def get_correlation(self, term1, term2):
        return np.exp(-self.beta * abs(term1 - term2))

    def set_correlation_and_covariance_matrix(self):
        self.correlation_matrix = np.ones((self.number_of_terms, self.number_of_terms))

        for i in range(0, self.number_of_terms):
            for j in range(0, self.number_of_terms):
                self.correlation_matrix[i, j] = self.get_correlation((i+1)*self.time_increment, (j+1)*self.time_increment)
        self.covariance = np.matmul(self.forward_vol_matrix, np.matmul(self.correlation_matrix, self.forward_vol_matrix))

    def set_diffusion_matrix(self):
        if self.mc_adjustment_factor == 0:
            self.chol_covariance = self.covariance
            self.working_chol_matrix = self.covariance
        else:
            self.chol_covariance = np.linalg.cholesky(self.covariance)

            if self.use_factor_reduction:
                self.working_chol_matrix = self.chol_covariance[:, :self.number_of_factors]

                for i in range(0, self.number_of_terms):
                    for j in range(0, self.number_of_factors):
                        sum = np.sum(np.power(self.chol_covariance[i,:self.number_of_factors], 2))

                        quotient = self.covariance[i, i] / sum
                        self.working_chol_matrix[i, j] = self.working_chol_matrix[i, j] * np.sqrt(quotient)
            else:
                self.working_chol_matrix = self.chol_covariance


    def get_integrated_covariance(self, time, term_i, term_j):
        first_line = 4*self.a*np.power(self.c, 2)*self.d*(np.exp(self.c*(time - term_j))
                                                                 + np.exp(self.c*(time - term_i))) \
                     + 4*np.power(self.c, 3)*np.power(self.d, 2)*time
        second_line = -4*self.b*self.c*self.d*np.exp(self.c*(time - term_i))*(self.c*(time - term_i) - 1) \
                      - 4*self.b*self.c*self.d*np.exp(self.c*(time - term_j))*(self.c*(time - term_j) - 1)
        third_line = np.exp(self.c*(2*time - term_i - term_j))\
                     *(2*np.power(self.a, 2)*np.power(self.c, 2)
                       + 2*self.a*self.b*self.c*(1 + self.c*(term_i + term_j - 2*time))
                       + np.power(self.b, 2)*(1 + 2*np.power(self.c, 2)*(time - term_i)*(time - term_j)
                                              + self.c*(term_i + term_j - 2*time)))
        multiplier = self.get_correlation(term_i, term_j)/(4*np.power(self.c, 3))
        return multiplier*(first_line + second_line + third_line)

    def get_correlated_volatility_cholesky_matrix(self, time):
        c_matrix = np.ones([self.number_of_terms,self.number_of_terms])

        for i in range(0, self.number_of_terms):
            term1 = i*self.time_increment
            for j in range(0, self.number_of_terms):
                term2 = j*self.time_increment
                c_matrix[i, j] = self.get_correlation(term1, term2)\
                               *self.calculate_volatility(time, term1)\
                               *self.calculate_volatility(time, term2)
        a_bar_matrix = np.linalg.cholesky(c_matrix)

        if self.use_factor_reduction:
            reduced = np.zeros([self.number_of_terms, self.number_of_factors])

            for i in range(0, self.number_of_terms):
                for j in range(0, self.number_of_factors):
                    sum = 0

                    for k in range(0, self.number_of_factors):
                        sum += np.power(a_bar_matrix[i, k],2)

                    quotient = c_matrix[i, i]/sum
                    reduced[i, j] = a_bar_matrix[i, j] * np.sqrt(quotient)
            return reduced
        return a_bar_matrix









