import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Imports swap curve and interpolates using linear interpolation, svensson fitting and cubic spline
# 2. Using linear interpolation bootstraps zero coupon bond values from swap rates
# 3. Generates Z matrix for use in Rebonato method
# 4. Calculates forward swap rates from forward rates for use in swaption pricing.
class Bootstrapping():
    def __init__(self, swap_curve_path):
        # path = 'SpotCurve cut down.csv'
        path = swap_curve_path
        self.term_name = 'term'
        self.yield_name = 'yield'
        self.term_increment = 0.5
        self.notional = 1.0
        self.min_term = 0.5

        self.imported_swap_curve = pd.read_csv(path)
        self.max_term = int(max(self.imported_swap_curve[self.term_name]))
        self.number_of_terms = int((self.max_term) / self.term_increment)
        self.set_linear_interpolated_swap_rates_vector()
        # self.svensson = sven.svensson()
        # self.svensson.fit_parameters(self.imported_swap_curve)
        # self.swap_curve_function = \
        #     np.poly1d(np.polyfit(self.imported_swap_curve[self.term_name],
        #                          self.imported_swap_curve[self.yield_name], 3))

        # xp = np.linspace(0.5, 50, 100)
        # plt.plot(self.imported_swap_curve[self.term_name], self.imported_swap_curve[self.yield_name],
        #          '.', xp, self.swap_curve_function(xp), '-', xp, self.swap_curve_function(xp), '--')
        # plt.ylim(-2, 2)
        # plt.show()

        # xp = np.linspace(0.5, 50, 100)
        # plt.plot(self.imported_swap_curve[self.term_name], self.imported_swap_curve[self.yield_name],
        #          '.', xp, self.svensson.get_value(xp), '-', xp, self.svensson.get_value(xp), '--')
        # plt.ylim(-2, 2)
        # plt.show()
        self.set_zero_coupon_prices_from_swap_rates()
        # xp = np.linspace(0.25, 29.75, 119)
        # plt.plot(xp, self.zero_coupon_prices,
        #          '.', xp, self.swap_curve_function(xp), '-', xp, self.swap_curve_function(xp), '--')
        # plt.ylim(-2, 2)
        # plt.show()
        self.set_forward_curve()

    # Ai in Jaekal notes. t = 0. start_index = i
    def get_t0_floating_leg_value(self, start_index, end_index):
        sum = 0
        for j in range(start_index, end_index):
            sum += self.zero_coupon_prices[j+1]*self.forward_curve[j]*self.term_increment*self.notional
        return sum

    # Bi in Jaekal notes. t = 0
    def get_t0_fixed_leg_value(self, start_index, end_index):
        sum = 0
        for j in range(start_index, end_index):
            sum += self.zero_coupon_prices[j+1]*self.term_increment*self.notional
        return sum

    def convert_index_to_term_forward(self, i):
        return i*self.term_increment

    def set_Z_matrix(self, number_of_indexes):
        self.Z_matrix = np.zeros((number_of_indexes, number_of_indexes, number_of_indexes))

        for start_index in range(0, number_of_indexes):
            for end_index in range(start_index + 1, number_of_indexes):
                for k in range(start_index, end_index):
                    self.Z_matrix[start_index, end_index, k] = self.get_Z(start_index, end_index, k)

    def get_Z(self, start_index, end_index, k):
        if k < start_index:
            return 0
        if k == start_index:
            return 1

        const_weights_approx = self.notional*self.zero_coupon_prices[k+1]\
                               *self.forward_curve[k]*self.term_increment\
                               /self.get_t0_floating_leg_value(start_index, end_index)
        top = (self.get_t0_floating_leg_value(start_index, end_index)
               *self.get_t0_fixed_leg_value(k, end_index) - self.get_t0_floating_leg_value(k, end_index)
               *self.get_t0_fixed_leg_value(start_index, end_index))\
              *self.forward_curve[k]*self.term_increment
        bottom = self.get_t0_floating_leg_value(start_index, end_index)*(1 + self.forward_curve[k]*self.term_increment)
        shape_correction = top/bottom
        return const_weights_approx + shape_correction

    def set_linear_interpolated_swap_rates_vector(self):
        self.interpolated_swap_rates = np.zeros(self.number_of_terms)
        previous_swap_length_steps = 0
        previous_swap_length = 0
        previous_swap_rate = 0
        count = 0

        for row_index, row in self.imported_swap_curve.iterrows():
            swap_length = row[self.term_name]
            swap_length_steps = int((swap_length-self.min_term)/ self.term_increment)
            swap_rate = float(row[self.yield_name])

            if row_index == 0:
                self.interpolated_swap_rates[count] = swap_rate
                previous_swap_length_steps = swap_length_steps
                previous_swap_length = swap_length
                previous_swap_rate = swap_rate
                count += 1
            else:
                distance = swap_length_steps - previous_swap_length_steps

                for i in range(1, distance):
                    interpolate_length = previous_swap_length + self.term_increment*i
                    interpolate_swap_rate = (interpolate_length - previous_swap_length)*swap_rate/(swap_length - previous_swap_length)\
                                + (swap_length - interpolate_length)*previous_swap_rate/(swap_length - previous_swap_length)
                    self.interpolated_swap_rates[count] = interpolate_swap_rate
                    count += 1
                self.interpolated_swap_rates[count] = swap_rate
                previous_swap_rate = swap_rate
                previous_swap_length = swap_length
                previous_swap_length_steps = swap_length_steps
                count += 1
        # np.savetxt('linear_interpolated_swap_values.csv', self.interpolated_swap_rates, delimiter=',')

    def get_forward_swap_rates(self, start, swap_length):
        start_index = int(start/self.term_increment)
        end_index = int((start + swap_length)/self.term_increment)
        top = self.zero_coupon_prices[start_index] - self.zero_coupon_prices[end_index]
        bottom = 0

        for i in range(start_index + 1, end_index + 1):
            bottom += self.term_increment * self.zero_coupon_prices[i]
        return top/bottom

    # start index = alpha, end_index = beta
    def get_forward_swap_rates_from_forward_rates(self, start_index, end_index, forward_rates):
        number_of_sims = len(forward_rates[0,:])

        product = 1/(1+ self.term_increment*forward_rates[start_index, :])
        for i in range(start_index + 1, end_index):
            product = product/(1 + self.term_increment*forward_rates[i, :])
        floating_leg = np.ones(number_of_sims) - product

        bottom = 0

        for i in range(start_index, end_index):
            product = 1/(1+ self.term_increment*forward_rates[start_index, :])

            for j in range(start_index + 1, i+1):
                product = product/(1 + self.term_increment*forward_rates[j, :])

            bottom += self.term_increment*product
        return floating_leg/bottom

    # forward_rates [term, sim]
    def get_discount_factors_from_forward_rates(self, start_index, end_index, forward_rates, lois):
        number_of_sims = len(forward_rates[0, :])
        lois_vector = np.ones(number_of_sims)*lois

        product = 1/(1 + self.term_increment*(forward_rates[start_index, :] - lois_vector))
        for i in range(start_index + 1, end_index):
            product/(1 + self.term_increment*(forward_rates[i, :] - lois_vector))
        return product

    def set_ois_adjusted_discount_factors(self, lois):
        self.ois_discount_factors = np.zeros(self.number_of_terms + 1)
        self.ois_discount_factors[0] = 1
        self.ois_discount_factors[1] = 1 / (1 + self.term_increment * (self.forward_curve[0] - lois))

        for i in range(2, self.number_of_terms + 1):
            self.ois_discount_factors[i] = self.ois_discount_factors[i-1]/ (1 + self.term_increment * (self.forward_curve[i-1] - lois))

    # zero_coupon_prices[0] = 1
    def set_zero_coupon_prices_from_swap_rates(self):
        self.zero_coupon_prices = np.zeros(self.number_of_terms + 1)

        self.zero_coupon_prices[0] = 1
        self.zero_coupon_prices[1] = 1/(1 + self.term_increment*self.interpolated_swap_rates[0])

        for i in range(2, self.number_of_terms + 1):
            rN = self.interpolated_swap_rates[i - 1]
            sum = 0
            for k in range(1, i):
                sum += self.zero_coupon_prices[k]
            self.zero_coupon_prices[i] = (1 - rN*sum*self.term_increment)/(1+ rN*self.term_increment)

    def get_pv_of_swap_t0(self, swap_length_steps, swap_rate, forward_sims):
        fixed_leg = 0
        floating_leg = 0

        for i in range(swap_length_steps):
            zero_coupon_bond = 1 / (1 + self.term_increment * (forward_sims[0, :]))

            for j in range(1, i + 1):
                zero_coupon_bond = zero_coupon_bond / (
                    1 + self.term_increment * (forward_sims[j, :]))
                fixed_leg += zero_coupon_bond * swap_rate * self.term_increment
                floating_leg += zero_coupon_bond * forward_sims[i, :] * self.term_increment
        return floating_leg - fixed_leg


    # forward_curve[0] = t = 0, 0.5 term spot rate
    def set_forward_curve(self):
        self.forward_curve = np.zeros(self.number_of_terms)
        self.forward_curve[0] = self.interpolated_swap_rates[0]

        for i in range(1, self.number_of_terms):
            self.forward_curve[i] = (self.zero_coupon_prices[i]/self.zero_coupon_prices[i+1] - 1)/self.term_increment

