import numpy as np
import Volatility as vol
import pandas as pd
import Bootstrapping as boot
import copy as copy
import BlackScholesSolver as bss
from LMM import *
import matplotlib.pyplot as plt
import CDSBootstrapping as cds_boot

# 1. calculates LOIS adjusted OIS simulations
# 2. calculates CVA for 5 year IRS with Goldman Sachs as at 31/12/17
# 3. Conducts sensitivity analysis to LOIS and LOIS-CDS correlation
class CVASwap():
    def __init__(self):
        self.recovery_rate = 0.4
        swaption_vol_cva_dataset_path = 'Data/SwaptionVolMatrix_5Y.csv'
        swap_curve_cva_dataset_path = 'Data/SpotCurve_5Y.csv'
        path = 'ois.csv'
        self.lmm = LMM(swaption_vol_cva_dataset_path, swap_curve_cva_dataset_path)
        self.time_increment = self.lmm.time_increment
        self.bootstrapping = self.lmm.bootstrapping
        self.lois = 0.00282385038573462
        self.working_lois = self.lois
        self.numeraire_index = 10
        self.set_ois_discount_curve(self.lois)
        self.cds_bootstrapping = cds_boot.CDSBootstrapping(self.recovery_rate,
                                                        self.time_increment)
        self.lmm.run_projection(self.numeraire_index, self.numeraire_index)
        ##[terms,time, sim]
        self.forward_sims = self.lmm.forward_sims
        self.number_of_sims = self.lmm.number_of_sims
        self.notional = 1
        self.LGD = self.notional*(1- self.recovery_rate)
        self.swap_term = 5
        self.set_ois_discount_sims()
        self.calculate_CVA_for_5_year_swap()
        self.calculate_martingale_ratios_for_CVA_dataset_bonds_at_expiry_ois()
        # uncomment this to run sensiivity analysis
        # self.run_sensitivity_analysis()

    def set_ois_discount_curve(self, lois):
        libor_forward_rates = self.lmm.starting_forward_curve
        self.ois_discount_factors = np.ones(self.numeraire_index + 1)

        for i in range(1, self.numeraire_index + 1):
            self.ois_discount_factors[i] = self.ois_discount_factors[i-1]/(1 + self.time_increment*(libor_forward_rates[i-1] - lois))

    def set_ois_discount_sims(self):
        lois_sims = self.working_lois * np.ones(self.number_of_sims)
        self.ois_DF = np.ones((self.numeraire_index + 1,
                           self.numeraire_index + 1,
                           self.number_of_sims))
        for n in range(self.numeraire_index + 1):
            for i in range(n + 1, self.numeraire_index + 1):
                df_prod = np.ones(self.number_of_sims)
                for k in range(n, i):
                    df_prod = df_prod / \
                              (np.ones(self.number_of_sims) + self.time_increment * (self.forward_sims[k, n, :] - lois_sims))
                self.ois_DF[i, n, :] = df_prod

    def get_pv_of_swap_alternate(self, swap_length_steps, strike, time_index):
        swap_rate_libor = self.lmm.get_forward_swap_rate(time_index, swap_length_steps)
        annuity = np.zeros(self.number_of_sims)

        for i in range(time_index + 1, self.numeraire_index+1):
            annuity += self.ois_DF[i,time_index,:]
        output = (swap_rate_libor - strike)*self.time_increment*annuity
        return output

    def calculate_exposure(self, swap_pv):
        output = np.maximum(swap_pv, np.zeros(self.number_of_sims))
        return output

    def run_sensitivity_analysis(self):
        swap_rate = float(self.bootstrapping.imported_swap_curve.iloc[self.swap_term][self.bootstrapping.yield_name])
        swap_length_steps = int(self.swap_term / self.time_increment)
        correlation_increments = 8
        lois_increments = 62
        cva_matrix = np.zeros((correlation_increments, lois_increments))
        lois_values = np.zeros(lois_increments)
        correlations = np.zeros(correlation_increments)
        cds_multipliers = np.zeros(lois_increments)

        for i in range(correlation_increments):
            correlation = 0.2 + 0.2*i
            correlations[i] = correlation
            for j in range(lois_increments):
                lois_multiplier = 0.1 + j*0.1
                lois_values[j] = self.lois*lois_multiplier
                self.working_lois = self.lois*lois_multiplier
                self.set_ois_discount_curve(self.working_lois)
                self.set_ois_discount_sims()
                cds_multiplier = correlation*lois_multiplier
                cds_multipliers[j] = cds_multiplier
                cva_matrix[i,j] = self.calculate_cva(swap_length_steps, swap_rate, cds_multiplier)
        np.savetxt('cva_matrix.csv', cva_matrix, delimiter=',')
        np.savetxt('lois_values.csv', lois_values, delimiter=',')

    def calculate_CVA_for_5_year_swap(self):
        swap_rate = float(self.bootstrapping.imported_swap_curve.iloc[self.swap_term][self.bootstrapping.yield_name])
        swap_length_steps = int(self.swap_term / self.time_increment)
        cds_multiplier = 1
        cva = self.calculate_cva(swap_length_steps, swap_rate, cds_multiplier)

        exposure_stats = self.get_stats(self.exposure, swap_length_steps)
        mtm_stats = self.get_stats(self.swap_mtm, swap_length_steps)
        discounted_exposure_stats = self.get_stats(self.discounted_exposures, swap_length_steps)

        np.savetxt('exposure_stats.csv', exposure_stats, delimiter=',')
        np.savetxt('mtm_stats.csv', mtm_stats, delimiter=',')
        np.savetxt('discounted_exposure_stats.csv', discounted_exposure_stats, delimiter=',')

    def calculate_martingale_ratios_for_CVA_dataset_bonds_at_expiry_ois(self):
        bonds = np.zeros(self.numeraire_index)
        ratio = np.zeros(self.numeraire_index)
        difference = np.zeros(self.numeraire_index)

        for i in range(1, self.numeraire_index):
            numeraire_value = self.ois_DF[self.numeraire_index, i, :]
            t0_value = self.ois_DF[self.numeraire_index, 0, 0]
            bonds[i] = np.mean(1 / numeraire_value) * t0_value
            difference[i] = bonds[i] - self.ois_DF[i, 0, 0]
            ratio[i] = bonds[i] / self.ois_DF[i, 0, 0]
        np.savetxt('martingale_ratio_at_bond_expiry_CVA_dataset_ois_adjustment.csv', ratio, delimiter=',')

    def get_stats(self, sims, swap_length_steps):
        output = np.zeros((swap_length_steps + 1, 4))
        output[:, 0] = np.mean(sims, axis=1)
        output[:, 1] = np.median(sims, axis=1)
        # no non-int percentiles in numpy so average 97 and 98th percentiles to get 97.5th percentile
        output[:, 2] = (np.percentile(sims, 97, axis=1) + np.percentile(sims, 98, axis=1)) / 2
        output[:, 3] = np.percentile(sims, 99, axis=1)
        return output


    def calculate_exposures(self, swap_length_steps, swap_rate):
        self.exposure = np.zeros((swap_length_steps + 1, self.number_of_sims))
        self.swap_mtm = np.zeros((swap_length_steps + 1, self.number_of_sims))

        for t in range(swap_length_steps):
            swap_pv = self.get_pv_of_swap_alternate(swap_length_steps, swap_rate,t)
            self.swap_mtm[t] = swap_pv
            swap_exposure = self.calculate_exposure(swap_pv)
            self.exposure[t] = swap_exposure
        return self.exposure

    def get_discounted_exposures(self, exposures, swap_length_steps):
        self.discounted_exposures = np.zeros((swap_length_steps + 1, self.number_of_sims))

        for i in range(swap_length_steps+1):
            self.discounted_exposures[i,:] = self.ois_discount_factors[self.numeraire_index]\
                                             *exposures[i]/self.ois_DF[self.numeraire_index,i,:]
        return self.discounted_exposures


    def calculate_cva(self, swap_length_steps, swap_rate, cds_multiplier):
        exposures = self.calculate_exposures(swap_length_steps, swap_rate)
        discounted_exposures = self.get_discounted_exposures(exposures, swap_length_steps)
        probability_of_survival = self.cds_bootstrapping.get_probability_of_survival(self.ois_discount_factors, cds_multiplier)
        # np.savetxt('probability_of_survival.csv', probability_of_survival, delimiter=',')
        sum = np.zeros(self.number_of_sims)
        for i in range(1, swap_length_steps):
            marginal_probability_of_default = probability_of_survival[i-1] \
                                              - probability_of_survival[i]
            intepolated_exposure = (discounted_exposures[i - 1] + discounted_exposures[i]) / 2
            sum += np.maximum(0, marginal_probability_of_default)*intepolated_exposure
        self.CVA_stochastic = sum*self.LGD
        self.CVA_mean = np.mean(self.CVA_stochastic)
        CVA_99 = np.percentile(self.CVA_stochastic, 99)
        CVA_ninety_seventh = (np.percentile(self.CVA_stochastic, 97) + np.percentile(self.CVA_stochastic, 98)) / 2
        CVA_seventy_fifth = np.percentile(self.CVA_stochastic, 75)
        CVA_twenty_fifth = np.percentile(self.CVA_stochastic, 25)
        s = 'The mean CVA value of a 5 year USD libor interest rate swap as at 31/12/17 is ' + str(self.CVA_mean) + ' for a notional of 1.'
        print(s)
        s = 'The 99th percentile CVA value of a 5 year USD libor interest rate swap as at 31/12/17 is ' + str(
            CVA_99) + ' for a notional of 1.'
        print(s)
        return self.CVA_mean



