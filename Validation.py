import numpy as np
import LMM as lmm
import matplotlib.pyplot as plt

# validates the LMM forward rate simulations using martingale tests and other
# tests
class Validation():
    def __init__(self):
        swaption_vol_cva_dataset_path = 'Data/SwaptionVolMatrix_5Y.csv'
        swap_curve_cva_dataset_path = 'Data/SpotCurve_5Y.csv'
        self.lmm = lmm.LMM(swaption_vol_cva_dataset_path, swap_curve_cva_dataset_path)
        self.calculate_martingale_ratios_for_CVA_dataset_bonds_at_expiry()

        swaption_vol_extended_dataset_path = 'Data/SwaptionVolMatrix.csv'
        swap_curve_extended_dataset_path = 'Data/SpotCurve.csv'
        self.lmm = lmm.LMM(swaption_vol_extended_dataset_path, swap_curve_extended_dataset_path)
        self.calculate_10_year_ZC_martingale_test()

        self.calculate_zero_coupon_bond_projections()
        # uncomment to do diffusion check
        # self.check_diffusion_has_zero_mean()

        ##[terms,time, sim]
        self.forward_sims = self.lmm.forward_sims
        self.number_of_terms = self.lmm.number_of_terms
        self.time_increment = self.lmm.time_increment
        self.bootstrapping = self.lmm.bootstrapping
        self.number_of_sims = self.lmm.number_of_sims

    def set_martingale_differences_for_zero_coupon_bond(self):
        self.martingale_differences = np.ones((self.number_of_terms, 3))
        # loop through zero coupon bonds
        for i in range(1, self.number_of_terms+1):
            bond_pv = self.get_expectation_of_zero_coupon_bond(i)
            t0_bond_pv = self.bootstrapping.zero_coupon_prices[i]
            self.martingale_differences[i - 1,0] = bond_pv
            self.martingale_differences[i - 1, 1] = t0_bond_pv
            self.martingale_differences[i - 1, 2] = bond_pv / t0_bond_pv - 1
        np.savetxt('martingale_test.csv', self.martingale_differences, delimiter=',')

    def calculate_martingale_ratios_for_CVA_dataset_bonds_at_expiry(self):
        numeraire_index = 10
        self.lmm.run_projection(numeraire_index, numeraire_index)
        bonds = np.zeros(numeraire_index)
        ratio = np.zeros(numeraire_index)
        difference = np.zeros(numeraire_index)

        for i in range(1, numeraire_index):
            numeraire_value = self.lmm.DF[numeraire_index, i,:]
            t0_value = self.lmm.DF[numeraire_index,0,0]
            bonds[i] = np.mean(1/numeraire_value)*t0_value
            difference[i] = bonds[i] - self.lmm.DF[i,0,0]
            ratio[i] = bonds[i]/self.lmm.DF[i,0,0]
        np.savetxt('martingale_ratio_at_bond_expiry_CVA_dataset.csv', ratio, delimiter=',')

    def calculate_zero_coupon_bond_projections(self):
        numeraire_index = 40
        start_bond = 20
        self.lmm.volatility.mc_adjustment_factor = 1
        self.lmm.volatility.a = 0.01368861
        self.lmm.volatility.b = 0.07921976
        self.lmm.volatility.c = 0.33920146
        self.lmm.volatility.d = 0.08416935
        self.lmm.volatility.instantiate_arrays()
        self.lmm.run_projection(numeraire_index, numeraire_index)
        bonds = np.zeros((numeraire_index - start_bond, numeraire_index))
        ratio = np.zeros((numeraire_index - start_bond, numeraire_index))

        for i in range(start_bond, numeraire_index):
            for j in range(i+1):
                numeraire_value = self.lmm.DF[numeraire_index, j, :]
                t0_numeraire_value = self.lmm.DF[numeraire_index, 0, 0]
                t0_ratio =  self.lmm.DF[i, 0, 0]/t0_numeraire_value
                bonds[i-start_bond,j] = np.mean(self.lmm.DF[i, j,:] / numeraire_value) * t0_numeraire_value
                ratio[i-start_bond,j] = (np.mean(self.lmm.DF[i, j,:] / numeraire_value))/t0_ratio
        np.savetxt('matingale_test_ratio_projections.csv', ratio, delimiter=',')

    def calculate_10_year_ZC_martingale_test(self):
        numeraire_index = 40
        start_bond = 20
        self.lmm.volatility.mc_adjustment_factor = 1
        self.lmm.volatility.a = 0.01368861
        self.lmm.volatility.b = 0.07921976
        self.lmm.volatility.c = 0.33920146
        self.lmm.volatility.d = 0.08416935
        self.lmm.volatility.instantiate_arrays()
        self.lmm.run_projection(numeraire_index, numeraire_index)
        # self.lmm.run_projection_predictor_corrector(numeraire_index, numeraire_index)

        bonds = np.zeros((numeraire_index - start_bond, numeraire_index))
        ratio = np.zeros((4, numeraire_index))
        for j in range(start_bond + 1):
            numeraire_value = self.lmm.DF[numeraire_index, j, :]
            t0_numeraire_value = self.lmm.DF[numeraire_index, 0, 0]
            t0_ratio = self.lmm.DF[start_bond, 0, 0] / t0_numeraire_value
            bonds[start_bond - start_bond, j] = np.mean(self.lmm.DF[start_bond, j, :] / numeraire_value) * t0_numeraire_value
            ratio[0, j] = np.percentile(self.lmm.DF[start_bond, j, :] / numeraire_value, 5) / t0_ratio
            ratio[1, j] = (np.mean(self.lmm.DF[start_bond, j, :] / numeraire_value)) / t0_ratio
            ratio[2, j] = np.percentile(self.lmm.DF[start_bond, j, :] / numeraire_value, 50) / t0_ratio
            ratio[3, j] = np.percentile(self.lmm.DF[start_bond, j, :] / numeraire_value, 95) / t0_ratio
        np.savetxt('10_year_ZC_martingale_test.csv', ratio, delimiter=',')

    def get_expectation_of_zero_coupon_bond(self, zero_coupon_index):
        forward_rate_index = zero_coupon_index - 1
        product = 1/(np.ones(self.number_of_sims) + self.time_increment*self.forward_sims[0,0,:])

        for i in range(1, forward_rate_index+1):
            product = product/(np.ones(self.number_of_sims) + self.time_increment*self.forward_sims[i,i,:])
        output = np.mean(product)
        return output

    def check_diffusion_has_zero_mean(self):
        number_of_tests = 40
        mean = np.zeros((number_of_tests,12))

        for i in range(number_of_tests):
            diffusion = self.lmm.get_diffusion()
            mean[i,:] = np.mean(diffusion, axis=1)
        mean_of_mean = np.mean(mean)


