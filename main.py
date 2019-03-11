import LMM as lmm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import Bootstrapping as boot
import CVASwap as cva
import Validation as val

#THE PURPOSE OF THIS SCRIPT IS TO CALL THE OTHER METHODS AND PRINT RESULTS
#run main to run the other scripts.
# modules are run for 10,000 simulations as a compromise between run-time
# and acheiving good results for the monte-carlo optimisation
# in the project 50,000 simulations was used.

swaption_vol_cva_dataset_path = 'Data/SwaptionVolMatrix_5Y.csv'
swap_curve_cva_dataset_path = 'Data/SpotCurve_5Y.csv'
swaption_vol_extended_dataset_path = 'Data/SwaptionVolMatrix.csv'
swap_curve_extended_dataset_path = 'Data/SpotCurve.csv'

# run cva calculation
cva_object = cva.CVASwap()

# Martingale testing and zero mean diffusion methods in here
validate = val.Validation()


# Carry out analytical calibration for extended dataset
libor_fit_parameters_extended = lmm.LMM(swaption_vol_extended_dataset_path, swap_curve_extended_dataset_path)
libor_fit_parameters_extended.volatility.mc_adjustment_factor = 1
libor_fit_parameters_extended.volatility.a = 0.5
libor_fit_parameters_extended.volatility.b = 0.5
libor_fit_parameters_extended.volatility.c = 0.5
libor_fit_parameters_extended.volatility.d = 0.5
libor_fit_parameters_extended.volatility.fit_parameters()
s = 'Extended dataset analytical calibration  - a = ' + str(libor_fit_parameters_extended.volatility.a)  \
    + '\n Extended dataset analytical calibration  - b = ' + str(libor_fit_parameters_extended.volatility.b) \
+ '\n Extended dataset analytical calibration  - c = ' + str(libor_fit_parameters_extended.volatility.c) \
+ '\n Extended dataset analytical calibration  - d = ' + str(libor_fit_parameters_extended.volatility.d)
print(s)


# calculate swaption vols and prices for cva dataset using factor reduction
libor_cva_dataset_factor_reduction = lmm.LMM(swaption_vol_cva_dataset_path, swap_curve_cva_dataset_path)
libor_cva_dataset_factor_reduction.use_factor_reduction = True
libor_cva_dataset_factor_reduction.volatility.use_factor_reduction = True
libor_cva_dataset_factor_reduction.volatility.instantiate_arrays()
libor_cva_dataset_factor_reduction.set_swaption_prices_for_atm_calibration()
libor_cva_dataset_factor_reduction.set_implied_volatilities_from_prices()


# fit mc adjustment factor
libor_cva_dataset_mc_adjustment = lmm.LMM(swaption_vol_cva_dataset_path, swap_curve_cva_dataset_path)
libor_cva_dataset_mc_adjustment.fit_adjustment_factor()
k = libor_cva_dataset_mc_adjustment.volatility.mc_adjustment_factor
s = 'k(monte carlo) calculated as = ' + str(k)
print(s)


# calculate swaption vols and prices for cva dataset
libor_cva_dataset = lmm.LMM(swaption_vol_cva_dataset_path, swap_curve_cva_dataset_path)
libor_cva_dataset.set_swaption_prices_for_atm_calibration()
libor_cva_dataset.set_implied_volatilities_from_prices()

# calculate swaption vols and prices for extended dataset
libor_extended_dataset = lmm.LMM(swaption_vol_extended_dataset_path, swap_curve_extended_dataset_path)
libor_extended_dataset.volatility.mc_adjustment_factor = 1
libor_extended_dataset.volatility.a =  0.01368861
libor_extended_dataset.volatility.b =  0.07921976
libor_extended_dataset.volatility.c = 0.33920146
libor_extended_dataset.volatility.d = 0.08416935
libor_extended_dataset.volatility.instantiate_arrays()
libor_extended_dataset.set_swaption_prices_for_atm_calibration()
libor_extended_dataset.set_implied_volatilities_from_prices()

# run extended dataset and produce charts
libor_extended_dataset_produce_charts = lmm.LMM(swaption_vol_extended_dataset_path, swap_curve_extended_dataset_path)
libor_extended_dataset_produce_charts.volatility.mc_adjustment_factor = 1
libor_extended_dataset_produce_charts.number_of_sims = 100
libor_extended_dataset_produce_charts.volatility.a =  0.01368861
libor_extended_dataset_produce_charts.volatility.b =  0.07921976
libor_extended_dataset_produce_charts.volatility.c = 0.33920146
libor_extended_dataset_produce_charts.volatility.d = 0.08416935
libor_extended_dataset_produce_charts.volatility.instantiate_arrays()
numeraire_index = 40
libor_extended_dataset_produce_charts.run_projection(numeraire_index, numeraire_index)
raw_sims = libor_extended_dataset_produce_charts.forward_sims
forward_sims = np.delete(raw_sims, (numeraire_index), axis=1)
mean = np.mean(forward_sims)
median = np.median(forward_sims)
upper_quartile = np.percentile(forward_sims, 75)
max = np.max(forward_sims)
numeraire_rate_sims = forward_sims[numeraire_index-1,:,:]
cash_rate = np.zeros((numeraire_index, libor_extended_dataset_produce_charts.number_of_sims))

for i in range(numeraire_index):
    cash_rate[i,:] = forward_sims[i,i,:]

min_cash = np.min(cash_rate)
lower_quartile_cash = np.percentile(cash_rate, 25)
min_numeraire = np.min(numeraire_rate_sims)
lower_numeraire = np.percentile(numeraire_rate_sims, 25)


plt.plot(numeraire_rate_sims)
plt.xticks([
    0,
1	,
2	,
3	,
4	,
5	,
6	,
7	,
8	,
9	,
10	,
11	,
12	,
13	,
14	,
15	,
16	,
17	,
18	,
19	,
20	,
21	,
22	,
23	,
24	,
25	,
26	,
27	,
28	,
29	,
30	,
31	,
32	,
33	,
34	,
35	,
36	,
37	,
38	,
39],
    [
        '0.0',
        '0.5',
        '1.0',
        '1.5',
        '2.0',
        '2.5',
        '3.0',
        '3.5',
        '4.0',
        '4.5',
        '5.0',
        '5.5',
        '6.0',
        '6.5',
        '7.0',
        '7.5',
        '8.0',
        '8.5',
        '9.0',
        '9.5',
        '10.0',
        '10.5',
        '11.0',
        '11.5',
        '12.0',
        '12.5',
        '13.0',
        '13.5',
        '14.0',
        '14.5',
        '15.0',
        '15.5',
        '16.0',
        '16.5',
        '17.0',
        '17.5',
        '18.0',
        '18.5',
        '19.0',
        '19.5'])
plt.show()

plt.plot(cash_rate)
plt.xticks([
    0,
1	,
2	,
3	,
4	,
5	,
6	,
7	,
8	,
9	,
10	,
11	,
12	,
13	,
14	,
15	,
16	,
17	,
18	,
19	,
20	,
21	,
22	,
23	,
24	,
25	,
26	,
27	,
28	,
29	,
30	,
31	,
32	,
33	,
34	,
35	,
36	,
37	,
38	,
39],
    [
        '0.0',
        '0.5',
        '1.0',
        '1.5',
        '2.0',
        '2.5',
        '3.0',
        '3.5',
        '4.0',
        '4.5',
        '5.0',
        '5.5',
        '6.0',
        '6.5',
        '7.0',
        '7.5',
        '8.0',
        '8.5',
        '9.0',
        '9.5',
        '10.0',
        '10.5',
        '11.0',
        '11.5',
        '12.0',
        '12.5',
        '13.0',
        '13.5',
        '14.0',
        '14.5',
        '15.0',
        '15.5',
        '16.0',
        '16.5',
        '17.0',
        '17.5',
        '18.0',
        '18.5',
        '19.0',
        '19.5'])
plt.show()






