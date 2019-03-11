import pandas as pd
import numpy as np

# bootstraps probabilites of survival from CDS spread given a discount curve
class CDSBootstrapping():
    def __init__(self, R, term_increment):
        path = 'Data/cds.csv'
        self.name = 'cds'
        self.imported_cds_spreads = pd.read_csv(path)
        self.R = R
        self.LGD = 1 - R
        self.term_increment = term_increment
        self.length = int(self.imported_cds_spreads.shape[0])
        self.set_interpolated_cds()

    def set_interpolated_cds(self):
        self.interpolated_cds = np.zeros(self.length)
        for i in range(self.length):
            self.interpolated_cds[i] = self.imported_cds_spreads.at[i, self.name]

    def get_probability_of_survival(self, ois_discount_curve, multiplier):
        probability_of_survival = np.zeros(self.length + 1)
        probability_of_survival[0] = 1

        probability_of_survival[1] = self.LGD/(self.LGD + self.term_increment*self.interpolated_cds[0]*multiplier)

        for i in range(2, self.length+1):
            main_term = probability_of_survival[i-1]\
                        *self.LGD/(self.LGD + self.term_increment*self.interpolated_cds[i-1]*multiplier)
            sum = 0
            for j in range(1, i):
                sum += ois_discount_curve[j]*(self.LGD*probability_of_survival[j-1] -
                                                 (self.LGD + self.term_increment*self.interpolated_cds[i-1]
                                                  *multiplier)*probability_of_survival[j])
            bottom = ois_discount_curve[i]*(self.LGD + self.term_increment*self.interpolated_cds[i-1]*multiplier)
            probability_of_survival[i] = main_term + sum/bottom
        return probability_of_survival

