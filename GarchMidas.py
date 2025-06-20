import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

from scipy.optimize import basinhopping
from scipy.optimize import minimize
from scipy.optimize import Bounds
import scipy.stats as ss

import statsmodels.formula.api as smf
import numdifftools as ndt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

class GarchMidas:

    def __init__(self, des: str, ret: pd.Series, macro={},
                 samplestart=dt.date(year=1900, month=1, day=1),
                 sampleend=dt.date.today(),
                 rolling_vol=20):

        self.des = des                  # general description of the model
        self.ret = ret                  # assumes daily return centered
        self.macro = macro              # a dictionary of tuples (pd.Series,freq,des,filter type, lags)
        self.__lags = []                  # faster for LL functions
        self.__filters = []               # faster for LL functions can be 'MAV', 'MIDAS' or 'Constrained'
        for macro_index in self.macro:
            self.__lags.append(self.macro[macro_index][4])
            self.__filters.append(self.macro[macro_index][3])

        self.samplestart = samplestart
        self.sampleend = sampleend
        self.rolling_vol = rolling_vol

        self.__macro_scalings = []          # contains scaling factors for macro factors to make sure that the thetas well-behaved
        self.__macro_means = []
        self.__macro_theta_guesses = []     # contains the guesses for thetas ahead of optimisation
        self.__MAVs = []                    # contains the rolling MAVs for each macro time series (after re-scale)
        self.__lagmatrices = []             # contains the lag matrices for each macro time series (after re-scale)
        self.__filters_params = []          # a list of tuples containing (K) or (w2) or (w1,w2) depending on filter type
        self.__filters_df = []                 # contains the orignal series, MAVs and filtered macro time series (optimal parameters)
        self.__weights_df = []              # contains the weights profiles for the optimal perameters
        self.macro_full_MAVs = pd.DataFrame()  # Do I need that????

        self.ret_sample = pd.Series()       # pd.Series, daily returns that will be used for fitting
        self.macro_sample = pd.DataFrame()    # DataFrame in sample for all macro data
        self.macro_sample_MAVs = pd.DataFrame()
        self.__ret_sample_array = np.array([], dtype=np.float64) # faster for calculations

        self.ret_full = pd.Series()  # pd.Series, daily returns for full valid data set
        self.macro_full = pd.DataFrame()  # DataFrame for all macro data (full valid data set)
        self.__ret_full_array = np.array([], dtype=np.float64) # faster for calculations

        self.__GARCH11_params = np.array([], dtype=np.float64)    # contains 5 parameters mu, alpha, beta, gamma and m
        self.GARCH11_params_full = pd.DataFrame()                 # contains standard errors and t-stats
        self.GARCH11_output = pd.DataFrame()                      # contains the time series result of the GARCH11 asymmetric fitting
        self.__GARCH11_output_test = pd.DataFrame()               # this is used to calculate diagnostics
        self.GARCH11_diagnostics = pd.DataFrame()                 # dataframe containing the diagnostics

        self.__GARCHMIDAS_Constrained_params = np.array([], dtype=np.float64)   # to be used in calculations

        self.__GARCHMIDAS_params = np.array([], dtype=np.float64)   # to be used in calculations
        self.GARCHMIDAS_params_full = pd.DataFrame()              # results with standard errors and t-stats
        self.GARCHMIDAS_output = pd.DataFrame()                   # time series of the fitted model
        self.__GARCHMIDAS_output_test = pd.DataFrame()              # this is only to run tests
        self.GARCHMIDAS_diagnostics = pd.DataFrame()              # diagnostics (in sample)

        self.__latest_data = dt.date.today()
        self.__final_start_sample = dt.date.today()
        self.__final_end_sample = dt.date.today()

    def __prepare_macro_series(self):
        self.__macro_scalings = []
        self.__macro_means = []
        self.__macro_theta_guesses = []
        self.__MAVs = []

        print('Processing macro data series...')

        aux_ret = self.ret.to_frame()
        for ts in self.macro:
            K = self.macro[ts][4]
            aux_df = aux_ret.join(self.macro[ts][0], how='left').ffill()
            aux_df.columns = ['target', 'macro_variable']

            self.__macro_means.append(aux_df['macro_variable'].mean())

            aux_df['log_rolling_var'] = 2 * np.log(aux_df['target'].rolling(self.rolling_vol).std())
            aux_df['macro_variable_centered'] = aux_df['macro_variable'] - aux_df['macro_variable'].mean()
            if K == 1:
                aux_df['macro_variable_centered_MAV'] = aux_df['macro_variable_centered']
            else:
                aux_df['macro_variable_centered_MAV'] = aux_df['macro_variable_centered'].rolling(K).mean()
            aux_df = aux_df.dropna()

            model = smf.ols(formula='log_rolling_var ~ macro_variable_centered_MAV', data=aux_df)
            res = model.fit()

            # all series are scaled so that the guess is theta = 1
            sc = res.params.iloc[1]
            my_guess = 1
            self.__macro_scalings.append(sc)
            self.__macro_theta_guesses.append(my_guess)

            aux_df['macro_variable_centered_MAV_scaled'] = sc * aux_df['macro_variable_centered_MAV']
            aux_MAV = aux_df['macro_variable_centered_MAV_scaled'].shift(1).to_frame()
            aux_MAV.columns = [ts + '_MAV']
            self.__MAVs.append(aux_MAV)  # we lag the MAV by 1 period to make sure data is available (delayed publication date)

    @staticmethod
    def __calculate_lags(input_ts: pd.Series, nblags: int) -> pd.DataFrame:
        """
        For a given Series, returns a DataFrame with all lagged Series
        NB: NaN are dropped
        """
        mydf = input_ts.to_frame()
        target_name = mydf.columns[0]
        for i in np.arange(1, nblags + 1, step=1):
            mydf[target_name + ' lag' + str(i)] = mydf[target_name].shift(i)

        lag_names = list()
        for name in mydf.columns:
            if name.find('lag') >= 0:
                lag_names.append(name)

        result = mydf[lag_names]
        result = result.dropna()
        return result

    def __calculatelagmatrices(self):
        """
        For each macro time series in the list, it works a DataFrame with lagged series and put them in a list
        """
        print('Calculating lagged macro data series...')

        for ts, sc, local_m in zip(self.macro, self.__macro_scalings, self.__macro_means):
            ts_lag = self.macro[ts][4]
            self.__lagmatrices.append(self.__calculate_lags(sc * (self.macro[ts][0] - local_m), ts_lag))

    @staticmethod
    def __getendnextmonth(currentdate: dt.date) -> dt.date:
        onemonth = relativedelta(months=1)
        return currentdate + onemonth

    @staticmethod
    def __getendnextquarter(currentdate: dt.date) -> dt.date:
        threemonths = relativedelta(months=3)
        return currentdate + threemonths

    def __compute_key_dates(self):
        self.__latest_data = self.ret_full.index[len(self.ret_full) - 1]
        self.__final_start_sample = self.ret_sample.index[0]
        self.__final_end_sample = self.ret_sample.index[len(self.ret_sample) - 1]

    def __derivesample(self):
        print('Splitting sample...')

        """This first part if for the SAMPLE"""

        min_date = self.samplestart
        max_date = self.sampleend

        # This is for the target which has the highest frequency
        if self.ret.index[0] > min_date:
            min_date = self.ret.index[0]
        if self.ret.index[len(self.ret.index)-1] < max_date:
            max_date = self.ret.index[len(self.ret.index)-1]

        #For each macro series, we make sure the sample is covered
        for ts, ts_lags in zip(self.macro, self.__lagmatrices):

            ts_last_date = ts_lags.index[len(ts_lags.index) - 1]

            if self.macro[ts][1] == 'M':
                ts_covered = self.__getendnextmonth(ts_last_date)
            else:
                ts_covered = self.__getendnextquarter(ts_last_date)

            if ts_covered < max_date:
                max_date = ts_covered

        self.ret_sample = self.ret.loc[min_date:max_date]

        # The sample of the return time series is used as a reference
        auxdf = self.ret_sample.to_frame()

        # Generates sample for the MAVs. Only needed for analysis
        auxdfMAVs = auxdf.join(self.__MAVs, how='left').ffill()
        auxdfMAVs = auxdfMAVs.dropna()
        self.macro_sample_MAVs = auxdfMAVs.drop(auxdfMAVs.columns[0], axis=1)

        # Generate
        auxdf = auxdf.join(self.__lagmatrices, how='left').ffill()
        auxdf = auxdf.dropna()
        self.ret_sample = auxdf[auxdf.columns[0]]
        self.macro_sample = auxdf.drop(auxdf.columns[0], axis=1)

        # np.arrays are better for ll functions
        self.__ret_sample_array = self.ret_sample.values

        """This part is for the FULL data set"""

        min_full = self.ret.index[0]
        max_full = self.ret.index[len(self.ret.index)-1]

        # For each macro series, we make sure the sample is covered
        for ts, ts_lags in zip(self.macro, self.__lagmatrices):

            ts_last_date = ts_lags.index[len(ts_lags.index) - 1]

            if self.macro[ts][1] == 'M':
                ts_covered = self.__getendnextmonth(ts_last_date)
            else:
                ts_covered = self.__getendnextquarter(ts_last_date)

            if ts_covered < max_full:
                max_full = ts_covered

        self.ret_full = self.ret.loc[min_full:max_full]

        # The retrun time series is used as a reference
        auxdf = self.ret_full.to_frame()

        # Generates the dataframe with MAVs for the full data. For analysis only
        auxdfMAVs = auxdf.join(self.__MAVs, how='left').ffill()
        auxdfMAVs = auxdfMAVs.dropna()
        self.macro_full_MAVs = auxdfMAVs.drop(auxdfMAVs.columns[0], axis=1)

        # Generates the dataframe with all lags for the full data
        auxdf = auxdf.join(self.__lagmatrices, how='left').ffill()
        auxdf = auxdf.dropna()

        self.ret_full = auxdf[auxdf.columns[0]]
        self.macro_full = auxdf.drop(auxdf.columns[0], axis=1)

        # np.arrays are better for ll functions
        self.__ret_full_array = self.ret_full.values

        self.__compute_key_dates()

    def __switch_to_constrained(self):
        for i, a in enumerate(self.__filters):
            if a =='MIDAS':
                self.__filters[i] ='Constrained'

    def __restore_filters(self):
        self.__filters = []
        for macro_index in self.macro:
            self.__filters.append(self.macro[macro_index][3])

    @staticmethod
    def __FullMIDASWeight(w1: float, w2: float, K: int) -> float:
        """Function to calculate the beta weights

        Parameters
        ----------
        w1 : float
            omega 1
        w2 : float
            omega 2
        K : int
            midas lags in whatever unit the macro data is (month, quarter)
        Returns
        -------
        np.ndarray
            vector containing the weight at each lag K
        """
        coeff_array = np.empty([K], dtype=np.float64)

        for k in np.arange(1, K, step=1):
            coeff_array[k-1] = ((k / K) ** (w1 - 1)) * ((1 - (k / K)) ** (w2 - 1))

        # special case for k=K
        coeff_array[K-1] = 0

        return_array = coeff_array / np.sum(coeff_array)

        return return_array

    @staticmethod
    def __ConstrainedMIDASWeight(w2: float, K: int) -> float:
        """Function to calculate the MIDAS weights in the "Constrained" case

        Parameters
        ----------
        w2 : float
            omega 2
        K : int
            midas lags in whatever unit the macro data is (month, quarter)
        Returns
        -------
        np.ndarray
            vector containing the weight at each lag K
        """
        coeff_array = np.empty([K], dtype=np.float64)

        for k in np.arange(1, K, step=1):
            coeff_array[k - 1] = (1 - (k / K)) ** (w2 - 1)

        # special case for k=K
        coeff_array[K - 1] = 0

        return_array = coeff_array / np.sum(coeff_array)

        return return_array

    @staticmethod
    def __MAVWeight(K: int) -> float:
        """Function to calculate the MAV weights

        Parameters
        ----------
        K : int
            midas lags in whatever unit the macro data is (month, quarter)
        Returns
        -------
        np.ndarray
            vector containing the weight at each lag K
        NB: the weights are constant and equal to 1/K...
        """

        return_array = (1/K) * np.ones([K])

        return return_array

    @staticmethod
    def __getasymindicator(x: float) -> float:
        if x < 0:
            return 1
        else:
            return 0

    def __objfunction_GARCH11_asymmetric(self, p: np.ndarray) -> float:
        """
        :param p: the parameters of the system that we are trying to calibrate
        The parameters are in this order alpha, beta, gamma  (GARCH (1,1)) and tau

        :return: Log Likelihood * (-1) of the sample for the parameters p
        NB: constant terms are excluded as well as the 0.5 factor
        """

        mu = p[0]
        alpha = p[1]
        beta = p[2]
        gamma = p[3]
        m = p[4]

        #Calculate tau
        if (m > 709) or (m < -100):
            return 9e15
        else:
            tau = np.exp(m)    # simply a scalar ... ensure tau is >0...

            # CALCULATE Series linked to returns
            shocks_squared = (self.__ret_sample_array - mu) ** 2
            asym_indicator_function = np.where((self.__ret_sample_array - mu) < 0, 1, 0)

            # CALCULATE g
            fixed_term = (1-alpha-beta-0.5*gamma) * np.ones(shocks_squared.shape[0])
            shocks_term = (alpha + gamma * asym_indicator_function) * shocks_squared
            shocks_term_scaled = shocks_term / tau
            g = np.zeros(shocks_squared.shape[0])
            #autroregressive term
            g[0] = 1   # assumes the first value is 1
            for i in (range(g.shape[0]-1)):
                g[i+1] = fixed_term[i+1] + shocks_term_scaled[i] + beta*g[i]

            # STEP 3 = Calculate Final Likelihood
            variance = g * tau
            if (variance < 0).any():
                return 9e15
            else:
                log_variance = np.log(variance)
                squares = shocks_squared / variance
                obj_fun = np.sum(log_variance + squares)

                if np.isnan(obj_fun) or np.isinf(obj_fun): obj_fun = 9e15

                return obj_fun

    def __objfunction_GARCHMIDAS_omegas_only_asymmetric(self, p: np.ndarray) -> float:
        """
        :param p: the parameters of the system that we are trying to calibrate
        are w1 and w2 (if needed).
        The following parameters are fixed:
        mu,alpha, beta, gamma  (GARCH (1,1)) and m and (theta_i)s
        We assume they are stored in self.__GARCHMIDAS_Constrained_params
        They are the result of a "constrained" optimisation (only w2s)

        :return: Log Likelihood * (-1) of the sample for the parameters p
        NB: constant terms are excluded as well as the 0.5 factor
        """

        mu = self.__GARCHMIDAS_Constrained_params[0]
        alpha = self.__GARCHMIDAS_Constrained_params[1]
        beta = self.__GARCHMIDAS_Constrained_params[2]
        gamma = self.__GARCHMIDAS_Constrained_params[3]
        m = self.__GARCHMIDAS_Constrained_params[4]

        # STEP 1 - CALCULATE TAU

        # Calculate smoothing coefficients for macro series
        all_arrays = list()
        running_parameter_constrained_index=5
        running_parameter_index=0
        for i in np.arange(0, len(self.macro), step=1):
            aux_theta = self.__GARCHMIDAS_Constrained_params[running_parameter_constrained_index]
            if self.__filters[i] == 'MAV':
                aux = aux_theta * self.__MAVWeight(self.__lags[i])
                running_parameter_constrained_index = running_parameter_constrained_index+1
            elif self.__filters[i] == 'MIDAS':
                aux = aux_theta * self.__FullMIDASWeight(p[running_parameter_index],
                                                                          p[running_parameter_index+1],
                                                                          self.__lags[i])
                running_parameter_constrained_index = running_parameter_constrained_index + 2
                running_parameter_index = running_parameter_index + 2
            else:
                aux = aux_theta * self.__ConstrainedMIDASWeight(p[running_parameter_index],
                                                                                 self.__lags[i])
                running_parameter_constrained_index = running_parameter_constrained_index + 2
                running_parameter_index = running_parameter_index + 1
            all_arrays.append(aux)

        all_phis = np.concatenate(all_arrays)

        #Calulate tau
        log_tau = m + self.macro_sample @ all_phis  # careful this returns a Series
        log_tau_array = log_tau.values  # back to a np.array
        # stop here if tau is going crazy...
        if np.isnan(log_tau_array).any() or np.isinf(log_tau_array).any():
            return 9e15
        else:
            if (log_tau_array > 709).any() or (log_tau_array < -100).any():
                return 9e15
            else:
                tau = np.exp(log_tau_array)

                # STEP 2 = CALCULATE Series linked to returns
                shocks_squared = (self.__ret_sample_array - mu) ** 2
                asym_indicator_function = np.where((self.__ret_sample_array - mu) < 0, 1, 0)

                # STEP 3 = CALCULATE g
                fixed_term = (1-alpha-beta-0.5*gamma) * np.ones(tau.shape[0])
                shocks_term = (alpha + gamma * asym_indicator_function) * shocks_squared
                shocks_term_scaled = shocks_term / tau
                g = np.zeros(tau.shape[0])
                #autroregressive term
                g[0] = 1   # assumes the first value is 1
                for i in (range(g.shape[0]-1)):
                    g[i+1] = fixed_term[i+1] + shocks_term_scaled[i] + beta*g[i]

                # STEP 4 = Calculate Final Likelihood
                variance = g * tau
                if (variance < 0).any():
                    return 9e15
                else:
                    log_variance = np.log(variance)
                    squares = shocks_squared / variance
                    obj_fun = np.sum(log_variance + squares)

                    if np.isnan(obj_fun) or np.isinf(obj_fun): obj_fun = 9e15

                    return obj_fun

    def __objfunction_GARCH_MixedFilters_asymmetric(self, p: np.ndarray) -> float:
        """
        :param p: the parameters of the system that we are trying to calibrate
        The parameters are in this order alpha, beta, gamma  (GARCH (1,1)) and m
        and then theta_i, w1_i, w2_i where i is an index on macro series
        depending on filter type (MAV, MIDAS, Constrained) parameters do vary

        :return: Log Likelihood * (-1) of the sample for the parameters p
        NB: constant terms are excluded as well as the 0.5 factor
        """

        mu = p[0]
        alpha = p[1]
        beta = p[2]
        gamma = p[3]
        m = p[4]

        # STEP 1 - CALCULATE TAU

        # Calculate smoothing coefficients for macro series
        all_arrays = list()
        running_parameter_index = 5
        for i in np.arange(0, len(self.macro),step=1):
            if self.__filters[i] == 'MAV':
                aux = p[running_parameter_index] * self.__MAVWeight(self.__lags[i])
                running_parameter_index=running_parameter_index+1
            elif self.__filters[i] == 'MIDAS':
                aux = p[running_parameter_index] * self.__FullMIDASWeight(p[running_parameter_index+1],
                                                                          p[running_parameter_index+2],
                                                                          self.__lags[i])
                running_parameter_index = running_parameter_index + 3
            else:
                aux = p[running_parameter_index] * self.__ConstrainedMIDASWeight(p[running_parameter_index+1],
                                                                                 self.__lags[i])
                running_parameter_index = running_parameter_index + 2
            all_arrays.append(aux)

        all_phis = np.concatenate(all_arrays)

        #Calulate tau
        log_tau = m + self.macro_sample @ all_phis  # careful this returns a Series
        log_tau_array = log_tau.values  # back to a np.array
        # stop here if tau is going crazy...
        if np.isnan(log_tau_array).any() or np.isinf(log_tau_array).any():
            return 9e15
        else:
            if (log_tau_array > 709).any() or (log_tau_array < -100).any():
                return 9e15
            else:
                tau = np.exp(log_tau_array)

                # STEP 2 = CALCULATE Series linked to returns
                shocks_squared = (self.__ret_sample_array - mu) ** 2
                asym_indicator_function = np.where((self.__ret_sample_array - mu) < 0, 1, 0)

                # STEP 3 = CALCULATE g
                fixed_term = (1-alpha-beta-0.5*gamma) * np.ones(tau.shape[0])
                shocks_term = (alpha + gamma * asym_indicator_function) * shocks_squared
                shocks_term_scaled = shocks_term / tau
                g = np.zeros(tau.shape[0])
                #autroregressive term
                g[0] = 1   # assumes the first value is 1
                for i in (range(g.shape[0]-1)):
                    g[i+1] = fixed_term[i+1] + shocks_term_scaled[i] + beta*g[i]

                # STEP 4 = Calculate Final Likelihood
                variance = g * tau
                if (variance < 0).any():
                    return 9e15
                else:
                    log_variance = np.log(variance)
                    squares = shocks_squared / variance
                    obj_fun = np.sum(log_variance + squares)

                    if np.isnan(obj_fun) or np.isinf(obj_fun): obj_fun = 9e15

                    return obj_fun

    def __full_negative_ll(self, p: np.ndarray, model='GARCHMIDAS') -> float:
        """
        For the optimisation we use a simplified function taking out constant terms and factors.
        This function restores the constants to get the exact -log(likelihood)
        This is used for standard errors, t-stats and BIC
        """
        BigN = len(self.ret_sample)
        const_term = BigN * np.log(2 * np.pi)
        if model == 'GARCHMIDAS':
            return 0.5*(const_term + self.__objfunction_GARCH_MixedFilters_asymmetric(p))
        else:
            return 0.5*(const_term + self.__objfunction_GARCH11_asymmetric(p))

    def __sample_fit_GARCH11_asymmetric(self, scaling=260,realisedwindow=260):
        mu = self.__GARCH11_params[0]
        alpha = self.__GARCH11_params[1]
        beta = self.__GARCH11_params[2]
        gamma = self.__GARCH11_params[3]
        m = self.__GARCH11_params[4]

        self.GARCH11_output = self.ret_full.to_frame()
        self.GARCH11_output.columns=['return']

        # Calulate avarage
        tau = np.exp(m)

        self.__GARCH11_output_test['log tau'] = m * np.ones(len(self.__ret_full_array))
        self.__GARCH11_output_test.index = self.GARCH11_output.index
        self.__GARCH11_output_test['tau'] = tau
        self.GARCH11_output['average vol'] = np.sqrt(tau * scaling)
        self.GARCH11_output['realised vol'] = self.GARCH11_output['return'].rolling(realisedwindow).std()
        self.GARCH11_output['realised vol'] = self.GARCH11_output['realised vol'] * np.sqrt(scaling)

        # CALCULATE Series linked to returns
        shocks_squared = (self.__ret_full_array - mu) ** 2
        asym_indicator_function = np.where((self.__ret_full_array - mu) < 0, 1, 0)

        # CALCULATE g
        fixed_term = (1 - alpha - beta - 0.5 * gamma) * np.ones(shocks_squared.shape[0])
        shocks_term = (alpha + gamma * asym_indicator_function) * shocks_squared
        shocks_term_scaled = shocks_term / tau
        g = np.zeros(shocks_squared.shape[0])
        # autroregressive term
        g[0] = 1  # assumes the first value is 1
        for i in (range(g.shape[0] - 1)):
            g[i + 1] = fixed_term[i + 1] + shocks_term_scaled[i] + beta * g[i]

        # STEP 3 = Calculate Final Vol
        vol = np.sqrt(g * tau * scaling)
        self.__GARCH11_output_test['log g'] = np.log(g)
        self.__GARCH11_output_test['g'] = g
        self.GARCH11_output['full vol'] = vol

    def simplechart_GARCH11(self, showsample=True, showrealised=True, showfull=True):
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        if showfull:
            ax.plot(self.GARCH11_output['full vol'],'r')
        if showrealised:
            ax.plot(self.GARCH11_output['realised vol'], 'k', label='Realised vol')
        ax.plot(self.GARCH11_output['average vol'], 'b', label=f"average={self.GARCH11_output['full vol'].iloc[0]: .2%}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        if showsample:
            ax.axvspan(self.ret_sample.index[0], self.ret_sample.index[len(self.ret_sample)-1], alpha=0.2,label='In sample')
        plt.title(self.des + ': GARCH(1,1) asymmetric')
        plt.legend()
        plt.show()

    def __guess_GARCH11(self):
        """
        :return: Guesses for the following parameters in that order
        mu
        alpha
        beta
        gamma
        m
        """
        #return [0.01, 0.5, 0.1, -0.1]
        return [0.01, 0.01, 0.7, 0.05, -10]

    def __bounds_GARCH11(self):
        lb_res = [-0.05, 0, 0, 0, -50]
        ub_res = [0.05, 1, 1, 1, 10]

        return (lb_res, ub_res)

    def fit_GARCH11(self, sample=True):

        if sample==False:
            self.__prepare_macro_series()
            self.__calculatelagmatrices()
            self.__derivesample()

        print('Fitting standard GARCH(1,1) asymmetric model with constant average (no macro variables)...')

        # Guess
        x0 = self.__guess_GARCH11()

        #Bounds
        lb_res, ub_res = self.__bounds_GARCH11()
        MyBounds = Bounds(lb_res, ub_res, keep_feasible=True)

        #Optimization parameters
        opt_params = {'bounds': MyBounds, \
                      'method': "L-BFGS-B"
                      }

        """result = basinhopping(self.__objfunction_GARCH11_asymmetric, x0, \
                                minimizer_kwargs=opt_params, \
                                niter=niter)
        """

        result = minimize(self.__objfunction_GARCH11_asymmetric, x0, \
                            method="L-BFGS-B", \
                            bounds=MyBounds)

        self.__GARCH11_params = result.x
        self.GARCH11_params_full = pd.DataFrame(result.x.T, \
                                                   index=self.__derive_params_names_GARCH11(), \
                                                   columns=['coeffs'])
        self.__sample_fit_GARCH11_asymmetric()
        self.diagnostics('GARCH11')

    def __sample_fit_GARCHMIDAS(self, scaling=260, realisedwindow=260):
        mu = self.__GARCHMIDAS_params[0]
        alpha = self.__GARCHMIDAS_params[1]
        beta = self.__GARCHMIDAS_params[2]
        gamma = self.__GARCHMIDAS_params[3]
        m = self.__GARCHMIDAS_params[4]

        self.GARCHMIDAS_output = self.ret_full.to_frame()
        self.GARCHMIDAS_output.columns=['return']
        self.GARCHMIDAS_output['realised vol'] = self.GARCHMIDAS_output['return'].rolling(realisedwindow).std()
        self.GARCHMIDAS_output['realised vol'] = self.GARCHMIDAS_output['realised vol'] * np.sqrt(scaling)

        # STEP 1 - CALCULATE TAU

        # Calculate smoothing coefficients for macro series
        all_arrays = list()
        running_parameter_index = 5
        for i in np.arange(0, len(self.macro), step=1):
            if self.__filters[i] == 'MAV':
                aux = self.__GARCHMIDAS_params[running_parameter_index] * self.__MAVWeight(self.__lags[i])
                running_parameter_index = running_parameter_index + 1
            elif self.__filters[i] == 'MIDAS':
                aux = self.__GARCHMIDAS_params[running_parameter_index] * self.__FullMIDASWeight(self.__GARCHMIDAS_params[running_parameter_index+1],
                                                                                                 self.__GARCHMIDAS_params[running_parameter_index+2],
                                                                                                 self.__lags[i])
                running_parameter_index = running_parameter_index + 3
            else:
                aux = self.__GARCHMIDAS_params[running_parameter_index] * self.__ConstrainedMIDASWeight(self.__GARCHMIDAS_params[running_parameter_index+1],
                                                                                                        self.__lags[i])
                running_parameter_index = running_parameter_index + 2
            all_arrays.append(aux)

        all_phis = np.concatenate(all_arrays)

        # Calulate tau
        log_tau = m + self.macro_full @ all_phis  # careful this returns a Series
        tau = np.exp(log_tau.values)  # back to a np.array

        self.__GARCHMIDAS_output_test['log tau'] = log_tau
        self.__GARCHMIDAS_output_test['tau'] = tau
        self.GARCHMIDAS_output['macro vol'] = np.sqrt(tau * scaling)

        # STEP 2 = CALCULATE Series linked to returns
        shocks_squared = (self.__ret_full_array - mu) ** 2
        asym_indicator_function = np.where((self.__ret_full_array - mu) < 0, 1, 0)

        # STEP 3 = CALCULATE g
        fixed_term = (1 - alpha - beta - 0.5 * gamma) * np.ones(tau.shape[0])
        shocks_term = (alpha + gamma * asym_indicator_function) * shocks_squared
        shocks_term_scaled = shocks_term / tau
        g = np.zeros(tau.shape[0])
        # autroregressive term
        g[0] = 1  # assumes the first value is 1
        for i in (range(g.shape[0] - 1)):
            g[i + 1] = fixed_term[i + 1] + shocks_term_scaled[i] + beta * g[i]

        # STEP 4 = Calculate Final Vol
        vol = np.sqrt(g * tau * scaling)

        self.__GARCHMIDAS_output_test['log g'] = np.log(g)
        self.__GARCHMIDAS_output_test['g'] = g

        self.GARCHMIDAS_output['full vol'] = vol

    def simplechart_GARCHMIDAS(self, showsample=True, showrealised=True, showfull=True,minvol=0, maxvol=0,):
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        if showfull:
            ax.plot(self.GARCHMIDAS_output['full vol'], 'r',label='Full model')
        if showrealised:
            ax.plot(self.GARCHMIDAS_output['realised vol'], 'k', label='Realised vol')

        ax.plot(self.GARCHMIDAS_output['macro vol'], 'b', label='Macro component of vol')

        if showsample:
            ax.axvspan(self.ret_sample.index[0], self.ret_sample.index[len(self.ret_sample)-1], alpha=0.2,label='In sample')

        if (minvol!=0):
            if (maxvol!=0):
                ax.set_ylim([minvol,maxvol])
            else:
                ax.set_ylim([minvol, None])
        else:
            ax.set_ylim([None, maxvol])

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.title(self.des + ' - GARCH-MIDAS')
        plt.legend()
        plt.show()

    def __extract_filters(self,target_freq, target_K):
        self.__filters_df = []
        self.__filters_params = []
        self.__weights_df = []

        basic_nb_parameters = self.__GARCHMIDAS_nbparams('GARCH11')
        nb_parameters = basic_nb_parameters

        aux_all_lags = np.arange(1,target_K+1,step=1)

        for i, ts_macro in enumerate(self.macro):
            K = self.macro[ts_macro][4]
            freq = self.macro[ts_macro][1]

            aux_df = self.macro[ts_macro][0].to_frame()
            aux_df.columns = [ts_macro]
            aux_df[ts_macro + '_MAV'] = aux_df[ts_macro].rolling(K).mean()
            aux_df[ts_macro + '_MAV'] = aux_df[ts_macro + '_MAV'].shift(1)   # makes sure we take previous period MAV
            filter_type = self.macro[ts_macro][3]

            sc = self.__macro_scalings[i]
            m = self.__macro_means[i]

            aux_df_lags = pd.DataFrame(aux_all_lags)
            aux_df_lags.columns = ['lags']
            aux_df_lags['MAV'] = self.__resample_extend_weights(self.__MAVWeight(K),freq,target_freq,target_K)

            if filter_type == 'MAV':
                self.__filters_params.append((K))
                nb_parameters += 1
            elif filter_type == 'Constrained':
                w2 = self.__GARCHMIDAS_params[nb_parameters+1]
                self.__filters_params.append((w2))
                aux_weights = self.__ConstrainedMIDASWeight(w2,K)
                aux_filter = self.__lagmatrices[i] @ aux_weights
                aux_df[ts_macro + '_filter'] = m + (aux_filter / sc)
                aux_df_lags['filter'] = self.__resample_extend_weights(aux_weights,freq,target_freq,target_K)
                nb_parameters += 2
            else:
                w1 = self.__GARCHMIDAS_params[nb_parameters + 1]
                w2 = self.__GARCHMIDAS_params[nb_parameters + 2]
                self.__filters_params.append((w1, w2))
                aux_weights = self.__FullMIDASWeight(w1, w2, K)
                aux_filter = self.__lagmatrices[i] @ aux_weights
                aux_df[ts_macro + '_filter'] = m + (aux_filter / sc)
                aux_df_lags['filter'] = self.__resample_extend_weights(aux_weights, freq, target_freq, target_K)
                nb_parameters += 3

            self.__filters_df.append(aux_df)
            self.__weights_df.append(aux_df_lags)

    def __compute_lags_weight_profiles(self):
        min_lag_freq = 'Q'
        for ts_macro in self.macro:
            if self.macro[ts_macro][1] == 'M':
                min_lag_freq = 'M'

        max_lag = 1
        for ts_macro in self.macro:
            if (self.macro[ts_macro][1] == 'Q') and (min_lag_freq == 'M'):
                aux_mult = 3
            else:
                aux_mult=1

            if (aux_mult*self.macro[ts_macro][4]) > max_lag:
                max_lag = aux_mult*self.macro[ts_macro][4]

        return min_lag_freq, max_lag

    def __resample_extend_weights(self, weights, freq, target_freq, target_K):
        if (target_freq != freq):  # this means we have quarterly and we need monthly
            aux_weights = np.repeat(weights,3)
        else:
            aux_weights = weights

        K= np.shape(aux_weights)[0]
        if K<target_K:
            delta_K = target_K - K
            aux_to_be_stacked = np.zeros(delta_K)
            aux_weights = np.concatenate([aux_weights,aux_to_be_stacked])

        return aux_weights.tolist()

    def __compute_filters(self):
        lag_freq, max_lag = self.__compute_lags_weight_profiles()
        self.__extract_filters(lag_freq, max_lag)

    def chart_filter(self):
        nb_macro_variables = len(self.macro)

        fig, axes = plt.subplots(nb_macro_variables, 2, sharex='col')

        for i, ts_macro in enumerate(self.macro):
            aux_df = self.__filters_df[i]
            aux_weights = self.__weights_df[i]
            aux_filter_type = self.macro[ts_macro][3]

            if nb_macro_variables == 1:
                ax=axes[0]
            else:
                ax = axes[i, 0]
            ax.set_title(ts_macro, fontsize=8)
            ax.plot(aux_df[ts_macro + '_MAV'],label='MAV',color='b')
            if aux_filter_type!='MAV':
                ax.plot(aux_df[ts_macro + '_filter'], label='MIDAS Filter', color='r')
            ax.plot(aux_df[ts_macro], label='Raw Series', color='g')
            ax.legend(loc="upper left", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)

            if nb_macro_variables == 1:
                ax = axes[1]
            else:
                ax = axes[i, 1]

            if i==0:
                ax.set_title('Filter weights', size=8)
            if aux_filter_type != 'MAV':
                ax.bar(aux_weights['lags'], 100*aux_weights['filter'], label='MIDAS Filter',fill=False,edgecolor = 'r')
            ax.plot(aux_weights['lags'], 100*aux_weights['MAV'], label='MAV', marker='o', markersize=2, color='b', linestyle='None')
            ax.set_ylim(0,110)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(loc="upper right", fontsize=8)

            if i == (nb_macro_variables-1):
                ax.set_xlabel('Lags', fontsize=8)

        fig.suptitle("Macro Variables - MIDAS")
        plt.show()

    def __derive_params_names_GARCH11(self):
        params_list = ['mu', 'alpha', 'beta', 'gamma', 'm']
        return params_list

    def __derive_params_names_GARCHMIDAS(self):
        params_list = self.__derive_params_names_GARCH11()
        for i in np.arange(1, len(self.macro)+1, step=1):
            if self.__filters[i-1] == 'MAV':
                params_list.append('theta' + str(i))
            elif self.__filters[i-1] == 'MIDAS':
                params_list.append('theta' + str(i))
                params_list.append('w1_' + str(i))
                params_list.append('w2_' + str(i))
            else:
                params_list.append('theta' + str(i))
                params_list.append('w2_' + str(i))

        return params_list

    def __guess_GARCHMIDAS_include_omegas_only(self, p):
        aux_GARCH = self.__GARCHMIDAS_Constrained_params[0:5]

        running_params_constrained = 5
        running_params_omegas = 0
        all_macro_arrays = list()
        for i in np.arange(0,len(self.macro), step=1):
            aux_theta = self.__GARCHMIDAS_Constrained_params[running_params_constrained]
            if self.__filters[i] == 'MAV':
                one_macro_array = np.array([aux_theta])
                running_params_constrained += 1
            elif self.__filters[i] == 'MIDAS':
                w1_guess = p[running_params_omegas]
                w2_guess = p[running_params_omegas+1]
                one_macro_array = np.array([aux_theta, w1_guess, w2_guess])
                running_params_omegas += 2
                running_params_constrained += 2
            else:
                w2_guess = p[running_params_omegas]
                one_macro_array = np.array([aux_theta, w2_guess])
                running_params_omegas += 1
                running_params_constrained += 2
            all_macro_arrays.append(one_macro_array)

        aux_macro = np.concatenate(all_macro_arrays)
        x = np.concatenate((aux_GARCH, aux_macro))

        return x

    def __guess_GARCHMIDAS_omegas_only(self, fixedguess=True):
        w1_guess = 1
        w2_fixed_guess = 20

        all_macro_arrays = list()
        running_params_constrained = 5
        for i in np.arange(0, len(self.macro), step=1):
            if self.__filters[i] == 'MAV':
                running_params_constrained += 1
            else:
                w2_guess = self.__GARCHMIDAS_Constrained_params[running_params_constrained + 1]
                if fixedguess:
                    w2_guess = w2_fixed_guess
                if self.__filters[i] == 'MIDAS':
                    one_macro_array = np.array([w1_guess, w2_guess])
                else:  # 'Constrained'
                    one_macro_array = np.array([w2_guess])
                running_params_constrained += 2
            all_macro_arrays.append(one_macro_array)

        x = np.concatenate(all_macro_arrays)

        return x

    def __guess_GARCHMIDAS_Mixed(self, useGARCH11=False):
        w1_guess = 1
        w2_guess = 20

        all_macro_arrays = list()
        for i in np.arange(0, len(self.macro), step=1):
            theta = self.__macro_theta_guesses[i]
            if self.__filters[i] == 'MAV':
                one_macro_array = np.array([theta])
            elif self.__filters[i] == 'MIDAS':
                one_macro_array = np.array([theta, w1_guess, w2_guess])
            else:
                one_macro_array = np.array([theta, w2_guess])
            all_macro_arrays.append(one_macro_array)

        macro_array = np.concatenate(all_macro_arrays)

        if useGARCH11==True:
            GARCHarray = self.__GARCH11_params
        else:
            GARCHarray = self.__guess_GARCH11()

        x = np.concatenate((GARCHarray, macro_array))
        return x

    def __bounds_GARCHMIDAS_omegas_only(self):

        # we are tryinh to optimise these two parameters so only "wide" bounds needed
        w2_min = 0
        w2_max = 500
        w1_min = 0.05
        w1_max = 100

        lb_macro_arrays = []
        ub_macro_arrays = []
        for i in np.arange(0, len(self.macro), step=1):
            if self.__filters[i] == 'MIDAS':
                lb_one = np.array([w1_min,w2_min])
                ub_one = np.array([w1_max,w2_max])
            if self.__filters[i] == 'Constrained':
                lb_one = np.array([w2_min])
                ub_one = np.array([w2_max])
            lb_macro_arrays.append(lb_one)
            ub_macro_arrays.append(ub_one)

        lb_res = np.concatenate(lb_macro_arrays)
        ub_res = np.concatenate(ub_macro_arrays)

        return (lb_res, ub_res)

    def __bounds_GARCHMIDAS_Mixed(self, wide=False):
        lb_GARCH , ub_GARCH = self.__bounds_GARCH11()

        theta_min = -5
        theta_max = 5
        w2_min = 0

        if wide:
            w1_min = 0.05
            w1_max = 100
            w2_max = 500
        else:
            w1_min = 0.5
            w1_max = 10
            w2_max = 200

        lb_macro_arrays = []
        ub_macro_arrays = []
        for i in np.arange(0,len(self.macro),step=1):
            if self.__filters[i] == 'MAV':
                lb_one = np.array([theta_min])
                ub_one = np.array([theta_max])
            elif self.__filters[i] == 'MIDAS':
                lb_one = np.array([theta_min,w1_min,w2_min])
                ub_one = np.array([theta_max,w1_max,w2_max])
            else:
                lb_one = np.array([theta_min, w2_min])
                ub_one = np.array([theta_max, w2_max])
            lb_macro_arrays.append(lb_one)
            ub_macro_arrays.append(ub_one)

        lb_macro = np.concatenate(lb_macro_arrays)
        ub_macro = np.concatenate(ub_macro_arrays)

        lb_res = np.concatenate((lb_GARCH, lb_macro))
        ub_res = np.concatenate((ub_GARCH, ub_macro))

        return (lb_res, ub_res)

    def __take_step_omegas(self,x):
        s = 1

        running_parameter_index = 0
        for i in np.arange(0, len(self.macro), step=1):
            if self.__filters[i] == 'MIDAS':
                x[running_parameter_index] += np.random.uniform(-0.1 * s, 0.1 * s)  # w1
                x[running_parameter_index + 1] += np.random.uniform(-5 * s, 5 * s)  # w2
                if x[running_parameter_index] < 0:
                    x[running_parameter_index] = 1e-8
                if x[running_parameter_index + 1] <= 1:
                    x[running_parameter_index + 1] = 1 + 1e-8
                running_parameter_index = running_parameter_index + 2
            if self.__filters[i] == 'Constrained':
                x[running_parameter_index] += np.random.uniform(-5 * s, 5 * s)  # w2
                if x[running_parameter_index] <= 1:
                    x[running_parameter_index] = 1 + 1e-8
                running_parameter_index = running_parameter_index + 1

        return x

    def __take_step_mixed(self, x):
        s = 1
        x[0] += np.random.uniform(-0.001*s, 0.001*s)  # mu 1% move
        x[1] += np.random.uniform(-0.02*s, 0.02*s)  # alpha 0.1 step
        x[2] += np.random.uniform(-0.01*s, 0.01*s)  # beta 0.1 step
        x[3] += np.random.uniform(-0.005*s, 0.005*s)  # gamma 0.05 step
        x[4] += np.random.uniform(-1 * s, 1 * s)  # m log 2 step

        running_parameter_index = 5
        for i in np.arange(0, len(self.macro), step=1):
            if self.__filters[i] == 'MAV':
                x[running_parameter_index] += np.random.uniform(-0.1 * s, 0.1 * s)
                if (x[running_parameter_index]*self.__macro_theta_guesses[i]) < 0:
                    if x[running_parameter_index] > 0:
                        x[running_parameter_index] = -1e-8
                    else:
                        x[running_parameter_index] = 1e-8
                running_parameter_index = running_parameter_index + 1
            elif self.__filters[i] == 'MIDAS':
                x[running_parameter_index] += np.random.uniform(-0.1 * s, 0.1 * s)  # theta  0.2 step
                if (x[running_parameter_index]*self.__macro_theta_guesses[i]) < 0:
                    if x[running_parameter_index] > 0:
                        x[running_parameter_index] = -1e-8
                    else:
                        x[running_parameter_index] = 1e-8
                x[running_parameter_index + 1] += np.random.uniform(-0.1 * s, 0.1 * s)  # w1
                x[running_parameter_index + 2] += np.random.uniform(-5 * s, 5 * s)  # w2
                if x[running_parameter_index + 2] <= 1:
                    x[running_parameter_index + 2] = 1 + 1e-8
                running_parameter_index = running_parameter_index + 3
            else:
                x[running_parameter_index] += np.random.uniform(-0.1 * s, 0.1 * s)  # theta  0.2 step
                if (x[running_parameter_index]*self.__macro_theta_guesses[i])<0:
                    if x[running_parameter_index] > 0:
                        x[running_parameter_index] = -1e-8
                    else:
                        x[running_parameter_index] = 1e-8
                x[running_parameter_index + 1] += np.random.uniform(-5 * s, 5 * s)  # w2
                if x[running_parameter_index + 1] <= 1:
                    x[running_parameter_index + 1] = 1 + 1e-8
                running_parameter_index = running_parameter_index + 2

        # Maintain feasibility
        if x[1] <= 1e-8:
            x[1] = 1e-8
        if x[2] >= 1.0:
            x[2] = 1 - 1e-8
        if (x[1]+x[2]+0.5*x[3]>1):
            x[3] = 1e-8

        return x

    @staticmethod
    def __print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))

    def fit_GARCHMIDAS(self, niter=30, method='2steps', niter_omegas=20):
        self.__prepare_macro_series()
        self.__calculatelagmatrices()
        self.__derivesample()

        # First fits the model without macro parameters
        self.fit_GARCH11(sample=True)

        if all(a =='MAV' for a in self.__filters):
            need_2_steps = False
        else:
            need_2_steps = True

        if (method == '2steps') and (need_2_steps):
            print('Two step method: raw constrained optimisation and then specific optimisation on MIDAS parameters... ')

            # FIRST STEP: CONSTRAINED OPTIMISATION
            self.__switch_to_constrained()
            print(f"Running constrained optimisation first (iterations={niter})...")

            # Guess
            x0 = self.__guess_GARCHMIDAS_Mixed(useGARCH11=True)

            # Bounds
            lb_res, ub_res = self.__bounds_GARCHMIDAS_Mixed()
            MyBounds = Bounds(lb_res, ub_res, keep_feasible=True)

            # Optimization parameters
            opt_params = {'bounds': MyBounds, \
                          'method': "L-BFGS-B" \
                          }

            result_one = basinhopping(self.__objfunction_GARCH_MixedFilters_asymmetric, x0, \
                                  minimizer_kwargs=opt_params, \
                                  take_step=self.__take_step_mixed, \
                                  callback=self.__print_fun, \
                                  niter=niter)

            self.__GARCHMIDAS_Constrained_params = result_one.x

            # SECOND STEP: OPTIMISATION ONLY ON FILTERS PARAMETERS
            self.__restore_filters()
            print(f"Optimising MIDAS filters parameters only (iterations={niter_omegas})...")

            #Guess
            w0 = self.__guess_GARCHMIDAS_omegas_only()

            # Bounds
            lb_res, ub_res = self.__bounds_GARCHMIDAS_omegas_only()
            MyBounds = Bounds(lb_res, ub_res, keep_feasible=True)

            # Optimization parameters
            opt_params = {'bounds': MyBounds, \
                          'method': "L-BFGS-B" \
                          }

            result_two = basinhopping(self.__objfunction_GARCHMIDAS_omegas_only_asymmetric, w0, \
                                  minimizer_kwargs=opt_params, \
                                  take_step=self.__take_step_omegas, \
                                  callback=self.__print_fun, \
                                  niter=niter_omegas)

            omegas_guess = result_two.x

            # THIRD STEP: FINAL OPTIMISATION - NO BASIN HOPPING
            print('Final optimisation...')

            # Guess
            x0 = self.__guess_GARCHMIDAS_include_omegas_only(omegas_guess)

            # Bounds
            lb_res, ub_res = self.__bounds_GARCHMIDAS_Mixed(wide=True)
            MyBounds = Bounds(lb_res, ub_res, keep_feasible=True)

            result = minimize(self.__objfunction_GARCH_MixedFilters_asymmetric, x0, \
                              method="L-BFGS-B", \
                              bounds=MyBounds)
        else:

            print(f"Running full optimisation (iterations = {niter})...")

            # Guess
            x0 = self.__guess_GARCHMIDAS_Mixed(useGARCH11=True)

            # Bounds
            lb_res, ub_res = self.__bounds_GARCHMIDAS_Mixed()
            MyBounds = Bounds(lb_res, ub_res, keep_feasible=True)

            # Optimization parameters
            opt_params = {'bounds': MyBounds, \
                          'method': "L-BFGS-B" \
                          }

            result = basinhopping(self.__objfunction_GARCH_MixedFilters_asymmetric, x0, \
                                  minimizer_kwargs=opt_params, \
                                  take_step = self.__take_step_mixed, \
                                  callback=self.__print_fun, \
                                  niter=niter)

        #print(result)

        self.__GARCHMIDAS_params = result.x
        self.GARCHMIDAS_params_full = pd.DataFrame(result.x.T, \
                                                   index=self.__derive_params_names_GARCHMIDAS(), \
                                                   columns=['coeffs'])
        self.__sample_fit_GARCHMIDAS(260)
        self.diagnostics()
        self.__compute_filters()

    def __GARCHMIDAS_nbparams(self, model='GARCHMIDAS', unconstrained=True):
        GARCH11_nbparams = 5   # mu, alpha, beta, gamma, m
        if model == 'GARCHMIDAS':
            MIDAS_nbparams = 0
            for i in np.arange(0,len(self.macro), step=1):
                if self.__filters[i] == 'MAV':
                    MIDAS_nbparams = MIDAS_nbparams + 1
                elif self.__filters[i] == 'MIDAS':
                    MIDAS_nbparams = MIDAS_nbparams + 3
                else:
                    MIDAS_nbparams = MIDAS_nbparams + 2
            return GARCH11_nbparams + MIDAS_nbparams
        else:
            return GARCH11_nbparams

    def __BIC(self,model='GARCHMIDAS'):
        if model=='GARCHMIDAS':
            p = self.__GARCHMIDAS_params
        else:
            p = self.__GARCH11_params

        result_BIC = 2 * self.__full_negative_ll(p, model) + np.log(len(self.ret_sample)) * self.__GARCHMIDAS_nbparams(model)
        result_BIC_norm = result_BIC / (2*len(self.ret_sample))
        return result_BIC, result_BIC_norm

    def __hessian_stats(self, model):
        relative_step = 0.0001
        sample_size = len(self.__ret_sample_array)
        if model == 'GARCHMIDAS':
            p = self.__GARCHMIDAS_params
            if (0 in p):
                print('Model mispecified')
                model_ok = False
            else:
                steps = p * relative_step
                h = ndt.Hessian(self.__objfunction_GARCH_MixedFilters_asymmetric, step=steps, method='central')(p)
                model_ok = True
        else:
            p = self.__GARCH11_params
            if (0 in p):
                print('Model mispecified')
                model_ok = False
            else:
                steps = p * relative_step
                h = ndt.Hessian(self.__objfunction_GARCH11_asymmetric, step=steps, method='central')(p)
                model_ok = True

        if model_ok == True:
            I = 0.5 * h / sample_size # this is required as the constant 0.5 is excluded. I is the info matrix. Sign is ok as obj is (-1) * loglike
            #print("Information Matrix=")
            #print(I)   # Information matrix

            try:
                inv = np.linalg.inv(I)
                #print("Inverse of Information Matrix=")
                #print(inv)
            except ValueError:
                print("Model mispecified")
                model_ok = False

            if model_ok:
                s_vars = np.diag(inv)
                s_errors = np.sqrt(s_vars / sample_size)
                t_stats = np.abs(p / s_errors)
                p_vals = 1 - ss.norm.cdf(t_stats)

        if model_ok == False:
                #s_vars = np.empty(self.__GARCHMIDAS_nbparams(model))*np.nan
                s_errors = np.empty(self.__GARCHMIDAS_nbparams(model))*np.nan
                t_stats = np.empty(self.__GARCHMIDAS_nbparams(model))*np.nan
                p_vals = np.empty(self.__GARCHMIDAS_nbparams(model))*np.nan

        return s_errors, t_stats, p_vals


    def diagnostics(self, model='GARCHMIDAS'):
        variance_test = self.ret_sample.rolling(self.rolling_vol).var()
        variance_test = variance_test.to_frame()
        variance_test.columns = ['variance test']

        if model=='GARCHMIDAS':
            auxdf = self.__GARCHMIDAS_output_test.join(variance_test, how='left')
        else:   # only GARCH11
            auxdf = self.__GARCH11_output_test.join(variance_test, how='left')

        auxdf = auxdf.dropna()
        BigN = len(auxdf)
        auxdf['variance'] = auxdf['tau'] * auxdf['g']
        auxdf['log variance'] = auxdf['log tau'] + auxdf['log g']
        auxdf['var diff absolute'] = np.abs(auxdf['variance test'] - auxdf['variance'])
        auxdf['var diff square'] = (auxdf['variance test'] - auxdf['variance'])**2
        auxdf['var ratio'] = auxdf['variance test'] / auxdf['variance']
        auxdf['log var ratio'] = np.log(auxdf['var ratio'])
        auxdf['qlike elememts'] = auxdf['var ratio'] - auxdf['log var ratio'] -1

        aux_VARX = 100 * auxdf['log tau'].var() / auxdf['log variance'].var()
        aux_MAE = auxdf['var diff absolute'].sum() / BigN
        aux_MSPE = auxdf['var diff square'].sum() / BigN
        aux_QLIKE = auxdf['qlike elememts'].sum() / BigN

        aux_BIC, aux_BIC_norm = self.__BIC(model)

        data = {'VARX': [aux_VARX], 'MAE': [aux_MAE], 'MSPE':[aux_MSPE], 'QLIKE':[aux_QLIKE], 'BIC':[aux_BIC], 'BIC(normalised)':[aux_BIC_norm]}
        if model == 'GARCHMIDAS':
            self.GARCHMIDAS_diagnostics = pd.DataFrame.from_dict(data, orient='index')
            self.GARCHMIDAS_diagnostics.columns=['value']
        else:
            self.GARCH11_diagnostics = pd.DataFrame.from_dict(data, orient='index')
            self.GARCH11_diagnostics.columns = ['value']

        serrors, t_stats, p_values = self.__hessian_stats(model)
        if model == 'GARCHMIDAS':
            self.GARCHMIDAS_params_full['Standard Error'] = serrors.tolist()
            self.GARCHMIDAS_params_full['t-stat'] = t_stats.tolist()
            self.GARCHMIDAS_params_full['p-value'] = p_values.tolist()
        else:
            self.GARCH11_params_full['Standard Error'] = serrors.tolist()
            self.GARCH11_params_full['t-stat'] = t_stats.tolist()
            self.GARCH11_params_full['p-value'] = p_values.tolist()

    def __macro_variables_table(self):
        short_names = []
        frequencies = []
        filter_types = []
        filter_lags = []

        for ts_macro in self.macro:
            short_names.append(ts_macro)
            frequencies.append(self.macro[ts_macro][1])
            filter_types.append(self.macro[ts_macro][3])
            filter_lags.append(self.macro[ts_macro][4])

        aux_df = pd.DataFrame(zip(short_names,frequencies,filter_types,filter_lags),
                              columns=['Short Name','Frequency','Filter','Lags'])
        aux_df.index = np.arange(1,len(self.macro)+1,step=1).tolist()
        return aux_df

    def __str__(self):
        macro_variables_details = self.__macro_variables_table()

        return f"******************************************************** \
               \n*********  GARCH-MIDAS model *************************** \
               \n******************************************************** \
               \n{self.des} \
               \nSample size = {len(self.ret_sample)} \
               \nSample start = {self.__final_start_sample: %d-%b-%Y} \
               \nSample end = {self.__final_end_sample: %d-%b-%Y} \
               \nLatest data point = {self.__latest_data: %d-%b-%Y} \
               \n******************************************************** \
               \n*********  GARCH(1,1) asymmetric - no macro ************ \
               \n******************************************************** \
               \n{self.GARCH11_params_full} \
               \nDiagnostics are calculated using rolling vol {self.rolling_vol} \
               \n{self.GARCH11_diagnostics} \
               \n******************************************************** \
               \n********* GARCH-MIDAS with all macro variables ********* \
               \n******************************************************** \
               \nNumber of macro variables = {len(self.macro)} \
               \n{macro_variables_details} \
               \n{self.GARCHMIDAS_params_full} \
               \nDiagnostics are calculated using rolling vol {self.rolling_vol} \
               \n{self.GARCHMIDAS_diagnostics}"

