import numpy as np
from scipy.optimize import root, minimize
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import norm
from multiprocessing import Pool
import time

from MarketData import *
import Calibration as calib
import Correlation as corr
import MonteCarlo as mc


def display_simulated_forward_rate(simulated_fwd, forwards):

    for i in range(len(forwards)):
        fwd = simulated_fwd[:, i]

        plt.hist(fwd, bins=20)
        plt.axvline(x=np.mean(fwd), color='g')
        plt.axvline(x=forwards[i], color='r')
        plt.grid(True, linestyle='--')
        plt.title('Simulated Swap Rate')
        plt.show()



def swaption_price(annuity, swap_rate, K, sigma, T, model):
    result = 0

    if model == 'log-normal':
        d1 = (np.log(swap_rate / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        result = annuity * (swap_rate * norm.cdf(d1) - K * norm.cdf(d2))

    elif model == 'normal':
        d = (swap_rate - K) / (sigma * np.sqrt(T))

        result = annuity * sigma * np.sqrt(T) * (d * norm.cdf(d) + norm.pdf(d))

    return result


def swaption_vol_calibration(target, annuity, swap_rate, K, T):
    if swap_rate == K:
        sigma = np.sqrt(2 * abs(np.log(swap_rate / K)) / T) + 0.01
    else:
        sigma = np.sqrt(2 * abs(np.log(swap_rate / K)) / T)
    bump_size = 0.001

    swaptn_func = partial(swaption_price, annuity=annuity, swap_rate=swap_rate, K=K, T=T, model='log-normal')

    black_price = swaptn_func(sigma=sigma)
    while abs(black_price - target) > 1e-8:
        black_price_plus = swaptn_func(sigma=sigma + bump_size)
        vega = (black_price_plus - black_price) / bump_size

        sigma = sigma - (black_price - target) / vega
        black_price = swaptn_func(sigma=sigma)

        # print(sigma, '\t', black_price, '\t', abs(black_price - target))

    return sigma

def extract_calibration_vol(matrix):
    nb_row, nb_col = matrix.shape

    v_calib = np.zeros((nb_row, nb_col))
    for i in range(nb_row):
        for j in range(nb_col - i):
            v_calib[i, j] = matrix[i, j]

    return v_calib


if __name__ == '__main__':
    # rho_inf, alpha, beta = Rebonato_three_factor_parameterization(corr_matrix)
    # corr_rebo = Rebonato_three_factor_parameterization(corr_matrix)
    # rho = lambda i, j: rho_inf + (1 - rho_inf) * np.exp(-abs(i - j) * (beta - alpha * (max(i, j) - 1)))
    # rebonato_corr = np.zeros((19, 19))
    #
    # for i in range(19):
    #     for j in range(19):
    #         rebonato_corr[i, j] = rho(i, j)

    # rho_generator = lambda i, j: 0.5 + (1 - 0.5) * np.exp(-0.05 * abs(i - j))
    #
    # corr_matrix = np.zeros((10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         corr_matrix[i, j] = rho_generator(i, j)

    corr_rebo = corr.Rebonato_three_factor_parameterization(corr_matrix)
    # corr_fr = corr.Schoenmakers_and_Coffey_parameterization(corr_matrix, 2)

    rank = 5
    rho_zero, theta = corr.eigenvalue_zeroing(corr_rebo, rank)
    rho_opt = corr.correlation_optimization(corr_rebo, theta.reshape(-1), rank)
    # print(np.round(rho_opt, 3))

    # term_corr = corr.terminal_correlation(11, corr_rebo)
    # print(np.round(term_corr, 3))

    M = len(forwards)
    discount_factors = np.zeros(M)
    for i in range(len(discount_factors)):
        if i == 0:
            discount_factors[i] = 1 / (1 + forwards[i])
        else:
            discount_factors[i] = discount_factors[i - 1] / (1 + forwards[i])

    v_calib = extract_calibration_vol(v_swaptn)
    LMM_GPC = np.zeros((10, 10))
    # v_calib = v_swaptn
    # LMM_GPC = np.zeros((19, 10))

    opt_maturity = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    swap_length = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    s = len(opt_maturity)

    # Cascade Calibration Algorithm
    row_count = 0
    LMM_GPC[0, 0] = v_calib[0, 0]
    for i in range(s):
        for j in range(s - i):
            if i != 0 or j != 0:
                T_alpha, T_beta = opt_maturity[i], swap_length[j]
                start, end = T_alpha - 1, T_alpha + T_beta - 1

                w, F = calib.calc_swap_rate(T_alpha, T_beta, forwards, discount_factors)
                swap_rate = np.sum(w * F)

                sub_sigma = LMM_GPC[row_count:row_count + T_beta, :T_alpha]
                LMM_GPC[j + row_count, i] = calib.cascade_calibration_algorithm(T_alpha, swap_rate, w, F,
                                                                                v_calib[i, j], sub_sigma,
                                                                                rho_opt[start:end, start:end])

        row_count += 1

    # print(LMM_GPC / 100)

    # # Rectangle Cascade Calibration Algorithm
    # row_count = 0
    # LMM_GPC[0, 0] = v_calib[0, 0]
    # for i in range(s):
    #     for j in range(s):
    #         T_alpha, T_beta = opt_maturity[i], swap_length[j]
    #         start, end = T_alpha - 1, T_alpha + T_beta - 1
    #
    #         w, F = calib.calc_swap_rate(T_alpha, T_beta, forwards, discount_factors)
    #         swap_rate = np.sum(w * F)
    #
    #         sub_sigma = LMM_GPC[row_count:row_count + T_beta, :T_alpha]
    #
    #         if start + s == end:
    #             result = calib.rectangular_cascade_calibration_algorithm(T_alpha, swap_rate, w, F,
    #                                                                      v_calib[i, j], sub_sigma,
    #                                                                      rho_opt[start:end, start:end])
    #             LMM_GPC[j + row_count, i] = result
    #
    #             for k in range(T_alpha - 1, -1, -1):
    #                 LMM_GPC[j + row_count, k] = result
    #
    #             break
    #
    #         else:
    #             LMM_GPC[j + row_count, i] = calib.cascade_calibration_algorithm(T_alpha, swap_rate, w, F,
    #                                                                             v_calib[i, j], sub_sigma,
    #                                                                             rho_opt[start:end, start:end])
    #     row_count += 1
    #
    # print(np.round(LMM_GPC / 100, 3))
    T_alpha, T_beta = 2, 4

    # Swaption Black Price
    A, tau = 0, 1.0
    sr = mc.calc_fwd_to_swap(forwards[T_alpha:T_alpha + T_beta], T_alpha, T_beta)
    K = sr
    for i in range(T_alpha, T_alpha + T_beta):
        A += tau * discount_factors[i]

    swaption = swaption_price(A, sr, K, v_swaptn[T_alpha - 1, T_beta - 1] / 100, T_alpha, 'log-normal')
    print(swaption)


    # Monte Carlo Pricing of Swaptions
    nb_paths = 100000

    print('Simulate start...')
    t1 = time.time()
    # # Single Process
    # simulate_F = mc.monte_carlo_simulation(nb_paths, forwards, LMM_GPC / 100, rho_opt, T_alpha, T_beta)

    # Multiple Process
    nb_worker = 4
    pool = Pool(processes=nb_worker)
    nb_sample_per_worker = int(nb_paths / nb_worker)
    nb_trail_per_worker = [nb_sample_per_worker] * nb_worker

    mc_sim = partial(mc.monte_carlo_simulation, forwards=forwards, fwd_vol=LMM_GPC / 100, fwd_corr=rho_opt, T_alpha=T_alpha, T_beta=T_beta)
    simulate_F = pool.map(mc_sim, nb_trail_per_worker)
    simulate_F = np.concatenate(simulate_F, axis=0)
    pool.close()
    pool.join()

    print('Simulation end, used ', time.time() - t1, ' secs')

    start, end = T_alpha, T_alpha + T_beta
    payoff, simulate_sr, simulate_annuity = np.zeros(nb_paths), np.zeros(nb_paths), np.zeros(nb_paths)

    # _, nb_steps, nb_fwd = simulate_annuity.shape

    for i in range(len(simulate_F)):
        F = simulate_F[i, :]
        swap_rate = mc.calc_fwd_to_swap(F, T_alpha, T_beta)
        simulate_sr[i] = swap_rate

        tau, annuity, fwd_disc_factor = 1.0, 0, 1.0
        for k in range(len(F)):
            fwd_disc_factor = fwd_disc_factor / (1 + tau * F[k])
            # temp *= (1 / (1 + tau * F[k]))
            annuity += (tau * fwd_disc_factor)

        payoff[i] = max(swap_rate - K, 0) * annuity * discount_factors[T_alpha - 1]
        simulate_annuity[i] = annuity

    # display_simulated_forward_rate(simulate_F, forwards[T_alpha:T_alpha + T_beta])

    # # # swap rate distribution
    # # plt.hist(simulate_sr, label='swap rate')
    # # plt.axvline(x=K, color='r')
    # # plt.axvline(x=np.mean(simulate_sr), color='g')
    # # plt.grid(True, linestyle='--')
    # # plt.title('Simulated Swap Rate')
    # # plt.show()
    #
    # # # annuity distribution
    # # plt.hist(simulate_annuity, label='Annuity')
    # # plt.axvline(x=A, color='g')
    # # plt.axvline(x=np.mean(simulate_annuity), color='r')
    # # plt.grid(True, linestyle='--')
    # # plt.title('Annuity Pattern')
    # # plt.show()
    # #
    # # # payoff disitribution
    # # plt.hist(payoff, label='swaption')
    # # plt.axvline(x=swaption, color='g')
    # # plt.axvline(x=np.mean(payoff), color='r')
    # # plt.grid(True, linestyle='--')
    # # plt.title('Payoff Pattern')
    # # plt.show()
    #
    # Confidence Interval
    z_table = {99: 2.58,
               98: 2.33,
               95.45: 2,
               95: 1.96,
               90: 1.65,
               68.27: 1}
    mean = np.sum(payoff) / nb_paths
    std = np.sqrt(np.sum(payoff**2) / nb_paths - mean**2)

    # Swaption Price
    lower_price, upper_price = mean - z_table[98] * std / np.sqrt(nb_paths), mean + z_table[98] * std / np.sqrt(nb_paths)
    lower_index, upper_index = abs(payoff - lower_price).argmin(), abs(payoff - upper_price).argmin()
    print(lower_price, '\t', mean, '\t', upper_price)

    # # implied Volatility
    # lower_sigma = swaption_vol_calibration(payoff[lower_index], simulate_annuity[lower_index], simulate_sr[lower_index], K, T_alpha)
    # upper_sigma = swaption_vol_calibration(payoff[upper_index], simulate_annuity[upper_index], simulate_sr[upper_index], K, T_alpha)
    # print(lower_sigma, '\t', upper_sigma)

    # imp_vol = swaption_vol_calibration(swaption, A, sr, K, T_alpha)
    # print(imp_vol)