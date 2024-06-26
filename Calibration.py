import numpy as np

def calc_swap_rate(T_alpha, T_beta, forwards, discount_factors):
    tau = 1
    annuity, swap_start, swap_end = 0, T_alpha, T_alpha + T_beta
    w, F = [], []

    for i in range(swap_start, swap_end):
        disc_factor = discount_factors[i]
        annuity += tau * disc_factor
        w.append(tau * disc_factor)
        F.append(forwards[i])

    # disc_factor_start, disc_factor_end = discount_factors[swap_start - 1], discount_factors[swap_end - 1]
    # swap_rate = (disc_factor_start - disc_factor_end) / annuity
    # print(swap_rate)
    w = np.array(w) / annuity

    return w, F

def cascade_calibration_algorithm(T_alpha, swap_rate, w, F, V, Sigma, rho):
    A, B, C, result = 0, 0, 0, 0
    tau = 1

    for i in range(len(w) - 1):
        for j in range(i + 1):
            sigma_i, sigma_j = Sigma[i, :], Sigma[j, :]

            vol = 0
            for h in range(T_alpha):
                vol += (tau * sigma_i[h] * sigma_j[h])

            if i == j:
                C += (w[i] * w[j] * F[i] * F[j] * rho[i, j] * vol)
            else:
                C += 2 * (w[i] * w[j] * F[i] * F[j] * rho[i, j] * vol)

    idx_beta = len(w) - 1
    for j in range(len(w) - 1):
        sigma_i, sigma_j = Sigma[idx_beta, :], Sigma[j, :]

        vol = 0
        for h in range(T_alpha - 1):
            vol += (tau * sigma_i[h] * sigma_j[h])

        C += 2 * (w[idx_beta] * w[j] * F[idx_beta] * F[j] * rho[idx_beta, j] * vol)

    # index = len(w) - 1
    for j in range(len(w) - 1):
        sigma_j = Sigma[j, :]
        B += 2 * (w[idx_beta] * w[j] * F[idx_beta] * F[j] * rho[idx_beta, j] * tau * sigma_j[T_alpha - 1])

    vol = 0
    for h in range(T_alpha - 1):
        vol += (tau * Sigma[-1, h]**2)
    C += (w[-1]**2 * F[-1]**2 * vol)
    A = w[-1]**2 * F[-1]**2 * tau

    C -= (T_alpha * swap_rate**2 * V**2)

    result = (-B + np.sqrt(B**2 - 4*A*C)) / (2 * A)

    return result


def rectangular_cascade_calibration_algorithm(T_alpha, swap_rate, w, F, V, Sigma, rho):
    A, B, C, result = 0, 0, 0, 0
    tau = 1

    temp = 0
    for h in range(T_alpha - 1):
        temp += (w[0]**2 * F[0]**2 * tau)
    A = w[-1]**2 * F[-1]**2 * tau + temp

    beta_index = len(w) - 1
    part_1, part_2 = 0, 0
    for j in range(len(w) - 1):
        sigma_j = Sigma[j, :]
        part_1 += 2 * (w[beta_index] * F[beta_index] * w[j] * F[j] * rho[beta_index, j] * tau * sigma_j[T_alpha - 1])

    for j in range(len(w) - 1):
        sigma_j = Sigma[j, :]
        vol = 0
        for h in range(T_alpha - 1):
            vol += tau * sigma_j[h]
        part_2 += 2 * (w[beta_index] * w[j] * F[beta_index] * F[j] * vol)

    B = part_1 + part_2

    for i in range(len(w) - 1):
        for j in range(len(w) - 1):
            sigma_i, sigma_j = Sigma[i, :], Sigma[j, :]

            vol = 0
            for h in range(T_alpha):
                vol += (tau * sigma_i[h] * sigma_j[h])

            C += (w[i] * w[j] * F[i] * F[j] * rho[i, j] * vol)

    C -= (T_alpha * swap_rate ** 2 * V ** 2)

    result = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    return result