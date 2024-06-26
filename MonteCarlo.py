import numpy as np
import matplotlib.pyplot as plt




def calc_fwd_to_swap(F, T_alpha, T_beta):
    tau = 1
    numerator, denumerator = 0, 0

    fwd_disc_factor = 1
    for i in range(len(F)):
        fwd_disc_factor *= 1 / (1 + tau * F[i])

    numerator = 1 - fwd_disc_factor

    for i in range(len(F)):
        temp = 1
        for j in range(i + 1):
            temp *= (1 / (1 + tau * F[j]))
        denumerator += (tau * temp)

    return numerator / denumerator


def calc_drift_term(F, sigma, rho, tau):
    drift = 0
    for i in range(len(rho)):
        drift += (rho[i] * tau * sigma[i] * F[i]) / (1 + tau * F[i])

    return drift


def generate_random_varialbes(fwd_corr, time_steps, T_alpha, T_beta):
    z = np.random.randn(T_beta, len(time_steps))

    if T_beta > 1:
        sub_corr = fwd_corr[T_alpha - 1:T_alpha + T_beta - 1, T_alpha - 1:T_alpha + T_beta - 1]
        cholesky_matrix = np.linalg.cholesky(sub_corr)

        z = np.dot(cholesky_matrix, z)

    return z


def monte_carlo_simulation(nb_paths, forwards, fwd_vol, fwd_corr, T_alpha, T_beta):
    nb_path = 0
    time_steps = np.linspace(0, T_alpha, 4 * T_alpha + 1)

    start, end = T_alpha, T_alpha + T_beta
    dt = time_steps[1] - time_steps[0]

    sim_F = np.zeros((nb_paths, T_beta))
    # sim_F = np.zeros((nb_paths, len(time_steps), T_beta))
    # swap_rate = np.zeroso(nb_paths)
    swap_rate_sim = np.zeros((nb_paths, len(time_steps)))

    # print('swap rate: ', calc_fwd_to_swap(forwards[start:end], T_alpha, T_beta))

    while nb_path < nb_paths:
        F = forwards[start:end]
        log_F, log_F_next = np.log(F), np.zeros(len(F))
        Z = generate_random_varialbes(fwd_corr, time_steps, T_alpha, T_beta)

        for i in range(1, len(time_steps)):
            # t = time_steps[i]

            sub_sigma = fwd_vol[start - 1:end - 1, int(np.ceil(i / 4)) - 1]
            z = Z[:, i]

            # if nb_path == 0:
            #     print(t, '\t', sub_corr)

            # simulate forward rates, alpha + 1, alpha + 2, ..., beta
            for k in range(T_beta):
                # drift term
                sub_rho = fwd_corr[start - 1 + k, start:start - 1 + k + 1]
                drift = calc_drift_term(F, sub_sigma, sub_rho, 1.0)

                log_F_next[k] = log_F[k] + sub_sigma[k] * drift * dt - 0.5 * sub_sigma[k] ** 2 * dt + sub_sigma[k] * z[k] * np.sqrt(dt)

            log_F = log_F_next
            F = np.exp(log_F)
            # sim_F[nb_path, i, :] = F
            # swap_rate_sim[nb_path, i] = calc_fwd_to_swap(F, T_alpha, T_beta)

        sim_F[nb_path, :] = F
        # swap_rate[nb_path] = calc_fwd_to_swap(F, T_alpha, T_beta)
        nb_path += 1

    # plt.hist(swap_rate)
    # plt.grid(True, linestyle='--')
    # plt.show()


    return sim_F
