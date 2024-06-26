import numpy as np
from scipy.optimize import root, minimize

def Rebonato_three_factor_parameterization(corr):
    M = corr.shape[0]

    # corr = np.sqrt(corr)
    # rho_inf = corr[0, M - 1]
    # rho_inf = 0.23551
    f = lambda rho_inf: (corr[0, M - 1] - rho_inf) / (1 - rho_inf) - ((corr[M - 2, M - 1] - rho_inf) / (1 - rho_inf))**(M - 1)
    sol = root(f, 0)
    rho_inf = sol.x[0]

    alpha = np.log((corr[0, 1] - rho_inf) / (corr[M - 2, M - 1] - rho_inf)) / (2 - M)
    beta = alpha - np.log((corr[0, 1] - rho_inf) / (1 - rho_inf))

    corr_rebo = np.zeros((M, M))
    rho_generaotr = lambda i, j: rho_inf + (1 - rho_inf) * np.exp(-abs(i - j) * (beta - alpha * (max(i, j) - 1)))

    for i in range(M):
        for j in range(M):
            corr_rebo[i, j] = rho_generaotr(i, j)

    return corr_rebo


def Schoenmakers_and_Coffey_parameterization(corr_matrix, factor):
    M = corr_matrix.shape[0]
    rho_SC = np.zeros((M, M))

    if factor == 3:
        beta = -np.log(corr_matrix[M - 2, M - 1])
        alpha_1 = 6 * np.log(corr_matrix[0, M - 1]) / ((M - 1) * (M - 2)) - 2 * np.log(corr_matrix[M - 2, M - 1]) / (M - 2) - 4 * np.log(corr_matrix[0, 1]) / (M - 2)
        alpha_2 = -6 * np.log(corr_matrix[0, M - 1]) / ((M - 1) * (M - 2)) + 4 * np.log(corr_matrix[M - 2, M - 1]) / (M - 2) + np.log(corr_matrix[0, 1]) / (M - 2)

        rho_generator = lambda i, j: np.exp(-abs(i - j) * (beta - alpha_2 / (6 * M - 18) * (i**2 + j**2 + i * j - 6 * i - 6 * j - 3 * M**2 + 15 * M - 7) + alpha_1 / (6 * M - 18) * (i**2 + j**2 + i * j - 3 * M * i - 3 * M * j + 3 * i + 3 * j + 3 * M**2 - 6 * M + 2)))

        for i in range(M):
            for j in range(M):
                rho_SC[i, j] = rho_generator(i, j)

        return rho_SC
    elif factor == 2:
        rho_inf = corr_matrix[0, M - 1]
        eta = (-np.log(corr_matrix[0, 1]) * (M - 1) + np.log(rho_inf)) / 2

        rho_generator = lambda i, j: np.exp(-abs(i - j) / (M - 1) * (-np.log(rho_inf) + eta * (i**2 + j**2 + i * j - 3 * M * i - 3 * M * j + 3 * i + 3 * j + 2 * M**2 - M - 4) / ((M - 2) * (M - 3))))

        for i in range(M):
            for j in range(M):
                rho_SC[i, j] = rho_generator(i, j)

        return rho_SC

    return corr_matrix

def eigenvalue_zeroing(corr, rank):
    M = corr.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(corr)

    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    P = eigenvectors[:, -rank:]

    Omega = np.zeros(rank)
    for k in range(rank):
        Omega[k] = eigenvalues[-(rank - k)]
    Omega = np.diag(np.sqrt(Omega))

    B = P @ Omega
    cov = B @ B.T

    rho = np.zeros((M, M))
    for i in range(M):
        for j in range(i):
            rho[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
            rho[j, i] = rho[i, j]

    # find corresponding theta
    theta = np.zeros((M, rank - 1))
    for i in range(M):
        for j in range(rank - 1):
            if j == 0:
                theta[i, j] = np.arccos(B[i, j])
            else:
                temp, k = 1, 0
                while k < j:
                    temp *= np.sin(theta[i, k])
                    k += 1
                theta[i, j] = np.arccos(B[i, j])

    return rho, theta


def theta_to_B_matrix(theta):
    nb_row, nb_rank = theta.shape
    nb_rank += 1

    B = np.zeros((nb_row, nb_rank))
    for i in range(nb_row):
        for j in range(nb_rank):
            if j == 0:
                B[i, j] = np.cos(theta[i, j])
            elif j == nb_rank - 1:
                B[i, j] = np.prod(np.sin(theta[i, :]))
            else:
                B[i, j] = np.cos(theta[i, j]) * np.prod(np.sin(theta[i, :j - 1]))

    return B

def obj_function(theta, args):
    target, rank = args[0], args[1]
    nb_row, nb_col = target.shape
    theta = theta.reshape((nb_row, rank - 1))

    # B = np.zeros((nb_row, rank))
    # for i in range(nb_row):
    #     for j in range(rank):
    #         if j == 0:
    #             B[i, j] = np.cos(theta[i, j])
    #         elif j == rank - 1:
    #             B[i, j] = np.prod(np.sin(theta[i, :]))
    #         else:
    #             B[i, j] = np.cos(theta[i, j]) * np.prod(np.sin(theta[i, :j - 1]))
    # B = np.array([np.cos(theta), np.sin(theta)]).T
    B = theta_to_B_matrix(theta)

    rho_B = B @ B.T
    error = np.sum((target - rho_B) ** 2)

    return error

def correlation_optimization(target, init_guess, rank):
    nb_row, nb_col = target.shape

    # init_guess = np.full((nb_row, rank - 1), np.pi / 2).reshape(-1)

    args = [target, rank]
    result = minimize(obj_function, init_guess, args, method='BFGS')
    theta_opt = result['x'].reshape((nb_row, rank - 1))

    B = theta_to_B_matrix(theta_opt)
    rho_opt = B @ B.T

    return rho_opt


def terminal_correlation(T_alpha, corr):
    M = corr.shape[0] - T_alpha
    term_corr = np.zeros((M, M))
    tau = 1
    for i in range(M):
        for j in range(i + 1):
            rho = corr[T_alpha + i - 1, T_alpha + j - 1]
            sigma_i, sigma_j = corr[T_alpha + i - 1, :], corr[T_alpha + j - 1, :]

            part_1, part_2, part_3 = 0, 0, 0
            for h in range(T_alpha):
                part_1 += (sigma_i[h] * sigma_j[h])
                part_2 += sigma_i[h]**2
                part_3 += sigma_j[h]**2

            term_corr[i, j] = rho * part_1 / (np.sqrt(part_2) * np.sqrt(part_3))
            term_corr[j, i] = term_corr[i, j]

    return term_corr