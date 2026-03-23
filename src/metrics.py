import numpy as np
import math

def calculate_sul(n, p, N, P):
    """
    Retorna a métrica SUL (Spatial Unfairness Likelihood) original do autor.
    Resolve o erro de log(0) de forma matematicamente segura.
    """
    rho = P/N

    def safe_x_log_prob(x, prob):
        if x <= 0 or prob <= 0:
            return 0
        return x * math.log(prob)

    l0max = safe_x_log_prob(P, rho) + safe_x_log_prob(N-P, 1-rho)

    if n == 0 or n == N:
        return 0 # Se a região está vazia ou engloba o mapa todo, a diferença de likelihood é zero.

    rho_in = p/n
    rho_out = (P-p)/(N-n)

    term1 = safe_x_log_prob(p, rho_in)
    term2 = safe_x_log_prob(n-p, 1-rho_in)
    term3 = safe_x_log_prob(P-p, rho_out)
    term4 = safe_x_log_prob(N-n - (P-p), 1-rho_out)

    l1max = term1 + term2 + term3 + term4

    return l1max - l0max

def calculate_meanvar(rhos):
    """
    Retorna a Variância das taxas (Baseline MeanVar de Xie et al.).
    """
    rhos = np.array(rhos)
    rhos = rhos[~np.isnan(rhos)] # Remove áreas sem dados (n=0)

    if len(rhos) == 0:
        return 0

    mean_rho = np.mean(rhos)
    return np.mean((rhos - mean_rho)**2)