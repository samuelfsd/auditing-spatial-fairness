import numpy as np
from src.metrics import calculate_sul

def get_random_types(N, P):
    """Cria o universo paralelo embaralhando as classes alvo."""
    return np.random.binomial(size=N, n=1, p=P/N)

def get_signif_threshold(signif_level, n_alt_worlds, regions, N, P):
    """
    Gera as permutações e retorna o valor SUL mínimo
    para uma região ser considerada estatisticamente significativa.
    """
    alt_max_likelis = []

    for _ in range(n_alt_worlds):
        alt_types = get_random_types(N, P)
        max_likeli = 0

        # Varre as regiões usando os dados 'falsos'
        for region in regions:
            pontos = region['points']
            n_r = len(pontos)
            if n_r == 0:
                continue

            p_r = alt_types[pontos].sum()
            likeli = calculate_sul(n_r, p_r, N, P)

            if likeli > max_likeli:
                max_likeli = likeli

        alt_max_likelis.append(max_likeli)

    alt_max_likelis.sort(reverse=True)

    # Pega o valor de corte (ex: p-valor de 0.005)
    k = int(signif_level * n_alt_worlds)

    return alt_max_likelis[k]