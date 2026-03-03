import argparse
import pandas as pd
import numpy as np
from src.functions import *
import os

def main():
    parser = argparse.ArgumentParser(description="Auditoria Espacial de Justiça Algorítmica")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['lar', 'crime', 'synth_fair', 'synth_unfair', 'semisynth'],
        default='lar', # Padrão: se não passar flag, roda o LAR
        help="Escolha qual dataset auditar (ex: lar, crime, synth_fair)."
    )

    args = parser.parse_args()

    print(f"init audit process in case: {args.dataset.upper()}")

    if args.dataset == 'lar':
        data_path = './data/LAR.csv'
        label = 'action_taken'
    elif args.dataset == 'crime':
        data_path = './data/Crime.csv'
        label = 'pred'
    elif args.dataset == 'synth_fair':
        data_path = './data/Synth_fair.csv'
        label = 'label'
    elif args.dataset == 'synth_unfair':
        data_path = './data/Synth_unfair.csv'
        label = 'label'
    elif args.dataset == 'semisynth':
        data_path = './data/Semisynth.csv'
        label = 'label'

    # validation csv
    if not os.path.exists(data_path):
        print(f"Erro: Arquivo não encontrado no caminho {data_path}")
        return

    # load data
    df = load_data(data_path)

    N, P = get_stats(df, label)
    print(f'N={N} pontos totais')
    print(f'P={P} casos positivos')

    true_types = get_true_types(df, label)

    #  creating tree R (Indexação Espacial)
    print("Criando RTree...")
    rtree_index = create_rtree(df)

    # creating seeds (kmeans, )
    print("Gerando sementes (KMeans)...")
    seeds = create_seeds(df, rtree_index, 100)

    radii = np.arange(0.05, 1.01, 0.05)
    print("Gerando regiões a partir das sementes...")
    regions = create_regions(df, rtree_index, seeds, radii)
    print(f'{len(regions)} regiões geradas.')

    # 4. calculating SUL (Spatial Unfairness Likelihood)
    direction = 'both'
    print(f"Buscando regiões injustas (direction: {direction})...")
    best_region, max_likeli, statistics = scan_regions(regions, true_types, N, P, direction=direction, verbose=False)

    # 5. metrics test (Monte Carlo)
    n_alt_worlds = 200 # 20 case u want test quickly
    signif_level = 0.005

    signif_thresh = get_signif_threshold(signif_level, n_alt_worlds, regions, N, P)
    print(f"Threshold de significância: {signif_thresh}")

    # filtering significant regions
    sorted_statistics = np.sort(statistics)
    top_k = len(statistics) - np.searchsorted(sorted_statistics, signif_thresh)
    print(f'{top_k} regiões significativas encontradas.')

    indexes = np.argsort(statistics)[::-1][:top_k]
    significant_regions = [ regions[i] for i in indexes ]

    # filtering overlapping regions
    print("Removendo regiões sobrepostas...")
    non_olap_regions = []
    centers = []
    for region in significant_regions:
        center = region['center']
        if center in centers:
            continue

        no_intersections = True
        for other in non_olap_regions:
            if intersects(region, other, df):
                no_intersections = False
                break

        if no_intersections:
            centers.append(center)
            non_olap_regions.append(region)

    print(f'{len(non_olap_regions)} regiões não sobrepostas mantidas.')

    # generating map
    if len(non_olap_regions) > 0:
        print("Gerando mapa com os resultados...")
        mapa_resultado = show_circular_regions(df, true_types, non_olap_regions[:5])
        mapa_resultado.save("resultado_auditoria.html")
        print("Mapa salvo com sucesso!")
    else:
        print("Nenhuma região significativa para gerar no mapa.")

if __name__ == "__main__":
    main()