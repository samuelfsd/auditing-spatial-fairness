import argparse
import pandas as pd
import numpy as np
from src.functions import *
import os

import argparse
import pandas as pd
import numpy as np
from src.functions import *
from src.partitioning import get_grid_regions, get_kmeans_regions, get_random_regions
from src.metrics import calculate_sul, calculate_meanvar
from src.simulation import get_signif_threshold
import os

def main():
    parser = argparse.ArgumentParser(description="Auditoria Espacial de Justiça Algorítmica")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['lar', 'crime', 'synth_fair', 'synth_unfair', 'semisynth'],
        default='lar',
        help="Escolha qual dataset auditar (ex: lar, crime, synth_fair)."
    )

    parser.add_argument(
        '--method',
        type=str, 
        choices=['kmeans', 'grid', 'random'], default='kmeans',
        help="Método principal para gerar o mapa de calor final"
    )

    args = parser.parse_args()

    print(f"\n[INICIANDO] Processo de auditoria | Dataset: {args.dataset.upper()}")

    # config routes with argparse
    if args.dataset == 'lar':
        data_path, label, radii = './data/LAR.csv', 'action_taken', np.arange(0.05, 1.01, 0.05)
    elif args.dataset == 'crime':
        data_path, label, radii = './data/Crime.csv', 'pred', np.arange(0.005, 0.1, 0.005)
    elif args.dataset == 'synth_fair':
        data_path, label, radii = './data/Synth_fair.csv', 'label', np.arange(0.01, 0.2, 0.01)
    elif args.dataset == 'synth_unfair':
        data_path, label, radii = './data/Synth_unfair.csv', 'label', np.arange(0.01, 0.2, 0.01)
    elif args.dataset == 'semisynth':
        data_path, label, radii = './data/Semisynth.csv', 'label', np.arange(0.01, 0.2, 0.01)

    if not os.path.exists(data_path):
        print(f"Erro: Arquivo não encontrado no caminho {data_path}")
        return

    # indexing
    print("Carregando dados e construindo R-Tree...")
    df = load_data(data_path)
    N, P = get_stats(df, label)
    true_types = get_true_types(df, label)
    rtree_index = create_rtree(df)

    print(f"   > N={N} pontos totais | P={P} casos positivos")

    # EXPERIMENTO DE COMPARAÇÃO NUMÉRICA (O "Chão de Fábrica")
    print("\n[FASE 1] Calculando Baseline 1 (Grade Fixa 20x20)...")
    lat_min, lat_max, lon_min, lon_max = df['lat'].min(), df['lat'].max(), df['lon'].min(), df['lon'].max()
    grid_regions = get_grid_regions(df, rtree_index, nx=20, ny=20)

    rhos_grid = []
    sul_grid_max = 0
    for reg in grid_regions:
        n_r = len(reg['points'])
        if n_r > 0:
            p_r = true_types[reg['points']].sum()
            rhos_grid.append(p_r / n_r)
            sul_grid_max = max(sul_grid_max, calculate_sul(n_r, p_r, N, P))

    meanvar_grid = calculate_meanvar(rhos_grid)

    print("[FASE 2] Calculando Baseline 2 (Sementes Aleatórias)...")
    random_regions = get_random_regions(df, rtree_index, n_seeds=100, radii=radii)
    sul_random_list = [calculate_sul(len(r['points']), true_types[r['points']].sum(), N, P) for r in random_regions]
    sul_random_max = max(sul_random_list)

    print("[FASE 3] Calculando Abordagem do Autor (K-Means Scan)...")
    kmeans_regions = get_kmeans_regions(df, rtree_index, n_seeds=100, radii=radii)
    sul_kmeans_list = [calculate_sul(len(r['points']), true_types[r['points']].sum(), N, P) for r in kmeans_regions]
    sul_kmeans_max = max(sul_kmeans_list)

    print("\n" + "="*50)
    print("TABELA DE COMPARAÇÃO DE MÉTODOS")
    print("="*50)
    print(f"Métrica Baseline (MeanVar da Grade): {meanvar_grid:.6f}")
    print(f"Métrica SUL Máxima (Grade Fixa):   {sul_grid_max:.4f}")
    print(f"Métrica SUL Máxima (Semente Aleatória): {sul_random_max:.4f}")
    print(f"Métrica SUL Máxima (K-Means Scan): {sul_kmeans_max:.4f}")
    print("="*50)

    # EXECUÇÃO DO MÉTODO ESCOLHIDO PARA O MAPA
    print(f"\n[FASE 4] Executando Análise de Significância para o método: {args.method.upper()}")
    
    # Roteamento baseado no argumento do terminal
    if args.method == 'kmeans':
        target_regions, target_statistics = kmeans_regions, sul_kmeans_list
    elif args.method == 'random':
        target_regions, target_statistics = random_regions, sul_random_list
    else:
        target_regions = grid_regions
        target_statistics = [calculate_sul(len(r['points']), true_types[r['points']].sum(), N, P) for r in grid_regions]

    n_alt_worlds = 200
    signif_level = 0.005
    print(f"   > Simulando {n_alt_worlds} mundos paralelos de Monte Carlo...")
    
    signif_thresh = get_signif_threshold(signif_level, n_alt_worlds, target_regions, N, P)
    print(f"   > Threshold de significância (p < 0.005): {signif_thresh:.4f}")

    sorted_statistics = np.sort(target_statistics)
    top_k = len(target_statistics) - np.searchsorted(sorted_statistics, signif_thresh)
    print(f"\n[FASE 5] {top_k} regiões significativas encontradas.")

    if top_k == 0:
        print("Nenhuma região estatisticamente significativa para gerar no mapa.")
        return

    indexes = np.argsort(target_statistics)[::-1][:top_k]
    significant_regions = [target_regions[i] for i in indexes]

    print("\n[ANÁLISE REGIÕES] Raio-X da Região Mais Injusta Encontrada:")
    pior_regiao = significant_regions[0] 

    n_in, p_in, rho_in = get_simple_stats(pior_regiao['points'], true_types)
    rho_global = P / N

    print(f"   > População local (n): {n_in} casos")
    print(f"   > Casos positivos locais (p): {p_in}")
    print(f"   > Taxa Local (rho): {rho_in * 100:.1f}%")
    print(f"   > Taxa Global do Mapa: {rho_global * 100:.1f}%")

    if rho_in < rho_global:
        print("   > TIPO DE VIÉS: NEGATIVO (O modelo parece prejudicar esta região)")
    else:
        print("   > TIPO DE VIÉS: POSITIVO (O modelo parece favorecer esta região)")
    print("-" * 50)

    print("Filtrando regiões sobrepostas...")
    non_olap_regions = []
    centers = []
    for region in significant_regions:
        center = region.get('center', region.get('grid_loc'))
        if center in centers: continue

        no_intersections = True
        for other in non_olap_regions:
            # A checagem de sobreposição suporta os métodos baseados em círculos/bounding boxes
            if args.method in ['kmeans', 'random'] and intersects(region, other, df):
                no_intersections = False
                break

        if no_intersections:
            centers.append(center)
            non_olap_regions.append(region)

    print(f"{len(non_olap_regions)} regiões não sobrepostas mantidas.")

    # Geração do mapa
    # if len(non_olap_regions) > 0:
    #     print("\n[FASE 6] Gerando mapa com os resultados...")

    #     # Como o 'random' usa a mesma estrutura geométrica do K-Means (Centro + Raio), podemos aproveitar a mesma função gráfica!
    #     if args.method in ['kmeans', 'random']:
    #         mapa_resultado = show_circular_regions(df, true_types, non_olap_regions[:5])
    #         nome_arquivo = f"resultado_{args.dataset}_{args.method}.html"
    #         mapa_resultado.save(nome_arquivo)
    #         print(f"SUCESSO! Mapa salvo como '{nome_arquivo}'.")
    #     else:
    #         print("Nota: O mapa visual para Grade Fixa requer adaptações. Verifique a tabela numérica acima.")

if __name__ == "__main__":
    main()