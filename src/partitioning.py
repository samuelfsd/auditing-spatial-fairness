from sklearn.cluster import KMeans
import random
import numpy as np

def get_grid_regions(df, rtree, nx=20, ny=20):
    """Método Baseline: Sobrepõe uma grade fixa sobre o mapa."""
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()

    regions = []
    for i in range(nx):
        lon_s = lon_min + (i/nx)*(lon_max-lon_min)
        lon_e = lon_min + ((i+1)/nx)*(lon_max-lon_min)
        for j in range(ny):
            lat_s = lat_min + (j/ny)*(lat_max-lat_min)
            lat_e = lat_min + ((j+1)/ny)*(lat_max-lat_min)

            # Busca na RTree todos os pontos dentro desse retângulo
            points = list(rtree.intersection((lon_s, lat_s, lon_e, lat_e)))

            regions.append({
                'points': points,
                'type': 'grid',
                'grid_loc': (i, j),
                'bounds': (lon_s, lat_s, lon_e, lat_e)
            })

    return regions

def get_kmeans_regions(df, rtree, n_seeds=100, radii=[0.01]):
    """Método do Autor: K-Means seguidos de expansão por raios."""
    X = df[['lon', 'lat']].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init='auto').fit(X)

    # Encontra o ponto de dado real mais próximo de cada centróide matemático
    seeds = []
    for c in kmeans.cluster_centers_:
        nearest_point = list(rtree.nearest([c[0], c[1]], 1))[0]
        seeds.append(nearest_point)

    regions = []
    for seed in seeds:
        lat, lon = df.loc[seed, 'lat'], df.loc[seed, 'lon']
        for r in radii:
            # Pega pontos em um quadrado (bounding box) ao redor do centro
            points = list(rtree.intersection((lon-r, lat-r, lon+r, lat+r)))

            regions.append({
                'points': points,
                'center': seed,
                'radius': r,
                'type': 'kmeans'
            })

    return regions

def get_random_regions(df, rtree, n_seeds=100, radii=[0.01]):
    """
    Método Baseline 'Burro': Escolhe pontos aleatórios do mapa como sementes.
    Ignora a densidade populacional completamente.
    """
    # Pega todos os IDs válidos do seu dataset
    indices = df.index.tolist()
    
    # Sorteia 'n_seeds' aleatoriamente
    # O min() garante que não dê erro se você pedir mais sementes do que os dados existentes
    seeds = random.sample(indices, min(n_seeds, len(indices)))

    regions = []
    for seed in seeds:
        lat, lon = df.loc[seed, 'lat'], df.loc[seed, 'lon']
        for r in radii:
            # Pega os pontos ao redor desse centro sorteado usando distância Euclidiana simples (bounding box)
            points = list(rtree.intersection((lon-r, lat-r, lon+r, lat+r)))

            regions.append({
                'points': points,
                'center': seed,
                'radius': r,
                'type': 'random'
            })

    return regions