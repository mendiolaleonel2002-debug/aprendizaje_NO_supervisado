import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('./data/retailmax.csv')
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaled_features = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)

for ax, k in zip(axes, [3, 5, 10]):
    model = KMeans(n_clusters=k, init='k-means++', max_iter=300,
                   n_init=10, random_state=42)
    plot_df = df.copy()
    plot_df['Cluster'] = model.fit_predict(scaled_features)
    sns.scatterplot(
        data=plot_df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set1',
        s=55,
        alpha=.85,
        ax=ax,
    )
    ax.set_title(f'K-Means con k={k}')
    ax.set_xlabel('Ingreso anual (k$)')
    ax.set_ylabel('Puntuación de gasto' if k == 3 else '')
    ax.legend(title='Cluster', fontsize=8)

fig.suptitle('Comparación de 3, 5 y 10 clusters', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('comparacion_clusters_k3_k5_k10.png', dpi=180, bbox_inches='tight')

# Método del codo solicitado en la actividad.
def calcular_wcss(datos):
    wcss = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300,
                        n_init=10, random_state=42)
        kmeans.fit(datos)
        wcss.append(kmeans.inertia_)
    return wcss

wcss = calcular_wcss(scaled_features)

plt.figure(figsize=(9, 5.5))
plt.plot(range(1, 11), wcss, color='#2563EB', marker='o', linewidth=2.5)
plt.axvline(5, color='#DC2626', linestyle='--', alpha=.8, label='Codo sugerido: k=5')
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.legend()
plt.tight_layout()
plt.savefig('metodo_del_codo_wcss.png', dpi=180, bbox_inches='tight')

# Visualización final del modelo seleccionado con cinco clusters.
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300,
                n_init=10, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='Set1')
plt.title('Clusters de clientes')
plt.tight_layout()
plt.savefig('clusters_clientes_k5.png', dpi=180, bbox_inches='tight')

# Visualización tridimensional del modelo con cinco clusters.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sns.set(style='whitegrid')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    df['Age'],
    c=df['Cluster'],
    cmap='Set1',
    s=50,
    alpha=.85,
)
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
ax.set_title('Clusters de clientes en 3D')
legend1 = ax.legend(*scatter.legend_elements(), title='Clusters')
ax.add_artist(legend1)
plt.tight_layout()
plt.savefig('clusters_clientes_k5_3d.png', dpi=180, bbox_inches='tight')
