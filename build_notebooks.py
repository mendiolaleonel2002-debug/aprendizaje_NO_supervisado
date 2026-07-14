from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return nb


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


eda = notebook([
    md("""
# 1. Análisis Exploratorio de Datos (EDA) — RetailMax

**Objetivo.** Comprender la estructura, calidad y relaciones del conjunto de datos antes de segmentar clientes. El análisis es reproducible: las estadísticas se calculan directamente desde `data/retailmax.csv`.
"""),
    code("""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['axes.titleweight'] = 'bold'

DATA_PATH = Path('data/retailmax.csv')
df = pd.read_csv(DATA_PATH, dtype={'CustomerID': str})
df.head()
"""),
    md("""
## Revisión inicial y calidad de datos

Primero verificamos dimensiones, tipos, duplicados, valores faltantes y rangos. `CustomerID` es un identificador, no una medida del cliente; por ello no se utilizará como variable de segmentación.
"""),
    code("""
print(f'Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas')
display(df.info())
display(df.describe(include='all').T)

quality = pd.DataFrame({
    'tipo': df.dtypes.astype(str),
    'faltantes': df.isna().sum(),
    'únicos': df.nunique()
})
display(quality)
print(f'Filas duplicadas: {df.duplicated().sum()}')
print(f'CustomerID duplicados: {df["CustomerID"].duplicated().sum()}')
"""),
    md("""
**Hallazgo de calidad.** Hay 200 registros, no existen valores faltantes ni filas/identificadores duplicados. Las edades (18–70), ingresos (15–137 k$) y puntuaciones (1–99) están dentro de rangos plausibles. El conjunto está listo para el análisis sin imputación ni eliminación de observaciones.

## 1. ¿Cuál es la distribución de edades de los clientes?
"""),
    code("""
age_stats = df['Age'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
display(age_stats.to_frame('Edad'))

fig, ax = plt.subplots()
sns.histplot(df['Age'], bins=15, kde=True, color='#2563EB', edgecolor='white', ax=ax)
ax.axvline(df['Age'].mean(), color='#DC2626', ls='--', lw=2,
           label=f'Media: {df["Age"].mean():.1f}')
ax.axvline(df['Age'].median(), color='#16A34A', ls=':', lw=2,
           label=f'Mediana: {df["Age"].median():.1f}')
ax.set(title='Distribución de edades', xlabel='Edad (años)', ylabel='Número de clientes')
ax.legend()
plt.show()
"""),
    md("""
**Interpretación.** La edad media es **38.85 años** y la mediana **36 años**; el 50% central se encuentra entre **28.75 y 49 años**. El rango completo es de 18 a 70 años y la cola hacia edades mayores explica que la media sea superior a la mediana. RetailMax atiende una base amplia, con mayor presencia relativa de adultos jóvenes.

## 2. ¿Existen diferencias en los ingresos anuales entre hombres y mujeres?
"""),
    code("""
income_by_gender = (df.groupby('Gender')['Annual Income (k$)']
                    .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
                    .round(2))
display(income_by_gender)

female_income = df.loc[df['Gender'].eq('Female'), 'Annual Income (k$)']
male_income = df.loc[df['Gender'].eq('Male'), 'Annual Income (k$)']
u_stat, p_value = mannwhitneyu(female_income, male_income, alternative='two-sided')
print(f'Mann–Whitney U = {u_stat:.0f}; p = {p_value:.3f}')

fig, ax = plt.subplots()
sns.boxplot(data=df, x='Gender', y='Annual Income (k$)', hue='Gender',
            palette={'Male': '#2563EB', 'Female': '#EC4899'}, legend=False, ax=ax)
sns.stripplot(data=df, x='Gender', y='Annual Income (k$)', color='#111827',
              alpha=.25, jitter=.18, size=4, ax=ax)
ax.set(title='Ingreso anual por género', xlabel='Género', ylabel='Ingreso anual (miles de USD)')
plt.show()
"""),
    md("""
**Interpretación.** Los hombres presentan una media de **62.23 k$** y las mujeres de **59.25 k$** (diferencia descriptiva de 2.98 k$); las medianas son 62.5 y 60 k$, respectivamente. Sin embargo, las distribuciones se superponen ampliamente y la prueba no paramétrica de Mann–Whitney no detecta evidencia de diferencia al nivel de 5% (**p = 0.414**). No conviene diseñar campañas suponiendo distinto poder adquisitivo únicamente por género.

## 3. ¿Cómo se distribuye la puntuación de gasto entre los diferentes rangos de edades?
"""),
    code("""
labels = ['18–29', '30–39', '40–49', '50–59', '60–70']
df['Age Range'] = pd.cut(df['Age'], bins=[17, 29, 39, 49, 59, 70], labels=labels)
spending_by_age = (df.groupby('Age Range', observed=True)['Spending Score (1-100)']
                   .agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2))
display(spending_by_age)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Gender',
                palette={'Male': '#2563EB', 'Female': '#EC4899'}, alpha=.8, ax=axes[0])
axes[0].set(title='Edad y puntuación de gasto', xlabel='Edad', ylabel='Puntuación de gasto')
sns.boxplot(data=df, x='Age Range', y='Spending Score (1-100)', hue='Age Range',
            palette='Blues', legend=False, ax=axes[1])
axes[1].set(title='Gasto por rango de edad', xlabel='Rango de edad', ylabel='Puntuación de gasto')
plt.tight_layout()
plt.show()
"""),
    md("""
**Interpretación.** Los grupos de 18–29 y 30–39 años tienen los promedios de gasto más altos (**58.58** y **61.10**) y también una gran dispersión. A partir de los 40 años el promedio disminuye: 34.95 (40–49), 34.72 (50–59) y 43.00 (60–70). Existe una excepción de alto gasto en 40–49, por lo que la edad orienta la segmentación pero no determina por sí sola el comportamiento individual.

## 4. ¿Cuál es la correlación entre el ingreso anual y la puntuación de gasto?
"""),
    code("""
income_spending_corr = df['Annual Income (k$)'].corr(df['Spending Score (1-100)'])
print(f'Correlación de Pearson: {income_spending_corr:.3f}')

fig, ax = plt.subplots()
sns.regplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
            scatter_kws={'alpha': .65, 's': 50, 'color': '#2563EB'},
            line_kws={'color': '#DC2626'}, ax=ax)
ax.set(title='Ingreso anual y puntuación de gasto', xlabel='Ingreso anual (k$)',
       ylabel='Puntuación de gasto')
plt.show()
"""),
    md("""
**Interpretación.** La correlación de Pearson es **0.010**, prácticamente nula. Esto significa que no existe una relación *lineal* global entre ingreso y gasto. La dispersión sí muestra subgrupos —sobre todo entre ingresos altos—, por lo que correlación nula no equivale a ausencia total de estructura.

## 5. ¿Cómo varía la puntuación de gasto en diferentes grupos de ingresos anuales?
"""),
    code("""
income_labels = ['Bajo (<40 k$)', 'Medio (40–69 k$)', 'Alto (≥70 k$)']
df['Income Range'] = pd.cut(df['Annual Income (k$)'], bins=[0, 39, 69, np.inf],
                            labels=income_labels)
spending_by_income = (df.groupby('Income Range', observed=True)['Spending Score (1-100)']
                      .agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2))
display(spending_by_income)

fig, ax = plt.subplots()
sns.violinplot(data=df, x='Income Range', y='Spending Score (1-100)',
               hue='Income Range', palette='Blues', inner='box', legend=False, ax=ax)
ax.set(title='Puntuación de gasto por rango de ingreso', xlabel='Rango de ingreso anual',
       ylabel='Puntuación de gasto')
plt.show()
"""),
    md("""
**Interpretación.** Las medias son muy similares: 49.74, 50.41 y 50.26 para ingreso bajo, medio y alto. La diferencia importante está en la **dispersión**: los grupos bajo y alto contienen clientes con gasto tanto muy bajo como muy alto, mientras el rango medio se concentra cerca de 50. Esta heterogeneidad respalda el uso de clustering.

## 6. ¿Cuál es la proporción de clientes por género?
"""),
    code("""
gender_counts = df['Gender'].value_counts()
gender_share = (df['Gender'].value_counts(normalize=True).mul(100).round(1))
display(pd.DataFrame({'Clientes': gender_counts, 'Porcentaje': gender_share}))

fig, ax = plt.subplots()
sns.barplot(x=gender_counts.index, y=gender_counts.values, hue=gender_counts.index,
            palette={'Male': '#2563EB', 'Female': '#EC4899'}, legend=False, ax=ax)
for patch, n in zip(ax.patches, gender_counts.values):
    ax.text(patch.get_x() + patch.get_width()/2, n + 2, f'{n} ({n/len(df):.0%})',
            ha='center', fontweight='bold')
ax.set(title='Clientes por género', xlabel='Género', ylabel='Número de clientes', ylim=(0, 125))
plt.show()
"""),
    md("""
**Interpretación.** Hay **112 mujeres (56%)** y **88 hombres (44%)**. La muestra presenta una mayoría femenina moderada, pero ambos grupos tienen representación suficiente para comparaciones descriptivas.

## 7. ¿Qué grupos de edad gastan más en promedio?
"""),
    code("""
mean_spending_age = (df.groupby('Age Range', observed=True)['Spending Score (1-100)']
                     .mean().sort_values(ascending=False))
display(mean_spending_age.round(2).to_frame('Gasto promedio'))

fig, ax = plt.subplots()
sns.barplot(x=mean_spending_age.index, y=mean_spending_age.values,
            hue=mean_spending_age.index, palette='viridis', legend=False, ax=ax)
for patch, value in zip(ax.patches, mean_spending_age.values):
    ax.text(patch.get_x() + patch.get_width()/2, value + 1, f'{value:.1f}', ha='center')
ax.set(title='Gasto promedio por grupo de edad', xlabel='Rango de edad',
       ylabel='Puntuación de gasto promedio', ylim=(0, 70))
plt.show()
"""),
    md("""
**Interpretación.** El grupo de **30–39 años** gasta más en promedio (**61.10**), seguido por 18–29 (**58.58**). El menor promedio corresponde a 50–59 (**34.72**). Estas cifras describen grupos y no implican que todos sus integrantes se comporten igual.

## 8. ¿Hay alguna relación entre la edad y el ingreso anual de los clientes?
"""),
    code("""
age_income_corr = df['Age'].corr(df['Annual Income (k$)'])
print(f'Correlación de Pearson: {age_income_corr:.3f}')

fig, ax = plt.subplots()
sns.regplot(data=df, x='Age', y='Annual Income (k$)',
            scatter_kws={'alpha': .65, 's': 50, 'color': '#7C3AED'},
            line_kws={'color': '#DC2626'}, ax=ax)
ax.set(title='Edad e ingreso anual', xlabel='Edad', ylabel='Ingreso anual (k$)')
plt.show()
"""),
    md("""
**Interpretación.** La correlación es **−0.012**, esencialmente nula. En esta muestra la edad no permite anticipar el ingreso anual mediante una tendencia lineal.

## 9. ¿Cuál es la distribución conjunta de la edad y el ingreso anual?
"""),
    code("""
fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(df['Age'], df['Annual Income (k$)'], gridsize=15, cmap='Blues', mincnt=1)
fig.colorbar(hb, ax=ax, label='Número de clientes')
ax.set(title='Distribución conjunta de edad e ingreso', xlabel='Edad',
       ylabel='Ingreso anual (k$)')
plt.show()
"""),
    md("""
**Interpretación.** La mayor densidad aparece alrededor de edades adultas jóvenes y medias con ingresos intermedios. También hay clientes de distintas edades en casi todos los niveles de ingreso, coherente con la correlación cercana a cero. El hexágono permite detectar concentración sin ocultar puntos superpuestos.

## 10. ¿Cómo se distribuyen los clientes en función de la puntuación de gasto y el género?
"""),
    code("""
spending_gender = (df.groupby('Gender')['Spending Score (1-100)']
                   .agg(['count', 'mean', 'median', 'std']).round(2))
display(spending_gender)

fig, ax = plt.subplots()
sns.violinplot(data=df, x='Gender', y='Spending Score (1-100)', hue='Gender',
               palette={'Male': '#2563EB', 'Female': '#EC4899'}, inner='box',
               cut=0, legend=False, ax=ax)
ax.set(title='Puntuación de gasto por género', xlabel='Género',
       ylabel='Puntuación de gasto')
plt.show()
"""),
    md("""
**Interpretación.** Las mujeres tienen una puntuación media de **51.53** y los hombres de **48.51**; ambos comparten mediana de **50** y sus distribuciones se superponen ampliamente. El género por sí solo aporta poca separación conductual.

## Síntesis visual y conclusiones del EDA
"""),
    code("""
numeric = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df[numeric].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, ax=axes[0])
axes[0].set_title('Correlaciones lineales')
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Age Range', palette='viridis', s=55, alpha=.85, ax=axes[1])
axes[1].set(title='Ingreso y gasto por edad', xlabel='Ingreso anual (k$)',
            ylabel='Puntuación de gasto')
plt.tight_layout()
plt.show()
"""),
    md("""
### Conclusión del EDA

- La base está completa y limpia; no requiere imputación.
- Edad y gasto tienen una relación lineal negativa moderada, pero ingreso y gasto no siguen una sola relación lineal. En el plano ingreso–gasto aparecen zonas diferenciadas que justifican clustering.
- Para K-Means se usarán **edad, ingreso anual y puntuación de gasto**. Se excluirán `CustomerID` (identificador) y `Gender` (categórica y sin una separación clara de ingresos). Las tres variables numéricas deberán estandarizarse para que sus escalas no distorsionen las distancias.
"""),
])


clustering = notebook([
    md("""
# 2. Segmentación de clientes con K-Means — RetailMax

**Objetivo.** Descubrir segmentos accionables usando edad, ingreso anual y puntuación de gasto. K-Means es no supervisado: no existe una etiqueta correcta previa; la calidad se evalúa mediante cohesión/separación y utilidad comercial.
"""),
    code("""
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

sns.set_theme(style='whitegrid', context='notebook')
RANDOM_STATE = 42
df = pd.read_csv(Path('data/retailmax.csv'), dtype={'CustomerID': str})
feature_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df.head()
"""),
    md("""
## Selección de features

Se utilizan edad, ingreso anual y puntuación de gasto porque describen etapa de vida, poder adquisitivo y comportamiento de compra. `CustomerID` no contiene comportamiento y `Gender` es categórica; se deja fuera de la distancia inicial, aunque se recuperará al perfilar los resultados.
"""),
    code("""
features = df[feature_columns]
features.head()
"""),
    md("""
## Conservación de índices

Seleccionar columnas no reinicia los índices. Por ello, cada fila de `features` sigue correspondiendo al mismo cliente de `df`, lo que permite agregar las etiquetas de cluster sin perder alineación.
"""),
    code("""
display(df.sample(5, random_state=42))
display(features.sample(5, random_state=42))
assert df.index.equals(features.index)
"""),
    md("""
## Estandarización

K-Means usa distancias euclidianas. `StandardScaler` transforma cada feature para que tenga media 0 y desviación estándar poblacional 1, evitando que una escala numérica domine a las demás.
"""),
    code("""
assert features.notna().all().all(), 'Existen valores faltantes en las variables del modelo.'
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print(type(scaled_features))
print('Dimensiones:', scaled_features.shape)
display(scaled_features[0:3])
display(pd.DataFrame(scaled_features, columns=feature_columns).agg(['mean', 'std']).round(3))
"""),
    md("""
## Creación inicial de 3 clusters

Siguiendo la implementación indicada en la actividad, primero se ajusta K-Means con `k=3`, inicialización `k-means++`, un máximo de 300 iteraciones y 10 inicializaciones. `fit_predict` entrena el modelo y devuelve una etiqueta para cada uno de los 200 clientes.
"""),
    code("""
kmeans_3 = KMeans(n_clusters=3, init='k-means++', max_iter=300,
                  n_init=10, random_state=42)
clusters_3 = kmeans_3.fit_predict(scaled_features)
df['Cluster_k3'] = clusters_3

print('Número de etiquetas:', len(clusters_3))
display(df['Cluster_k3'].value_counts().sort_index().rename('Clientes').to_frame())

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster_k3', palette='Set1', s=70, alpha=.8, ax=ax)
ax.set(title='Implementación inicial de K-Means (k=3)', xlabel='Ingreso anual (k$)',
       ylabel='Puntuación de gasto')
plt.show()
"""),
    md("""
## Evaluación adicional del número de clusters

Se comparan valores de `k=2` a `k=10` con dos criterios complementarios:

- **Inercia:** suma de distancias cuadráticas dentro de los clusters; buscamos un “codo”. Siempre disminuye al aumentar `k`.
- **Silhouette:** combina cohesión y separación; valores mayores son mejores.
"""),
    code("""
ks = range(2, 11)
inertias, silhouettes = [], []

for k in ks:
    candidate = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = candidate.fit_predict(scaled_features)
    inertias.append(candidate.inertia_)
    silhouettes.append(silhouette_score(scaled_features, labels))

selection = pd.DataFrame({'k': list(ks), 'inercia': inertias, 'silhouette': silhouettes})
display(selection.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(selection['k'], selection['inercia'], marker='o', color='#2563EB')
axes[0].set(title='Método del codo', xlabel='Número de clusters (k)', ylabel='Inercia')
axes[1].plot(selection['k'], selection['silhouette'], marker='o', color='#16A34A')
axes[1].set(title='Coeficiente silhouette', xlabel='Número de clusters (k)', ylabel='Silhouette')
axes[1].axvline(6, color='#DC2626', ls='--', label='k seleccionado = 6')
axes[1].legend()
plt.tight_layout()
plt.show()
"""),
    md("""
**Decisión.** Se selecciona **k = 6**. Presenta el mayor silhouette de los candidatos (**0.427**) y mantiene tamaños interpretables. El codo ya muestra rendimientos decrecientes en esta zona. Aunque `k=5` también sería una opción comercial razonable, `k=6` separa clientes jóvenes de ingreso medio de los adultos mayores con gasto medio, añadiendo una distinción útil para comunicación y canal.

## Modelo final y visualización
"""),
    code("""
K = 6
model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
raw_labels = model.fit_predict(scaled_features)

# Nombres estables basados en los centroides, en vez de depender del número arbitrario de K-Means.
raw_centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=feature_columns)

def segment_name(row):
    income, spend, age = row['Annual Income (k$)'], row['Spending Score (1-100)'], row['Age']
    if income >= 70 and spend >= 55: return 'VIP de alto valor'
    if income >= 70 and spend < 55: return 'Alto ingreso, bajo gasto'
    if income < 40 and spend >= 50: return 'Jóvenes entusiastas'
    if income < 40 and spend < 50: return 'Maduros cautelosos'
    if age >= 45: return 'Tradicionales de gasto medio'
    return 'Jóvenes de gasto medio'

name_map = {i: segment_name(row) for i, row in raw_centers.iterrows()}
df['Cluster'] = raw_labels
df['Segment'] = df['Cluster'].map(name_map)
print(f'Silhouette del modelo final: {silhouette_score(scaled_features, raw_labels):.3f}')

palette = dict(zip(sorted(df['Segment'].unique()), sns.color_palette('tab10', K)))
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Segment', palette=palette, s=75, alpha=.85, ax=ax)
centers = raw_centers.copy()
centers['Segment'] = centers.index.map(name_map)
sns.scatterplot(data=centers, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Segment', palette=palette, marker='X', s=300, edgecolor='black',
                legend=False, ax=ax)
ax.set(title='Segmentos de clientes y centroides', xlabel='Ingreso anual (k$)',
       ylabel='Puntuación de gasto')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""),
    md("""
## Perfil de los segmentos

Los centroides son promedios, no clientes “ideales”. El porcentaje femenino se reporta para describir, pero género no participó en el ajuste.
"""),
    code("""
profile = (df.groupby('Segment')
           .agg(Clientes=('CustomerID', 'size'),
                Edad_media=('Age', 'mean'),
                Ingreso_medio_k=('Annual Income (k$)', 'mean'),
                Gasto_medio=('Spending Score (1-100)', 'mean'),
                Porcentaje_mujeres=('Gender', lambda s: s.eq('Female').mean() * 100))
           .assign(Porcentaje_clientes=lambda x: x['Clientes'] / len(df) * 100)
           .sort_values('Ingreso_medio_k', ascending=False))
display(profile.round(1))

long_profile = (profile[['Edad_media', 'Ingreso_medio_k', 'Gasto_medio']]
                .rename(columns={'Edad_media': 'Edad', 'Ingreso_medio_k': 'Ingreso',
                                 'Gasto_medio': 'Gasto'})
                .apply(lambda col: (col - col.mean()) / col.std())
                .reset_index().melt(id_vars='Segment', var_name='Variable', value_name='z'))
plt.figure(figsize=(10, 5))
sns.barplot(data=long_profile, x='Segment', y='z', hue='Variable')
plt.axhline(0, color='black', lw=1)
plt.title('Perfil relativo de cada segmento')
plt.xlabel('')
plt.ylabel('Diferencia respecto al promedio (z)')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()
"""),
    md("""
## Recomendaciones de marketing

| Segmento | Tamaño | Rasgo distintivo | Acción sugerida |
|---|---:|---|---|
| **VIP de alto valor** | 39 (19.5%) | 32.7 años, 86.5 k$ de ingreso, gasto 82.1 | Programa VIP, acceso anticipado, bundles premium y recompensas por recomendación. Prioridad alta para retención. |
| **Alto ingreso, bajo gasto** | 33 (16.5%) | 41.9 años, mayor ingreso (88.9 k$), gasto 17.0 | Investigar barreras con encuesta/A-B test; ofertas personalizadas y demostraciones de valor, evitando descuentos indiscriminados. |
| **Jóvenes entusiastas** | 24 (12.0%) | 25.2 años, ingreso bajo (25.8 k$), gasto alto (76.9) | Fidelización móvil, promociones asequibles, referidos y productos de entrada. Vigilar fatiga promocional. |
| **Maduros cautelosos** | 21 (10.5%) | 45.5 años, ingreso bajo (26.3 k$), gasto 19.4 | Campañas de valor, descuentos selectivos, productos esenciales y mensajes de ahorro. |
| **Tradicionales de gasto medio** | 45 (22.5%) | 56.3 años, ingreso 54.3 k$, gasto 49.1 | Comunicación simple, servicio/beneficios de confianza y venta cruzada moderada. |
| **Jóvenes de gasto medio** | 38 (19.0%) | 26.7 años, ingreso 57.6 k$, gasto 47.8 | Nutrición digital, recomendaciones personalizadas y pruebas para elevar frecuencia y ticket. |

## Validación, límites y siguiente paso
"""),
    code("""
sample_silhouette = silhouette_samples(scaled_features, raw_labels)
validation = (pd.DataFrame({'Segment': df['Segment'], 'silhouette_individual': sample_silhouette})
              .groupby('Segment')['silhouette_individual']
              .agg(['mean', 'min', 'max']).round(3))
display(validation)

export_columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
                  'Spending Score (1-100)', 'Cluster', 'Segment']
df[export_columns].to_csv('data/retailmax_segmentado.csv', index=False)
print('Archivo generado: data/retailmax_segmentado.csv')
"""),
    md("""
### Conclusión

K-Means revela seis perfiles diferenciados y accionables. La principal oportunidad inmediata es retener a los **VIP de alto valor** y convertir, mediante experimentos, al grupo de **alto ingreso y bajo gasto**. Los segmentos no deben tratarse como verdades permanentes: dependen de sólo tres variables, K-Means presupone grupos aproximadamente compactos y la puntuación de gasto es una métrica interna.

Antes de desplegar campañas se recomienda: (1) medir respuesta con pruebas A/B y un grupo de control, (2) añadir datos de recencia, frecuencia, valor monetario, canal y categorías, (3) reentrenar y monitorear tamaños/centroides periódicamente, y (4) evitar decisiones sensibles basadas en género o edad. El éxito debe medirse con conversión, ingreso incremental, margen, retención y bajas de comunicación, no sólo con silhouette.
"""),
])


nbf.write(eda, ROOT / '1_EDA.ipynb')
nbf.write(clustering, ROOT / '2_clustering.ipynb')
