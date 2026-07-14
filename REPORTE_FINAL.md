# Reporte final — Segmentación de clientes de RetailMax

## Resumen ejecutivo

Se analizaron 200 clientes sin valores faltantes ni duplicados. Para segmentarlos se aplicó K-Means a edad, ingreso anual y puntuación de gasto, después de estandarizar las variables. Primero se implementó el modelo de tres clusters solicitado en la actividad. Como extensión analítica, se comparó `k=2…10` y se eligieron seis clusters para el modelo comercial final porque `k=6` obtuvo el mejor coeficiente silhouette (0.427) y produjo perfiles útiles para marketing.

Los dos frentes con mayor potencial son: retener a 39 clientes VIP de alto valor y entender/activar a 33 clientes con alto ingreso pero bajo gasto. Las campañas deben validarse con pruebas A/B y métricas de negocio; la segmentación es una hipótesis accionable, no una relación causal.

## Hallazgos del análisis exploratorio

- La edad media es 38.85 años y la mediana 36; el rango va de 18 a 70 años.
- El ingreso medio masculino (62.23 k$) supera descriptivamente al femenino (59.25 k$), pero la diferencia no es estadísticamente clara (Mann–Whitney, p=0.414).
- Los clientes de 18–39 años tienen mayor gasto medio y mayor variabilidad. Después de los 40 años el gasto tiende a disminuir, con excepciones individuales.
- Las mujeres representan 56% de la muestra y los hombres 44%; la mediana de gasto es 50 en ambos grupos.
- La correlación ingreso–gasto es 0.010 y la correlación edad–ingreso es −0.012: no hay tendencias lineales globales apreciables.
- A pesar de esa correlación nula, ingreso y gasto forman zonas visibles con dispersión muy distinta por rango de ingreso, por lo que una única campaña para toda la base perdería información comercial importante.

## Metodología

Se excluyó `CustomerID` por ser sólo un identificador y `Gender` por ser categórica y no aportar una separación descriptiva clara. Se comprobó que el DataFrame de features conserva los índices originales y se estandarizaron las tres variables numéricas para evitar que sus escalas dominaran la distancia euclidiana. La implementación inicial usa exactamente `k=3`, `init='k-means++'`, `max_iter=300`, `n_init=10` y `random_state=42`. Después se comparó `k=2…10` mediante inercia y silhouette; `k=6` alcanzó el mayor silhouette (0.427) y se utilizó como modelo final ampliado.

## Segmentos y acciones

| Segmento | Clientes | Perfil promedio | Recomendación |
|---|---:|---|---|
| VIP de alto valor | 39 | 32.7 años; 86.5 k$; gasto 82.1 | Retención VIP, acceso anticipado y recompensas por recomendación. |
| Alto ingreso, bajo gasto | 33 | 41.9 años; 88.9 k$; gasto 17.0 | Investigar barreras y probar propuestas de valor personalizadas. |
| Jóvenes entusiastas | 24 | 25.2 años; 25.8 k$; gasto 76.9 | Fidelización móvil, referidos y productos de entrada. |
| Maduros cautelosos | 21 | 45.5 años; 26.3 k$; gasto 19.4 | Mensajes de ahorro, esenciales y descuentos selectivos. |
| Tradicionales de gasto medio | 45 | 56.3 años; 54.3 k$; gasto 49.1 | Confianza, servicio y venta cruzada moderada. |
| Jóvenes de gasto medio | 38 | 26.7 años; 57.6 k$; gasto 47.8 | Recomendaciones digitales para elevar frecuencia y ticket. |

## Plan de implementación y medición

1. Priorizar VIP y alto ingreso/bajo gasto por su potencial económico.
2. Dividir aleatoriamente cada segmento entre tratamiento y control; no comparar segmentos entre sí como si fueran equivalentes.
3. Medir conversión, ingreso y margen incremental, ticket, retención y tasa de baja de comunicaciones.
4. Incorporar recencia, frecuencia, valor monetario, canal y categoría en la siguiente versión.
5. Reentrenar periódicamente y vigilar cambios en tamaños, centroides y desempeño de campañas.

## Limitaciones

El análisis usa una muestra pequeña y sólo tres variables conductuales/demográficas. K-Means favorece grupos compactos y obliga a asignar cada cliente a un solo cluster. Silhouette mide estructura geométrica, no rentabilidad. Edad y género deben utilizarse con cuidado para evitar trato discriminatorio. Las recomendaciones son hipótesis que requieren experimentación antes de una adopción amplia.

## Entregables reproducibles

- `1_EDA.ipynb`: calidad de datos, visualizaciones y respuestas a las tres preguntas.
- `2_clustering.ipynb`: preparación, selección de k, modelo, perfiles, recomendaciones y exportación.
- `data/retailmax_segmentado.csv`: asignación de segmento por cliente, generada al ejecutar el segundo notebook.
