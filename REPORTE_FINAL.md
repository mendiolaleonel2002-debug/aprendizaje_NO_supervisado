# Reporte de recomendaciones para el equipo de marketing de RetailMax

## Resumen ejecutivo

El presente reporte traduce los resultados del análisis exploratorio y del modelo K-Means en acciones concretas para el equipo de marketing de RetailMax. Se analizaron 200 clientes considerando edad, ingreso anual y puntuación de gasto. Después de estandarizar estas variables y comparar distintas cantidades de grupos, el método del codo mostró que **cinco clusters** ofrecen un equilibrio adecuado entre detalle e interpretabilidad. La segmentación resultante permite abandonar una estrategia uniforme y diseñar comunicaciones más relevantes para cada perfil.

La principal recomendación es concentrar los primeros experimentos en dos segmentos: los clientes de ingreso y gasto altos, por su valor actual, y los clientes de ingreso alto pero gasto bajo, por su potencial de crecimiento. Las campañas deben evaluarse mediante grupos de control y métricas incrementales; pertenecer a un cluster no demuestra que una campaña causará una compra.

## Metodología y hallazgos generales

El conjunto de datos no presenta valores faltantes ni registros duplicados. La edad media es 38.85 años y el ingreso anual promedio es 60.56 mil dólares. Las correlaciones ingreso–gasto (0.010) y edad–ingreso (−0.012) son prácticamente nulas. Sin embargo, la visualización conjunta revela perfiles diferenciados que una correlación lineal no captura.

Para K-Means se excluyó `CustomerID`, porque sólo identifica clientes, y se dejaron inicialmente fuera las categorías de género. Las variables numéricas fueron transformadas con `StandardScaler` para que edad, ingreso y gasto tuvieran una contribución comparable en las distancias. Se probaron modelos con 3, 5 y 10 clusters y se calculó el WCSS entre 1 y 10. El cambio de pendiente alrededor de cinco justificó seleccionar **k=5** para las recomendaciones siguientes.

## Recomendaciones por segmento

### Cluster 0 — Clientes cautelosos de bajo valor actual

Este grupo reúne 20 clientes con edad media de **46.25 años**, ingreso anual medio de **26.75 k$** y puntuación de gasto de **18.35**. Se recomienda utilizar mensajes centrados en ahorro, productos esenciales, cupones con monto mínimo y beneficios fáciles de comprender. No conviene invertir inicialmente en descuentos agresivos, pues el ingreso disponible y el gasto observado son bajos. El objetivo debe ser aumentar gradualmente la frecuencia sin reducir excesivamente el margen.

### Cluster 1 — Jóvenes entusiastas

Los 54 integrantes tienen una edad media de **25.19 años**, ingreso de **41.09 k$** y gasto de **62.24**. Aunque su poder adquisitivo es moderado, muestran buena disposición de compra. RetailMax puede ofrecer programas de referidos, recompensas móviles, productos de entrada, paquetes asequibles y contenido para redes sociales. Es importante controlar la presión promocional y evitar incentivar consumo poco responsable.

### Cluster 2 — Clientes VIP de alto valor

Este segmento contiene 40 clientes con edad media de **32.88 años**, ingreso de **86.10 k$** y la mayor puntuación de gasto: **81.53**. Debe recibir prioridad de retención mediante acceso anticipado, atención preferente, recomendaciones premium, recompensas por lealtad y experiencias exclusivas. El éxito no debe medirse sólo por ventas inmediatas, sino también mediante retención, margen, frecuencia y valor de vida del cliente.

### Cluster 3 — Alto ingreso y bajo gasto

Los 39 clientes del grupo tienen edad media de **39.87 años**, ingreso de **86.10 k$** y gasto de apenas **19.36**. Representan la mayor oportunidad de activación, pero no debe asumirse que responderán a descuentos. Se recomienda investigar barreras mediante encuestas breves y pruebas A/B de propuestas distintas: conveniencia, surtido premium, entrega, garantía o servicio personalizado. Las campañas deben explicar valor y relevancia antes de reducir precios.

### Cluster 4 — Clientes maduros de gasto medio

Este es un grupo de 47 clientes con edad media de **55.64 años**, ingreso de **54.38 k$** y gasto de **48.85**. Se sugieren comunicaciones claras, beneficios de confianza, servicio posventa, recordatorios oportunos y venta cruzada moderada. Puede funcionar una combinación de canales digitales y tradicionales, permitiendo que el cliente elija su medio preferido.

## Plan de implementación y medición

RetailMax debería ejecutar campañas piloto dentro de cada segmento, asignando aleatoriamente clientes a tratamiento y control. Las métricas principales serán conversión incremental, ingreso y margen incremental, ticket promedio, frecuencia, retención y tasa de baja de comunicaciones. Comparar únicamente el desempeño bruto entre clusters sería incorrecto, porque parten de comportamientos diferentes.

Se recomienda comenzar con los clusters 2 y 3: proteger el valor existente del grupo VIP y experimentar con la activación del grupo de alto ingreso y bajo gasto. Después podrán desplegarse campañas de fidelización para jóvenes entusiastas y estrategias de frecuencia para los segmentos 0 y 4.

## Limitaciones y uso responsable

La segmentación utiliza una muestra pequeña y sólo tres variables. K-Means obliga a asignar cada cliente a un único grupo y favorece clusters compactos. Además, la puntuación de gasto es una medida interna, no una estimación directa de rentabilidad. En una siguiente versión conviene incorporar recencia, frecuencia, valor monetario, canal, categorías compradas y respuesta histórica a campañas.

La edad y el género deben manejarse con cuidado. Al incorporar género como variable estandarizada, el modelo tiende a formar grupos separados por sexo, lo que puede generar campañas poco útiles o discriminatorias. Se recomienda utilizar género sólo para auditar resultados y equidad, no para restringir ofertas. Finalmente, los clusters deben recalcularse periódicamente y validarse con resultados comerciales reales.

## Entregables reproducibles

- `1_EDA.ipynb`: análisis de calidad, diez preguntas exploratorias, visualizaciones e interpretaciones.
- `2_clustering.ipynb`: escalamiento, modelos K-Means, método del codo, perfiles y recomendaciones.
- `data/retailmax_segmentado.csv`: asignación reproducible de segmentos por cliente.
