# Descriviendo categorias vino

Muchas veces nos toca trabajar con datos que han sido segmentados en categorías, esto puede deberse por la implementación de una metodología tipo Kmens, segmentación de la variable respuesta, por criterio experto, etc., independiente de su origen, cabe la pregunta ¿Qué atributos presenta cada una de las categorías que las hace única? 
A lo largo de este repositorio encontrarán la implementación de distintas técnicas para caracterizar cada una de las categorías existentes en un set de datos.

Syntax original: https://towardsdatascience.com/best-practices-for-visualizing-your-cluster-results-20a3baac7426


# Distribución de atributos según categoría mediante Boxplots:
Una forma muy simple de analizar los datos es a través del uso de boxplot, a continuación, se muestra la distribución de cada variable según categoría.
[![Figure-1.png](https://i.postimg.cc/QdmDmn1t/Figure-1.png)](https://postimg.cc/zyy4X7dZ)

De los resultados anteriores es posible observar que variables como los sulfatos y el alcohol tienden a aumentar a medida que incrementa la categoría (calidad/quality) del vino.

# Desviación porcentual según categoría:
Otra forma de analizar las categorías es a través de la desviación porcentual de cada variable con respecto a la media global. A continuación, se muestra los resultados:
[![Figure-2.png](https://i.postimg.cc/5t767SDB/Figure-2.png)](https://postimg.cc/949mzZm0)

De los resultados anteriores es posible observar que existen categorias en las cuales las variables alcanzan a desviarse en un 50% en relación al promedio global.


# Gráfico de radar
Otra forma propuesta para analizar los datos es a través del gráfico de radar, que nos permite visualizar qué tan representada estaba cierta variable en cada categoría.
[![Figure-3.png](https://i.postimg.cc/SNnRR92R/Figure-3.png)](https://postimg.cc/hfqc3f0R)

De los resultados, es posible observar que cada categoría tiene una distribución de atributos unica.

