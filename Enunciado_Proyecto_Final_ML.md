# PROYECTO FINAL - Machine Learning
## Reconocimiento de Actividad Humana con Machine Learning (HAR Dataset)

* **Modalidad:** Trabajo en equipo (grupos de 3)
* **Duración:** 4 semanas (3 entregas + presentación)
* **Entregable principal:** Jupyter Notebook + presentación

---

## 1. Visión General del Problema

El reconocimiento de actividad humana (HAR, por sus siglas en inglés) es un problema fundamental en el campo del aprendizaje automático aplicado a sistemas de salud, deporte y dispositivos móviles. El objetivo es determinar automáticamente qué actividad está realizando una persona a partir de datos de sensores inerciales, sin necesidad de observación directa.

En este proyecto trabajarán con el dataset UCI HAR (Human Activity Recognition Using Smartphones), un conjunto de datos ampliamente estudiado que ha servido como benchmark en la literatura de ML. Los datos fueron recolectados de 30 voluntarios (edades 19-48) que realizaron seis actividades cotidianas mientras portaban un smartphone en la cintura. A partir de las señales del acelerómetro y giroscopio se extrajeron 561 características en el dominio del tiempo y la frecuencia.

### 1.1 Dataset: UCI HAR

| Atributo | Detalle |
| --- | --- |
| **Fuente** | UCI Machine Learning Repository / PhysioNet |
| **Sujetos** | 30 voluntarios, edades 19-48 años |
| **Clases (actividades)** | 6: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| **Muestras totales** | 10,299 (7,352 entrenamiento / 2,947 prueba) |
| **Features** | 561 variables numéricas, ya extraídas y normalizadas [-1, 1] |
| **Dispositivo** | Samsung Galaxy S II (acelerómetro + giroscopio, 50 Hz) |
| **Descarga** | [archive.ics.uci.edu/dataset/240](https://archive.ics.uci.edu/dataset/240) |

### 1.2 Objetivo del Proyecto

Desarrollar un pipeline completo de Machine Learning que, dado el conjunto de features del dataset HAR, sea capaz de clasificar con precisión la actividad que realiza un sujeto. El trabajo debe estar completamente documentado en Jupyter Notebooks y culmina con un video explicativo donde el equipo explica sus decisiones y resultados.

> [!NOTE]
> **Pregunta central del proyecto:** Dado un conjunto de 561 features extraídas de sensores inerciales, ¿cuál es la actividad que realiza el sujeto? ¿Qué modelo clasifica mejor y por qué?

### 1.3 Estructura General

El proyecto se divide en tres fases iterativas. Cada fase tiene su propio notebook de entrega y construye sobre la anterior. Las rúbricas de evaluación son documentos separados que el profesor entregará al inicio de cada fase.

| Fase | Nombre | Entregable | Semana |
| :---: | --- | --- | :---: |
| **1** | Preprocesamiento y Visualización | Notebook 1 | Semana 2 |
| **2** | Modelado y Benchmarking | Notebook 2 | Semana 3 |
| **3** | Resultados y Storytelling | Notebook 3 + Slides + Video | Semana 4 |
| **Bono** | Interpretabilidad del Modelo | Sección adicional en Notebook 3 | Semana 4 |

### 1.4 Equipos y Reglas

* Equipos de 3 integrantes (elección propia).
* Cada equipo trabaja sobre el mismo dataset; las diferencias estarán en las decisiones metodológicas.
* Los notebooks deben poder ejecutarse de forma reproducible: incluir seed fijo (`random_state=42`), instalación de dependencias y rutas relativas.
* El código debe estar comentado y limpio. Celdas markdown deben explicar cada sección.
* Prohibido copiar código de otros equipos. El uso de fuentes externas debe citarse.
* Entrega a través de aula virtual.

---

## 2. Fase 1: Preprocesamiento y Visualización

* **Objetivo:** Comprender el dataset HAR, preparar los datos para el modelado y comunicar los hallazgos exploratorios mediante visualizaciones.
* **Entregable:** `notebook_fase1_grupoXX.ipynb` (un notebook por equipo)

### 2.1 Descarga y Carga del Dataset

El dataset está disponible en UCI ML Repository. El notebook base incluye el código de descarga. Asegúrense de que los paths sean relativos para que el notebook corra en cualquier máquina.

**Actividades:**
1. Descargar el dataset usando el código provisto en el notebook base.
2. Cargar los archivos `X_train.txt`, `y_train.txt`, `X_test.txt`, `y_test.txt` y `features.txt`.
3. Asignar los nombres de columnas usando `features.txt`.
4. Convertir las etiquetas numéricas a nombres de actividad: `{1: WALKING, 2: WALKING_UPSTAIRS, 3: WALKING_DOWNSTAIRS, 4: SITTING, 5: STANDING, 6: LAYING}`.

> [!TIP]
> **Punto de partida en el notebook:** El notebook base tiene celdas marcadas con `# TODO`. Deben completar esas celdas y agregar celdas markdown que expliquen sus observaciones.

### 2.2 Inspección Inicial del Dataset

Antes de cualquier procesamiento, deben responder preguntas básicas sobre la estructura del dataset.

**Actividades:**
5. Mostrar las dimensiones del dataset de entrenamiento y prueba (`df.shape`).
6. Mostrar los primeros 5 registros y los tipos de datos de cada columna.
7. Verificar si existen valores faltantes (NaN). Si no los hay, explicar por qué (el dataset ya fue preprocesado).
8. Verificar si existen valores duplicados en el conjunto de entrenamiento.
9. Calcular estadísticos descriptivos básicos: media, desviación estándar, min, max por feature (`df.describe()`).

> [!NOTE]
> En una celda markdown, respondan: ¿Qué observan sobre el rango de los valores? ¿Necesitan normalizar? ¿Por qué?

### 2.3 Análisis de Balance de Clases

El desbalance de clases es un problema crítico en clasificación. Antes de modelar, deben entender la distribución de las actividades.

**Actividades:**
10. Contar el número de muestras por clase en el set de entrenamiento.
11. Graficar la distribución de clases con un gráfico de barras. Incluir etiquetas con el conteo y porcentaje de cada clase.
12. Evaluar si el dataset está balanceado. Justificar con los números.
13. En markdown: ¿Cómo podría afectar el desbalance al modelo? ¿Qué métrica sería más adecuada que la accuracy simple?

### 2.4 Visualización y Exploración de Features

Con 561 features, es imposible visualizarlas todas individualmente. El objetivo es identificar patrones y features relevantes usando técnicas de visualización inteligente.

#### 2.4.1 Distribución de features clave
14. El dataset agrupa las features por dominio: tiempo (t) y frecuencia (f), y por fuente: cuerpo (Body) y gravedad (Gravity). Seleccionar al menos 4 features representativas (una por grupo) y graficar su distribución por clase usando boxplots.
15. Analizar: ¿qué features separan mejor las clases? ¿Cuáles son menos informativas?

#### 2.4.2 Mapa de calor de correlación
16. Calcular la matriz de correlación entre features. Dado el alto número de features, seleccionar las 20 features con mayor varianza o las primeras 30 features.
17. Visualizar con un heatmap (`seaborn.heatmap`). Identificar grupos de features altamente correlacionadas.
18. En markdown: ¿qué implicaciones tiene la alta correlación entre features para algunos modelos?

#### 2.4.3 Reducción de dimensionalidad para visualización
19. Aplicar PCA (Principal Component Analysis) para reducir a 2 dimensiones.
20. Graficar el espacio PCA coloreando por clase de actividad. Usar una paleta de colores distinguible.
21. Interpretar: ¿las actividades están bien separadas en el espacio PCA? ¿Cuáles se solapan?

### 2.5 Preparación Final de Datos

**Actividades:**
22. Confirmar que el split train/test ya viene predefinido en el dataset (no deben crear su propio split aleatorio). Explicar en markdown por qué esto es importante.
23. Exportar `X_train`, `X_test`, `y_train`, `y_test` como variables limpias para usar en la Fase 2.
24. Crear una celda resumen al final del notebook que imprima: dimensiones finales, número de clases, nombres de clases y si hay valores faltantes.

### 2.6 Entregable de la Fase 1

**Checklist de entrega:**
* Notebook ejecutable de principio a fin sin errores.
* Celdas markdown con análisis escrito en cada sección.
* Mínimo 5 visualizaciones: distribución de clases, boxplots, heatmap, PCA, y una adicional a elección.
* Celda de resumen final.
* Nombrar el archivo: `notebook_fase1_grupoXX.ipynb`

| Criterio | Peso | Descripción |
| --- | :---: | --- |
| **Carga y descripción del dataset** | 20% | Carga correcta, tipos de datos, dimensiones, nulos, duplicados. |
| **Análisis de balance de clases** | 20% | Conteo, gráfico, discusión del desbalance. |
| **Visualizaciones exploratorias** | 35% | Calidad e informatividad de los gráficos (boxplot, heatmap, PCA). Mínimo 5 visualizaciones. |
| **Preparación de datos** | 10% | Split correcto, variables limpias, celda de resumen. |
| **Calidad del análisis escrito** | 15% | Celdas markdown con interpretaciones coherentes y relevantes. |

---

## 3. Fase 2: Modelado y Benchmarking

* **Objetivo:** Entrenar múltiples modelos de clasificación, evaluarlos con métricas adecuadas y construir una tabla de benchmarking que permita comparar su desempeño de forma rigurosa.
* **Entregable:** `notebook_fase2_grupoXX.ipynb` (continúa desde Fase 1)

### 3.1 Métricas de Evaluación

Dado que el problema tiene 6 clases, la accuracy sola no es suficiente. Deben evaluar cada modelo con las siguientes métricas:

* **Accuracy global:** porcentaje total de muestras correctamente clasificadas.
* **F1-Score macro:** promedio del F1 de cada clase sin ponderar por frecuencia.
* **F1-Score weighted:** promedio del F1 ponderado por el número de muestras de cada clase.
* **Matriz de confusión:** tabla de predicciones vs. clase real.
* **Reporte de clasificación:** precisión, recall y F1 por clase (`sklearn.metrics.classification_report`).

> [!IMPORTANT]
> Siempre reportar métricas sobre el conjunto de prueba (`X_test`, `y_test`). Nunca tomar decisiones finales sobre el set de entrenamiento.

### 3.2 Modelo Baseline

Antes de entrenar modelos reales, se debe establecer un baseline: un modelo trivial que sirva como piso mínimo de comparación. Si un modelo 'inteligente' no supera al baseline, algo está mal.

**Actividades:**
25. Entrenar un `DummyClassifier` con estrategia `'most_frequent'`. Reportar accuracy y F1-macro.
26. Calcular la accuracy de un clasificador que simplemente predice la clase más frecuente. Confirmar que coincide con el `DummyClassifier`.
27. En markdown: ¿qué accuracy base esperarían superar? ¿Por qué es importante este paso?

### 3.3 Modelos Clásicos

Entrenar los siguientes modelos usando los parámetros por defecto de sklearn (no optimizar aún). El objetivo es obtener una primera medición comparativa.

* **Modelo 1:** Logistic Regression
* **Modelo 2:** K-Nearest Neighbors (KNN, k=5)
* **Modelo 3:** Decision Tree
* **Modelo 4:** Random Forest
* **Modelo 5:** Support Vector Machine (SVM con kernel RBF)

**Para cada modelo:**
28. Entrenar con `X_train`, `y_train` (usar `random_state=42` donde aplique).
29. Predecir sobre `X_test`.
30. Calcular: accuracy, F1-macro, F1-weighted, reporte de clasificación.
31. Graficar la matriz de confusión (usar `ConfusionMatrixDisplay`).

### 3.4 Tabla de Benchmarking

Una vez entrenados todos los modelos, construir una tabla comparativa. Esta es la pieza central de la Fase 2.

**Actividades:**
32. Crear un DataFrame de pandas con los resultados de todos los modelos.
33. Columnas: `Modelo`, `Accuracy_Test`, `F1_Macro`, `F1_Weighted`, `Tiempo_Entrenamiento` (usar `time.time()`).
34. Ordenar la tabla de mayor a menor `F1-Macro`.
35. Graficar un barplot comparativo de `F1-Macro` por modelo.
36. En markdown: identificar el mejor modelo y justificar la elección. Considerar no sólo la métrica sino también el tiempo de entrenamiento.

### 3.5 Validación Cruzada

La evaluación sobre un solo split test puede ser ruidosa. Usar cross-validation para obtener una estimación más robusta del desempeño del mejor modelo.

**Actividades:**
37. Aplicar 5-fold cross-validation sobre `X_train` del mejor modelo seleccionado.
38. Reportar la media y desviación estándar del F1-macro a través de los folds.
39. En markdown: ¿el desempeño en CV es consistente con el desempeño en test? ¿Hay sobreajuste?

### 3.6 Análisis de Actividades Difíciles

No todas las actividades son igual de fáciles de clasificar. Este análisis es clave para entender las limitaciones del modelo.

**Actividades:**
40. Extraer del reporte de clasificación del mejor modelo el F1 por clase.
41. Identificar las dos actividades con menor F1. Hipotetizar por qué son más difíciles (pista: miren la matriz de confusión).
42. En markdown: ¿qué tienen en común físicamente las actividades que se confunden?

### 3.7 Entregable de la Fase 2

**Checklist de entrega:**
* Baseline implementado y reportado.
* Mínimo 5 modelos entrenados con métricas completas.
* Tabla de benchmarking en DataFrame de pandas.
* Matrices de confusión de todos los modelos.
* Cross-validation del mejor modelo.
* Análisis de actividades difíciles.
* Nombrar el archivo: `notebook_fase2_grupoXX.ipynb`

| Criterio | Peso | Descripción |
| --- | :---: | --- |
| **Baseline implementado** | 10% | DummyClassifier correcto con métricas reportadas. |
| **Modelos entrenados correctamente** | 25% | Los 5 modelos implementados, métricas calculadas en test set. |
| **Tabla de benchmarking** | 25% | DataFrame completo, ordenado, con tiempo de entrenamiento y gráfico comparativo. |
| **Cross-validation del mejor modelo** | 20% | CV de 5 folds, media y std reportadas, análisis de sobreajuste. |
| **Análisis de actividades difíciles** | 10% | Identificación correcta de actividades con bajo F1 e hipótesis fundamentada. |
| **Calidad del análisis escrito** | 10% | Justificaciones coherentes en markdown para cada decisión. |

---

## 4. Fase 3: Resultados y Storytelling

* **Objetivo:** Comunicar los resultados del proyecto de forma clara y coherente. Esta fase integra todo el trabajo anterior y pone énfasis en la narrativa: ¿qué aprendieron? ¿qué recomendarían? ¿qué limitaciones tiene el modelo?
* **Entregable:** `notebook_fase3_grupoXX.ipynb` + slides (PDF) + video (5 min)

### 4.1 Notebook Final

El notebook de la Fase 3 es el documento final del proyecto. Debe integrar y resumir el trabajo de las tres fases con una narrativa clara.

**Estructura requerida del notebook:**
43. Portada: título, integrantes del equipo, fecha.
44. Introducción: descripción del problema HAR y del dataset. ¿Por qué es un problema relevante?
45. Resumen de la Fase 1: principales hallazgos del EDA. Incluir las visualizaciones más relevantes (no todas).
46. Resumen de la Fase 2: tabla de benchmarking final. Modelo seleccionado y justificación.
47. Análisis de errores: ¿qué actividades confunde más el modelo? ¿Tiene sentido físico esta confusión?
48. Limitaciones del modelo: ¿en qué escenarios podría fallar? ¿Qué mejoras podrían implementarse?
49. Conclusión: resumen de los aprendizajes principales del proyecto.

### 4.2 Presentación (Slides)

La presentación está dirigida a un público técnico (sus compañeros de clase y el profesor). Deben ser capaces de explicar el problema, la metodología y los resultados en 10 minutos, seguido de 5 minutos de preguntas.

**Estructura sugerida (10-12 slides):**
* **Slide 1:** Título y equipo.
* **Slide 2:** El problema HAR. ¿Por qué es importante?
* **Slide 3-4:** Descripción del dataset y hallazgos del EDA.
* **Slide 5-6:** Modelos probados y tabla de benchmarking.
* **Slide 7:** Análisis de errores (matriz de confusión del mejor modelo).
* **Slide 8:** Limitaciones y mejoras posibles.
* **Slide 9:** Conclusiones.
* **Slide 10:** ¿Preguntas?

**Tips para la presentación:**
* No lean las slides. Usen las slides como apoyo visual.
* Cada integrante debe presentar al menos una sección.
* Eviten slides de texto puro: usen gráficos y tablas de sus notebooks.

### 4.3 Video (5 minutos)

Grabar un video corto que explique el proyecto. El video puede ser una grabación de pantalla con voz en off mientras muestran el notebook o las slides.

**Requisitos:**
* **Duración:** 4 a 6 minutos (no menos, no más).
* Deben aparecer todos los integrantes (al menos con voz).
* **Cubrir:** contexto del problema, principales hallazgos del EDA, modelo seleccionado y por qué, resultado final.
* No es necesario producción profesional, pero debe ser claro y audible.
* **Formato de entrega:** MP4, o link a YouTube/Drive.

### 4.4 Entregable de la Fase 3

**Checklist de entrega:**
* `notebook_fase3_grupoXX.ipynb`: ejecutable, con estructura completa.
* `slides_grupoXX.pdf`: 10-12 slides, estructura requerida.
* `video_grupoXX` (link o archivo MP4): 4-6 minutos.
* Todo en una carpeta comprimida: `proyecto_ml_grupoXX.zip`

| Criterio | Peso | Descripción |
| --- | :---: | --- |
| **Notebook final completo** | 30% | Estructura requerida, narrativa coherente, análisis de errores y limitaciones. |
| **Calidad de la presentación** | 25% | Slides claras, gráficos informativos, flujo lógico. |
| **Presentación oral** | 25% | Claridad, manejo de preguntas, participación de todos los integrantes. |
| **Video** | 20% | Cumple duración, cubre contenido requerido, se entiende claramente. |

---

## 5. Bono: Interpretabilidad del Modelo

* **Puntaje adicional:** hasta +1.0 punto sobre la nota final del proyecto.

Entrenar un modelo de ML y saber que predice bien no es suficiente. Un aspecto crítico en aplicaciones reales (especialmente en salud) es entender POR QUÉ el modelo toma sus decisiones. Esta sección bono introduce técnicas de interpretabilidad.

### 5.1 Importancia de Features

Algunos modelos (como Random Forest) proveen directamente una medida de importancia de cada feature. Esta técnica responde: ¿qué variables usa más el modelo para tomar decisiones?

**Actividades:**
50. Extraer feature importances del modelo Random Forest entrenado en la Fase 2 (`.feature_importances_`).
51. Graficar un barplot horizontal con las 20 features más importantes.
52. Analizar: ¿qué tipo de feature (tiempo vs. frecuencia, acelerómetro vs. giroscopio) es más relevante?

### 5.2 Valores SHAP

SHAP (SHapley Additive exPlanations) es una técnica de interpretabilidad modelo-agnóstica que explica la contribución de cada feature a la predicción de una muestra específica. Es el estándar actual de la industria para explicabilidad en ML.

**Instalación requerida:**
```bash
pip install shap
```
```python
import shap
```

**Actividades:**
53. Instalar e importar la librería `shap`.
54. Crear un explicador SHAP para el mejor modelo. Para Random Forest: `shap.TreeExplainer(model)`. Para otros modelos: `shap.KernelExplainer` (más lento, usar muestra pequeña).
55. Calcular los valores SHAP para el set de prueba (o una muestra de 200 observaciones para mayor velocidad).
56. Graficar el SHAP summary plot: muestra las features más influyentes globalmente.
57. Analizar al menos un caso específico: elegir una muestra del test set y graficar su SHAP waterfall plot. Explicar por qué el modelo tomó la decisión que tomó para esa muestra.

### 5.3 Análisis de Interpretabilidad

En una celda markdown final del bono, responder las siguientes preguntas:
* ¿Las features más importantes tienen sentido físico para el problema HAR? ¿Cuáles son y por qué creen que son relevantes?
* ¿El modelo usa features de tiempo, frecuencia, o ambas?
* ¿Encontraron algún hallazgo sorprendente o contra-intuitivo?
* Si tuvieran que explicar este modelo a un médico o kinesiólogo, ¿cómo lo harían basándose en el análisis SHAP?

| Criterio | Peso | Descripción |
| --- | :---: | --- |
| **Feature importance implementada y analizada** | 40% | Gráfico correcto, top-20 features, análisis por tipo de feature. |
| **SHAP values calculados y graficados** | 40% | Summary plot correcto, waterfall plot de un caso específico. |
| **Análisis de interpretabilidad escrito** | 20% | Respuestas a las 4 preguntas, con profundidad y coherencia. |

---

## 6. Cronograma y Fechas Clave

| Semana | Actividad en clase | Tarea del equipo | Entrega |
| --- | --- | --- | --- |
| **Semana 1** | Presentación del proyecto. Explicación del dataset HAR. | Descargar el dataset. Revisar el enunciado. Distribuir el trabajo en el equipo. | — |
| **Semana 2** | Clase sobre EDA y preprocesamiento. | Desarrollar Fase 1 (EDA, visualizaciones, preparación de datos). | Notebook Fase 1 |
| **Semana 3** | Clase sobre modelos y benchmarking. | Desarrollar Fase 2 (baseline, modelos, benchmarking, CV). | Notebook Fase 2 |
| **Semana 4** | Presentaciones orales (15 min por grupo). | Desarrollar Fase 3 (notebook final, slides, video). | Notebook F3 + Slides + Video |

### 6.1 Ponderación del Proyecto

El proyecto final tiene la siguiente ponderación sobre la nota del curso (a definir por el profesor):

| Componente | Peso | Entrega |
| --- | :---: | --- |
| **Fase 1: Preprocesamiento y Visualización** | 25% | Semana 2 |
| **Fase 2: Modelado y Benchmarking** | 35% | Semana 3 |
| **Fase 3: Resultados y Storytelling** | 40% | Semana 4 |
| **Bono: Interpretabilidad** | +hasta 1.0 pto | Semana 4 (integrado en Fase 3) |
