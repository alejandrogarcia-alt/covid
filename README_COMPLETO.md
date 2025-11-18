# 🩺 Sistema de Visión Artificial para Detección de COVID-19

## 📋 Descripción del Proyecto

Sistema de **Computer-Aided Diagnosis (CAD)** basado en Deep Learning para asistir en la detección de COVID-19 a partir de radiografías de tórax. Utiliza Transfer Learning con MobileNetV2 y técnicas de interpretabilidad (Grad-CAM) para proporcionar predicciones explicables.

**Institución:** Centro de Diagnóstico por Imágenes
**Diplomatura:** Inteligencia Artificial - Universidad Tecnológica Nacional (UTN)

### 👥 Autores
- Pablo Salera
- Alejandro García
- Mirta Beatriz Arce
- Mariano Buonifacino
- Vanesa Galvagno

---

## 🎯 Objetivos del Sistema

### Beneficios Esperados
- ✅ Reducción del tiempo de revisión preliminar
- ✅ Estandarización de criterios visuales
- ✅ Mejora del flujo de trabajo interno
- ✅ Apoyo confiable al diagnóstico médico
- ✅ Priorización de casos sospechosos

### ⚠️ Consideraciones Importantes
- **NO reemplaza** el diagnóstico médico profesional
- Requiere **supervisión médica permanente**
- Debe complementarse con pruebas PCR/antígeno
- Función exclusiva de **apoyo al diagnóstico (CAD)**

---

## 🏗️ Arquitectura del Sistema

### Backend (Deep Learning)
- **Modelo Base:** MobileNetV2 (Transfer Learning from ImageNet)
- **Clasificación:** Binaria (Normal vs COVID-Compatible)
- **Estrategia:** Transfer Learning con Fine-Tuning en 2 fases
- **Data Augmentation:** Rotación, zoom, flip, contraste
- **Framework:** TensorFlow/Keras

### Frontend (Interfaz Web)
**Streamlit** con 3 secciones principales:

#### 📊 Sección 1: Configuración de Entrenamiento
- Ajuste de hiperparámetros (epochs, learning rate, batch size, dropout)
- Configuración de callbacks (early stopping, reduce LR)
- Inicio del proceso de entrenamiento
- Visualización en tiempo real

#### 📈 Sección 2: Dashboard de Métricas con Gemini AI
- Visualización de métricas (Accuracy, Precision, Recall, F1, AUC)
- Matriz de confusión
- Curvas ROC y Precision-Recall
- **Análisis inteligente con Gemini AI** que proporciona:
  - Evaluación del rendimiento del modelo
  - Identificación de problemas (overfitting, underfitting)
  - Sugerencias de mejora de hiperparámetros
  - Recomendaciones para optimización

#### 🔬 Sección 3: Diagnóstico con Grad-CAM
- Carga de radiografías para predicción
- Visualización del resultado (Normal / COVID-Compatible)
- **Mapa de calor Grad-CAM** que muestra las regiones de atención del modelo
- Interpretación clínica y recomendaciones
- Sistema auditable y explicable

---

## 📁 Estructura del Proyecto

```
COVID IA/
├── app/
│   ├── app.py                    # App Streamlit original (básica)
│   └── app_complete.py           # App Streamlit completa (3 secciones)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py     # Carga y preparación de datos
│   ├── model.py                  # Definición del modelo CNN
│   ├── train.py                  # Script de entrenamiento original
│   ├── train_configurable.py    # Script de entrenamiento configurable
│   ├── predict.py                # Predicción con Grad-CAM
│   ├── metrics.py                # Métricas avanzadas (CM, ROC, AUC)
│   └── gemini_analyzer.py        # Integración con Gemini AI
│
├── data/                         # (vacío, datasets se cargan de /Downloads)
├── notebooks/                    # Notebooks de experimentación
├── requirements.txt              # Dependencias del proyecto
└── README_COMPLETO.md           # Este archivo
```

---

## 🚀 Instalación y Configuración

### 1. Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 4GB+ RAM (8GB+ recomendado)
- GPU opcional (CUDA compatible) para entrenamiento más rápido

### 2. Clonar o Navegar al Proyecto

```bash
cd "/Users/amgarcia71/Development/COVID IA"
```

### 3. Crear Entorno Virtual (Recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- tensorflow
- streamlit
- google-generativeai (Gemini AI)
- opencv-python
- matplotlib
- seaborn
- plotly
- scikit-learn
- pandas
- pillow

### 5. Verificar Dataset

El dataset debe estar en: `/Users/amgarcia71/Downloads/Dataset/`

Estructura esperada:
```
Dataset/
├── COVID/         # ~1000 imágenes de casos COVID
└── Normal/        # ~1000 imágenes de casos normales
```

---

## 🎮 Uso del Sistema

### Opción 1: Aplicación Completa (Recomendada)

```bash
streamlit run app/app_complete.py
```

Esto abrirá la aplicación en tu navegador con las 3 secciones completas.

### Opción 2: Aplicación Básica Original

```bash
streamlit run app/app.py
```

### Opción 3: Entrenamiento desde CLI

```bash
python src/train_configurable.py \
  --initial_epochs 10 \
  --fine_tune_epochs 10 \
  --initial_lr 0.001 \
  --fine_tune_lr 0.0001 \
  --batch_size 32 \
  --dropout 0.2
```

---

## 📖 Guía de Uso Paso a Paso

### Paso 1: Entrenar el Modelo

1. Abre la aplicación: `streamlit run app/app_complete.py`
2. Ve a la sección **"⚙️ Entrenamiento"**
3. Ajusta los hiperparámetros según tus necesidades:
   - **Épocas de Extracción:** 10 (recomendado para pruebas)
   - **Épocas de Fine-Tuning:** 10
   - **Learning Rate Inicial:** 0.001
   - **Learning Rate Fine-Tuning:** 0.0001
   - **Batch Size:** 32
   - **Dropout:** 0.2
4. Haz clic en **"🚀 Iniciar Entrenamiento"**
5. Espera a que el entrenamiento complete (puede tomar 30-60 minutos)

### Paso 2: Analizar Métricas con Gemini AI

1. Ve a la sección **"📈 Métricas y Análisis"**
2. Revisa las visualizaciones:
   - Historial de entrenamiento (accuracy/loss)
   - Matriz de confusión
   - Curva ROC
   - Resumen de métricas
3. Para obtener análisis con IA:
   - Ingresa tu **Google Gemini API Key** ([Obtener aquí](https://makersuite.google.com/app/apikey))
   - Haz clic en **"🤖 Generar Análisis con Gemini AI"**
4. Revisa las recomendaciones:
   - Resumen ejecutivo
   - Análisis del historial de entrenamiento
   - Análisis de métricas de evaluación
   - Sugerencias de hiperparámetros

### Paso 3: Realizar Diagnósticos

1. Ve a la sección **"🔬 Diagnóstico"**
2. Carga una radiografía de tórax (PNG, JPG, JPEG)
3. El sistema mostrará:
   - **Predicción:** Normal o COVID-Compatible
   - **Nivel de confianza:** Porcentaje
   - **Mapa de calor Grad-CAM:** Regiones donde el modelo se enfocó
4. Interpreta el resultado con criterio médico profesional

---

## 📊 Métricas del Sistema

### Métricas de Clasificación
- **Accuracy:** Precisión general del modelo
- **Precision:** De los casos predichos como COVID, cuántos son realmente COVID
- **Recall (Sensibilidad):** De los casos reales de COVID, cuántos detecta el modelo
- **F1-Score:** Media armónica entre Precision y Recall
- **AUC-ROC:** Área bajo la curva ROC (rendimiento general)

### Matriz de Confusión
- **Verdaderos Positivos (TP):** COVID correctamente identificado
- **Verdaderos Negativos (TN):** Normal correctamente identificado
- **Falsos Positivos (FP):** Normal predicho como COVID
- **Falsos Negativos (FN):** COVID predicho como Normal ⚠️ **MÁS PELIGROSO**

---

## 🤖 Integración con Gemini AI

### Configuración de API Key

1. Obtén tu API key en: https://makersuite.google.com/app/apikey
2. Ingresa la API key en la sección "📈 Métricas y Análisis"
3. El sistema utilizará Gemini para:
   - Analizar curvas de entrenamiento
   - Interpretar métricas de evaluación
   - Detectar overfitting/underfitting
   - Sugerir ajustes de hiperparámetros
   - Generar recomendaciones clínicas

### Tipos de Análisis Disponibles

#### 1. Análisis del Historial de Entrenamiento
- Evaluación de convergencia
- Detección de sobreajuste
- Recomendaciones de epochs y learning rate

#### 2. Análisis de Métricas de Evaluación
- Interpretación clínica de métricas
- Balance entre Sensibilidad y Especificidad
- Evaluación de seguridad para uso clínico

#### 3. Análisis Visual de Gráficas
- Interpretación de patrones en visualizaciones
- Identificación de problemas en curvas
- Análisis de distribución de predicciones

#### 4. Sugerencias de Hiperparámetros
- Valores optimizados para cada parámetro
- Justificación técnica de cambios
- Impacto esperado de ajustes

---

## 🔬 Grad-CAM (Interpretabilidad)

### ¿Qué es Grad-CAM?

**Gradient-weighted Class Activation Mapping** es una técnica que visualiza las regiones de la imagen que el modelo utilizó para tomar su decisión.

### Interpretación del Mapa de Calor

- 🔴 **Zonas Rojas/Calientes:** Alta importancia en la decisión
- 🟡 **Zonas Amarillas:** Importancia moderada
- 🔵 **Zonas Azules/Frías:** Baja importancia

### Utilidad Clínica

1. **Auditabilidad:** El médico puede verificar si las regiones destacadas son clínicamente relevantes
2. **Detección de sesgos:** Identifica si el modelo se enfoca en artefactos en lugar de patología
3. **Confianza:** Aumenta la confianza en predicciones cuando las regiones coinciden con hallazgos clínicos
4. **Educación:** Ayuda a entender qué patrones visuales aprende el modelo

---

## ⚠️ Limitaciones y Consideraciones

### Limitaciones Técnicas
- Entrenado solo con datasets públicos (puede tener sesgo)
- Sensible a la calidad de la imagen
- No diagnostica otras patologías pulmonares
- Requiere radiografías PA o AP de buena calidad

### Limitaciones Clínicas
- **NO es un diagnóstico definitivo**
- Radiografía puede ser normal en fases tempranas de COVID-19
- Falsos negativos son posibles y peligrosos
- Debe complementarse con PCR/antígeno
- Requiere interpretación médica profesional

### Consideraciones Éticas
- Supervisión médica obligatoria
- No reemplaza el juicio clínico
- Privacidad de datos del paciente
- Transparencia en las limitaciones
- Consentimiento informado del paciente

---

## 🛠️ Solución de Problemas

### Error: "Model file not found"
**Solución:** Entrena un modelo primero en la sección "⚙️ Entrenamiento"

### Error: "Dataset not found"
**Solución:** Verifica que el dataset esté en `/Users/amgarcia71/Downloads/Dataset/` con las carpetas `COVID/` y `Normal/`

### Error: Gemini API
**Solución:**
- Verifica que tu API key sea válida
- Asegúrate de tener créditos disponibles en tu cuenta de Google
- Revisa la conexión a internet

### Entrenamiento muy lento
**Solución:**
- Reduce el batch size (ej: de 32 a 16)
- Reduce el número de epochs
- Usa una GPU si está disponible
- Cierra otras aplicaciones

### Out of Memory
**Solución:**
- Reduce el batch size (ej: de 32 a 16 o 8)
- Cierra otras aplicaciones
- Usa un modelo más pequeño (MobileNetV2 ya es ligero)

---

## 📚 Referencias y Recursos

### Datasets Públicos Utilizados
- COVID-19 Radiography Database
- COVIDx Dataset

### Frameworks y Librerías
- TensorFlow/Keras: https://www.tensorflow.org/
- Streamlit: https://streamlit.io/
- Google Gemini AI: https://ai.google.dev/

### Papers de Referencia
- Transfer Learning for Medical Image Classification
- Grad-CAM: Visual Explanations from Deep Networks
- COVID-19 Detection from Chest X-rays

---

## 📞 Soporte y Contribuciones

Este es un proyecto académico desarrollado como Trabajo Integrador Final para la Diplomatura en Inteligencia Artificial de la UTN.

### Contacto
Para consultas académicas o técnicas, contactar a los autores del proyecto.

---

## 📄 Licencia

Este proyecto es de uso académico y educativo. No está aprobado para uso clínico en producción sin las validaciones y aprobaciones regulatorias correspondientes.

---

## ✅ Checklist de Implementación

- [x] Modelo de Deep Learning con Transfer Learning
- [x] Data Augmentation
- [x] Entrenamiento en 2 fases (Feature Extraction + Fine-Tuning)
- [x] Métricas avanzadas (Confusion Matrix, ROC, AUC)
- [x] Grad-CAM para interpretabilidad
- [x] Frontend con Streamlit (3 secciones)
- [x] Configuración de hiperparámetros
- [x] Dashboard de métricas
- [x] Integración con Gemini AI
- [x] Sistema de diagnóstico con mapas de calor
- [x] Documentación completa

---

**🎓 Diplomatura en Inteligencia Artificial - Universidad Tecnológica Nacional (UTN)**

*Sistema desarrollado con fines educativos y de investigación.*
