"""
Complete COVID-19 Detection System - Streamlit Application
Includes 3 sections as specified in the project requirements:
1. Training Configuration
2. Metrics Dashboard with Gemini Analysis
3. Diagnostic Testing with Grad-CAM
"""

import streamlit as st
import os
import sys
import json
import subprocess
import threading
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import glob

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict_and_visualize
from train_configurable import TrainingConfig, train_model
from gemini_analyzer import GeminiAnalyzer


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="COVID-19 Detection System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("<h1 style='text-align: center; color: #1f77b4;'>🩺 COVID-19<br/>Detection System</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegación",
    ["📊 Panel de Control", "⚙️ Entrenamiento", "📈 Métricas y Análisis", "🔬 Diagnóstico"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='info-box'>
<b>Sistema CAD</b><br/>
Computer-Aided Diagnosis<br/>
<small>Herramienta de apoyo al diagnóstico médico</small>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>
<b>Autores:</b> Pablo Salera, Alejandro García, Mirta Beatriz Arce,
Mariano Buonifacino, Vanesa Galvagno
<br/><br/>
<b>Institución:</b> Centro de Diagnóstico por Imágenes
<br/><br/>
<b>Diplomatura:</b> Inteligencia Artificial - UTN
</small>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_training_results():
    """Get the most recent training results directory."""
    results_dirs = glob.glob('training_results_*')
    if not results_dirs:
        return None
    return max(results_dirs, key=os.path.getmtime)


def load_json_file(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {filepath}: {str(e)}")
        return None


# ============================================================================
# PAGE 1: DASHBOARD / CONTROL PANEL
# ============================================================================

if page == "📊 Panel de Control":
    st.markdown("<div class='main-header'>🩺 Sistema de Detección de COVID-19</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <h3>Bienvenido al Sistema de Visión Artificial para Detección de COVID-19</h3>
    <p>
    Este sistema utiliza <b>Deep Learning</b> con <b>Transfer Learning</b> para asistir en la
    detección de patrones compatibles con COVID-19 en radiografías de tórax.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 Resumen del Sistema</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-label'>ARQUITECTURA</div>
        <div class='metric-value'>MobileNetV2</div>
        <small>Transfer Learning con ImageNet</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-label'>CLASIFICACIÓN</div>
        <div class='metric-value'>Binaria</div>
        <small>Normal vs COVID-Compatible</small>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
        <div class='metric-label'>INTERPRETABILIDAD</div>
        <div class='metric-value'>Grad-CAM</div>
        <small>Mapas de calor explicativos</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🎯 Objetivos del Sistema</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### ✅ Beneficios Esperados
        - Reducción del tiempo de revisión preliminar
        - Estandarización de criterios visuales
        - Mejora del flujo de trabajo interno
        - Apoyo confiable al diagnóstico médico
        - Priorización de casos sospechosos
        """)

    with col2:
        st.markdown("""
        #### ⚠️ Consideraciones Importantes
        - **NO reemplaza** el diagnóstico médico profesional
        - Requiere **supervisión médica permanente**
        - Debe complementarse con pruebas PCR/antígeno
        - Función exclusiva de **apoyo al diagnóstico (CAD)**
        - Sujeto a validación y monitoreo continuo
        """)

    st.markdown("<div class='section-header'>🔧 Estado Actual del Sistema</div>", unsafe_allow_html=True)

    # Check if model exists
    model_files = glob.glob('training_results_*/final_model.h5')

    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        latest_dir = os.path.dirname(latest_model)

        st.success(f"✅ Modelo entrenado encontrado: `{latest_model}`")

        # Try to load metrics
        metrics_file = os.path.join(latest_dir, 'metrics_summary.json')
        if os.path.exists(metrics_file):
            metrics = load_json_file(metrics_file)
            if metrics:
                st.markdown("#### 📊 Métricas del Último Modelo")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                with col5:
                    st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
    else:
        st.warning("⚠️ No se encontró ningún modelo entrenado. Ve a la sección '⚙️ Entrenamiento' para entrenar un modelo.")

    st.markdown("<div class='section-header'>🚀 Comenzar</div>", unsafe_allow_html=True)

    st.markdown("""
    Utiliza el menú de navegación lateral para:
    1. **⚙️ Entrenamiento**: Configurar hiperparámetros y entrenar el modelo
    2. **📈 Métricas y Análisis**: Visualizar métricas y obtener análisis con Gemini AI
    3. **🔬 Diagnóstico**: Realizar predicciones en radiografías con mapas de calor explicativos
    """)


# ============================================================================
# PAGE 2: TRAINING CONFIGURATION (Section 1)
# ============================================================================

elif page == "⚙️ Entrenamiento":
    st.markdown("<div class='main-header'>⚙️ Configuración de Entrenamiento</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>Sección 1: Configuración de Hiperparámetros</b><br/>
    Ajusta los parámetros de entrenamiento según tus necesidades y recursos disponibles.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📝 Configuración de Hiperparámetros</div>", unsafe_allow_html=True)

    with st.form("training_config_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔄 Épocas de Entrenamiento")
            initial_epochs = st.number_input(
                "Épocas de Extracción de Características",
                min_value=1, max_value=100, value=10, step=1,
                help="Número de épocas para la fase inicial (cabeza congelada)"
            )

            fine_tune_epochs = st.number_input(
                "Épocas de Fine-Tuning",
                min_value=0, max_value=100, value=10, step=1,
                help="Número de épocas para fine-tuning (descongelar capas)"
            )

            st.markdown("#### 📊 Learning Rates")
            initial_lr = st.number_input(
                "Learning Rate Inicial",
                min_value=0.00001, max_value=0.1, value=0.001, step=0.0001, format="%.5f",
                help="Tasa de aprendizaje para la fase inicial"
            )

            fine_tune_lr = st.number_input(
                "Learning Rate Fine-Tuning",
                min_value=0.000001, max_value=0.01, value=0.0001, step=0.00001, format="%.6f",
                help="Tasa de aprendizaje para fine-tuning (debe ser más baja)"
            )

        with col2:
            st.markdown("#### 🎛️ Parámetros del Modelo")
            batch_size = st.selectbox(
                "Batch Size",
                options=[8, 16, 32, 64],
                index=2,
                help="Tamaño del lote (afecta velocidad y memoria)"
            )

            dropout_rate = st.slider(
                "Dropout Rate",
                min_value=0.0, max_value=0.7, value=0.2, step=0.05,
                help="Tasa de dropout para regularización (0.0-0.7)"
            )

            st.markdown("#### ⏱️ Callbacks y Early Stopping")
            early_stopping_patience = st.number_input(
                "Early Stopping Patience",
                min_value=1, max_value=20, value=5, step=1,
                help="Épocas sin mejora antes de detener el entrenamiento"
            )

            reduce_lr_patience = st.number_input(
                "Reduce LR Patience",
                min_value=1, max_value=10, value=3, step=1,
                help="Épocas sin mejora antes de reducir el learning rate"
            )

        submit_button = st.form_submit_button("🚀 Iniciar Entrenamiento", use_container_width=True)

    if submit_button:
        st.markdown("<div class='section-header'>🔄 Proceso de Entrenamiento</div>", unsafe_allow_html=True)

        # Create configuration
        config = TrainingConfig(
            initial_epochs=initial_epochs,
            fine_tune_epochs=fine_tune_epochs,
            initial_learning_rate=initial_lr,
            fine_tune_learning_rate=fine_tune_lr,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience
        )

        # Display configuration
        st.json(config.to_dict())

        st.info("⏳ Iniciando entrenamiento... Este proceso puede tomar varios minutos u horas dependiendo de la configuración.")

        # Create a progress container
        progress_container = st.empty()

        with st.spinner("Entrenando modelo..."):
            try:
                # Train model
                model, history, metrics, save_dir = train_model(config)

                st.success(f"✅ Entrenamiento completado exitosamente!")
                st.success(f"📁 Resultados guardados en: `{save_dir}`")

                # Display quick metrics
                st.markdown("#### 📊 Resultados Rápidos")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                with col4:
                    st.metric("AUC", f"{metrics.get('auc', 0):.3f}")

                st.balloons()

                st.info("💡 Ve a la sección '📈 Métricas y Análisis' para ver el análisis completo y las recomendaciones de Gemini AI.")

            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                st.exception(e)


# ============================================================================
# PAGE 3: METRICS DASHBOARD WITH GEMINI ANALYSIS (Section 2)
# ============================================================================

elif page == "📈 Métricas y Análisis":
    st.markdown("<div class='main-header'>📈 Dashboard de Métricas y Análisis IA</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>Sección 2: Visualización de Métricas y Análisis con Gemini AI</b><br/>
    Visualiza las métricas de entrenamiento y evaluación, y obtén recomendaciones inteligentes
    para mejorar el modelo usando Gemini AI.
    </div>
    """, unsafe_allow_html=True)

    # Get latest training results
    latest_dir = get_latest_training_results()

    if not latest_dir:
        st.warning("⚠️ No se encontraron resultados de entrenamiento. Primero entrena un modelo en la sección '⚙️ Entrenamiento'.")
    else:
        st.success(f"📁 Cargando resultados de: `{latest_dir}`")

        # Load training history
        history_file = os.path.join(latest_dir, 'training_history.json')
        metrics_file = os.path.join(latest_dir, 'metrics_summary.json')
        config_file = os.path.join(latest_dir, 'training_config.json')

        history_data = load_json_file(history_file)
        metrics_data = load_json_file(metrics_file)
        config_data = load_json_file(config_file)

        # Display configuration used
        with st.expander("⚙️ Ver Configuración de Entrenamiento Utilizada"):
            if config_data:
                st.json(config_data)

        st.markdown("<div class='section-header'>📊 Métricas de Rendimiento</div>", unsafe_allow_html=True)

        if metrics_data:
            # Display main metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                accuracy = metrics_data.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy:.3f}")

            with col2:
                precision = metrics_data.get('precision', 0)
                st.metric("Precision", f"{precision:.3f}")

            with col3:
                recall = metrics_data.get('recall', 0)
                st.metric("Recall (Sensibilidad)", f"{recall:.3f}")

            with col4:
                f1 = metrics_data.get('f1_score', 0)
                st.metric("F1-Score", f"{f1:.3f}")

            with col5:
                auc = metrics_data.get('auc', 0)
                st.metric("AUC", f"{auc:.3f}")

        st.markdown("<div class='section-header'>📈 Visualizaciones</div>", unsafe_allow_html=True)

        # Display visualizations
        viz_files = {
            'Historial de Entrenamiento': 'training_history.png',
            'Resumen de Métricas': 'metrics_summary.png',
            'Matriz de Confusión': 'confusion_matrix.png',
            'Curva ROC': 'roc_curve.png',
            'Curva Precision-Recall': 'precision_recall_curve.png'
        }

        tabs = st.tabs(list(viz_files.keys()))

        for tab, (title, filename) in zip(tabs, viz_files.items()):
            with tab:
                img_path = os.path.join(latest_dir, filename)
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=title, use_container_width=True)
                else:
                    st.warning(f"Imagen no encontrada: {filename}")

        st.markdown("<div class='section-header'>🤖 Análisis con Gemini AI</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='warning-box'>
        <b>⚠️ Configuración Requerida</b><br/>
        Para utilizar el análisis con Gemini AI, necesitas una API key de Google.
        Ingresa tu API key a continuación.
        </div>
        """, unsafe_allow_html=True)

        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Obtén tu API key en https://makersuite.google.com/app/apikey"
        )

        if st.button("🤖 Generar Análisis con Gemini AI", use_container_width=True):
            if not gemini_api_key:
                st.error("❌ Por favor, ingresa tu API key de Gemini.")
            else:
                with st.spinner("🤖 Gemini AI está analizando las métricas..."):
                    try:
                        analyzer = GeminiAnalyzer(api_key=gemini_api_key)

                        # Collect image paths
                        image_paths = [
                            os.path.join(latest_dir, filename)
                            for filename in viz_files.values()
                            if os.path.exists(os.path.join(latest_dir, filename))
                        ]

                        # Perform comprehensive analysis
                        analysis_results = analyzer.comprehensive_analysis(
                            history_dict=history_data if history_data else {},
                            metrics_dict=metrics_data if metrics_data else {},
                            image_paths=image_paths,
                            hyperparams=config_data if config_data else {}
                        )

                        # Generate executive summary
                        executive_summary = analyzer.generate_executive_summary(analysis_results)

                        # Display results
                        st.markdown("<div class='section-header'>📋 Resumen Ejecutivo</div>", unsafe_allow_html=True)
                        st.markdown(executive_summary)

                        # Display detailed analyses in expanders
                        with st.expander("📊 Análisis del Historial de Entrenamiento"):
                            st.markdown(analysis_results.get('training_analysis', 'No disponible'))

                        with st.expander("📈 Análisis de Métricas de Evaluación"):
                            st.markdown(analysis_results.get('evaluation_analysis', 'No disponible'))

                        with st.expander("🎨 Análisis Visual de Gráficas"):
                            st.markdown(analysis_results.get('visual_analysis', 'No disponible'))

                        with st.expander("⚙️ Sugerencias de Hiperparámetros"):
                            st.markdown(analysis_results.get('hyperparameter_suggestions', 'No disponible'))

                        st.success("✅ Análisis completado exitosamente!")

                    except Exception as e:
                        st.error(f"❌ Error al generar análisis con Gemini: {str(e)}")
                        st.exception(e)


# ============================================================================
# PAGE 4: DIAGNOSTIC TESTING WITH GRAD-CAM (Section 3)
# ============================================================================

elif page == "🔬 Diagnóstico":
    st.markdown("<div class='main-header'>🔬 Diagnóstico con Radiografías</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>Sección 3: Test y Diagnóstico con Grad-CAM</b><br/>
    Carga una radiografía de tórax para obtener una predicción del modelo y visualizar
    el mapa de calor (Grad-CAM) que muestra las regiones en las que el modelo se enfocó para tomar su decisión.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box'>
    <b>⚠️ ADVERTENCIA MÉDICA IMPORTANTE</b><br/>
    - Este sistema es una <b>herramienta de apoyo al diagnóstico (CAD)</b>, NO un sistema de diagnóstico definitivo<br/>
    - Las predicciones deben ser <b>validadas por un radiólogo certificado</b><br/>
    - Debe complementarse con pruebas virales (PCR, test de antígeno) y evaluación clínica<br/>
    - NO utilizar como única base para decisiones médicas
    </div>
    """, unsafe_allow_html=True)

    # Find latest model
    model_files = glob.glob('training_results_*/final_model.h5')

    if not model_files:
        st.error("❌ No se encontró ningún modelo entrenado. Primero entrena un modelo en la sección '⚙️ Entrenamiento'.")
    else:
        latest_model = max(model_files, key=os.path.getmtime)
        st.success(f"✅ Modelo cargado: `{latest_model}`")

        st.markdown("<div class='section-header'>📤 Cargar Radiografía</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Selecciona una imagen de radiografía de tórax (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:
            # Save uploaded file
            temp_image_path = "temp_xray.png"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display uploaded image
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 🖼️ Imagen Original")
                original_img = Image.open(uploaded_file)
                st.image(original_img, caption="Radiografía Cargada", use_container_width=True)

            with st.spinner("🔬 Analizando radiografía..."):
                try:
                    # Make prediction
                    predicted_class, confidence, heatmap_img = predict_and_visualize(
                        temp_image_path,
                        latest_model
                    )

                    with col2:
                        st.markdown("#### 🔥 Mapa de Calor (Grad-CAM)")
                        st.image(heatmap_img, caption="Atención del Modelo", use_container_width=True)

                    st.markdown("<div class='section-header'>📋 Resultado del Análisis</div>", unsafe_allow_html=True)

                    # Display prediction result
                    if predicted_class == "COVID":
                        st.markdown(f"""
                        <div class='warning-box' style='border-left: 5px solid #dc3545;'>
                        <h3 style='color: #dc3545;'>⚠️ COMPATIBLE CON COVID-19</h3>
                        <p style='font-size: 1.2rem;'>
                        <b>Nivel de Confianza:</b> {confidence:.1%}
                        </p>
                        <p>
                        El modelo ha identificado patrones compatibles con COVID-19 en la radiografía.
                        </p>
                        <p style='margin-top: 1rem; padding: 0.5rem; background-color: #fff; border-radius: 5px;'>
                        <b>ACCIÓN REQUERIDA:</b><br/>
                        1. Revisar inmediatamente con un radiólogo certificado<br/>
                        2. Realizar prueba PCR o test de antígeno<br/>
                        3. Evaluar sintomatología clínica del paciente<br/>
                        4. Considerar aislamiento preventivo según protocolos
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='success-box'>
                        <h3 style='color: #28a745;'>✅ RADIOGRAFÍA NORMAL</h3>
                        <p style='font-size: 1.2rem;'>
                        <b>Nivel de Confianza:</b> {confidence:.1%}
                        </p>
                        <p>
                        El modelo no ha identificado patrones significativos compatibles con COVID-19.
                        </p>
                        <p style='margin-top: 1rem; padding: 0.5rem; background-color: #fff; border-radius: 5px;'>
                        <b>CONSIDERACIONES:</b><br/>
                        1. Este resultado NO descarta COVID-19 definitivamente<br/>
                        2. La radiografía puede ser normal en fases tempranas<br/>
                        3. Considerar prueba PCR si hay sintomatología<br/>
                        4. Evaluación clínica integral del paciente es fundamental
                        </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Interpretation of heatmap
                    st.markdown("<div class='section-header'>🔍 Interpretación del Mapa de Calor</div>", unsafe_allow_html=True)

                    st.markdown("""
                    <div class='info-box'>
                    <h4>¿Qué muestra el mapa de calor (Grad-CAM)?</h4>
                    <p>
                    El mapa de calor visualiza las <b>regiones de la radiografía</b> en las que el modelo
                    de Deep Learning se enfocó para tomar su decisión de clasificación.
                    </p>
                    <ul>
                    <li><b style='color: red;'>Zonas Rojas/Calientes:</b> Alta importancia en la decisión del modelo</li>
                    <li><b style='color: yellow;'>Zonas Amarillas:</b> Importancia moderada</li>
                    <li><b style='color: blue;'>Zonas Azules/Frías:</b> Baja importancia</li>
                    </ul>
                    <p>
                    Este mapa permite al radiólogo <b>auditar</b> la decisión del modelo y verificar si
                    las regiones destacadas corresponden con hallazgos clínicos relevantes.
                    </p>
                    <p style='margin-top: 1rem;'>
                    <b>⚠️ Importante:</b> Un mapa de calor concentrado en regiones anatómicamente irrelevantes
                    puede indicar un error del modelo o un sesgo en los datos de entrenamiento.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional metrics if available
                    latest_dir = get_latest_training_results()
                    if latest_dir:
                        metrics_file = os.path.join(latest_dir, 'metrics_summary.json')
                        if os.path.exists(metrics_file):
                            metrics = load_json_file(metrics_file)
                            if metrics:
                                st.markdown("<div class='section-header'>📊 Rendimiento del Modelo Actual</div>", unsafe_allow_html=True)

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Sensibilidad (Recall)", f"{metrics.get('recall', 0):.1%}",
                                            help="Capacidad de detectar casos COVID positivos")
                                with col2:
                                    st.metric("Especificidad", "N/A",
                                            help="Capacidad de identificar casos normales")
                                with col3:
                                    st.metric("Precisión", f"{metrics.get('precision', 0):.1%}",
                                            help="De los casos predichos como COVID, cuántos son realmente COVID")
                                with col4:
                                    st.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}",
                                            help="Rendimiento general del clasificador")

                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                except Exception as e:
                    st.error(f"❌ Error al procesar la imagen: {str(e)}")
                    st.exception(e)

        else:
            st.info("👆 Por favor, carga una imagen de radiografía de tórax para comenzar el análisis.")

            # Show example/instructions
            with st.expander("💡 Instrucciones y Recomendaciones"):
                st.markdown("""
                ### Formato de Imagen Recomendado
                - **Tipo:** Radiografía de tórax (PA o AP)
                - **Formato:** PNG, JPG, JPEG
                - **Calidad:** Alta resolución preferiblemente
                - **Vista:** Posteroanterior (PA) o Anteroposterior (AP)

                ### Limitaciones del Sistema
                - Solo detecta patrones en radiografías de tórax
                - No diagnostica otras patologías pulmonares
                - Sensible a la calidad de la imagen
                - Puede tener falsos positivos o negativos

                ### Interpretación Clínica
                - Siempre validar con un radiólogo certificado
                - Complementar con historia clínica del paciente
                - Realizar pruebas confirmatorias (PCR/antígeno)
                - Considerar el contexto epidemiológico
                """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>
<p>
<b>Sistema CAD - Computer-Aided Diagnosis</b><br/>
Herramienta de apoyo al diagnóstico médico para detección de COVID-19 en radiografías de tórax<br/>
<small>Diplomatura en Inteligencia Artificial - Universidad Tecnológica Nacional (UTN)</small>
</p>
</div>
""", unsafe_allow_html=True)
