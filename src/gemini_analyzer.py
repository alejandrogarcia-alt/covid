"""
Gemini AI Analyzer module for COVID-19 detection model.
Uses Google's Gemini API to analyze training metrics and provide recommendations.
"""

import google.generativeai as genai
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from PIL import Image


class GeminiAnalyzer:
    """
    Analyzes model training metrics and evaluation results using Gemini AI.
    Provides intelligent recommendations for model improvement.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini Analyzer.

        Args:
            api_key: Google API key for Gemini. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please provide it or set GEMINI_API_KEY environment variable."
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_training_history(self, history_dict: Dict) -> str:
        """
        Analyze training history and provide insights.

        Args:
            history_dict: Dictionary containing training history (loss, accuracy, etc.)

        Returns:
            str: Analysis and recommendations from Gemini
        """
        prompt = f"""
Eres un experto en Deep Learning y diagnóstico médico asistido por IA.
Analiza el siguiente historial de entrenamiento de un modelo de detección de COVID-19
basado en radiografías de tórax usando Transfer Learning con MobileNetV2.

HISTORIAL DE ENTRENAMIENTO:
{json.dumps(history_dict, indent=2)}

INSTRUCCIONES:
1. Analiza las curvas de pérdida (loss) y precisión (accuracy) tanto de entrenamiento como validación.
2. Identifica si hay sobreajuste (overfitting), subajuste (underfitting) o buen balance.
3. Evalúa la convergencia del modelo.
4. Proporciona recomendaciones específicas sobre:
   - Ajustes de hiperparámetros (learning rate, batch size, epochs)
   - Técnicas de regularización (dropout, weight decay)
   - Data augmentation
   - Arquitectura del modelo
   - Cualquier otro aspecto relevante

FORMATO DE RESPUESTA:
- Usa un tono profesional pero accesible
- Estructura tu respuesta con secciones claras
- Incluye métricas específicas cuando sea posible
- Prioriza las recomendaciones más impactantes
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al analizar con Gemini: {str(e)}"

    def analyze_evaluation_metrics(self, metrics_dict: Dict) -> str:
        """
        Analyze evaluation metrics (confusion matrix, ROC, etc.) and provide insights.

        Args:
            metrics_dict: Dictionary containing evaluation metrics

        Returns:
            str: Analysis and recommendations from Gemini
        """
        prompt = f"""
Eres un experto en Machine Learning médico y sistemas CAD (Computer-Aided Diagnosis).
Analiza las siguientes métricas de evaluación de un modelo de detección de COVID-19.

MÉTRICAS DE EVALUACIÓN:
{json.dumps(metrics_dict, indent=2)}

CONTEXTO CLÍNICO IMPORTANTE:
- Este es un sistema de APOYO al diagnóstico, NO reemplaza al médico
- Falsos negativos (no detectar COVID cuando sí existe) son MÁS PELIGROSOS que falsos positivos
- El sistema se usa para PRIORIZACIÓN y CRIBADO inicial
- Debe complementarse SIEMPRE con pruebas PCR y evaluación médica

INSTRUCCIONES DE ANÁLISIS:
1. Evalúa el balance entre Sensibilidad (Recall) y Especificidad
2. Analiza la Matriz de Confusión: identifica el tipo y cantidad de errores
3. Interpreta el valor de AUC (Area Under Curve)
4. Evalúa Precisión y Recall desde una perspectiva clínica
5. Identifica si hay sesgo hacia alguna clase

RECOMENDACIONES REQUERIDAS:
1. ¿El modelo es SEGURO para uso clínico como herramienta de apoyo?
2. ¿Qué ajustes mejorarían la Sensibilidad sin sacrificar demasiado la Especificidad?
3. ¿Se recomienda ajustar el umbral de decisión (actualmente 0.5)?
4. ¿Qué métricas son preocupantes y por qué?
5. Sugerencias concretas de mejora

FORMATO:
- Estructura clara con secciones
- Incluye cifras específicas
- Destaca riesgos clínicos si los hay
- Prioriza la seguridad del paciente
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al analizar con Gemini: {str(e)}"

    def analyze_with_images(self, image_paths: List[str], metrics_dict: Dict) -> str:
        """
        Analyze training/evaluation with visual charts.

        Args:
            image_paths: List of paths to metric visualization images
            metrics_dict: Dictionary with numerical metrics

        Returns:
            str: Analysis and recommendations from Gemini
        """
        prompt = f"""
Eres un experto en Deep Learning médico. Analiza las siguientes gráficas y métricas
de un modelo de detección de COVID-19 en radiografías de tórax.

MÉTRICAS NUMÉRICAS:
{json.dumps(metrics_dict, indent=2)}

INSTRUCCIONES:
1. Examina cuidadosamente todas las gráficas proporcionadas
2. Analiza patrones visuales en las curvas de entrenamiento
3. Interpreta la matriz de confusión visual
4. Evalúa la curva ROC y su área bajo la curva
5. Proporciona un análisis integral combinando datos numéricos y visuales

ENFOQUE:
- Identifica problemas de overfitting/underfitting
- Evalúa el balance entre clases
- Analiza la distribución de predicciones
- Recomienda mejoras específicas y accionables

IMPORTANTE:
- Este es un sistema médico CAD (Computer-Aided Diagnosis)
- La seguridad del paciente es prioritaria
- Sensibilidad (detectar COVID) es crítica
        """

        try:
            # Load images
            images = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    images.append(img)

            # Create content with images
            content = [prompt] + images

            response = self.model.generate_content(content)
            return response.text
        except Exception as e:
            return f"Error al analizar con Gemini: {str(e)}"

    def suggest_hyperparameter_adjustments(self,
                                          current_params: Dict,
                                          metrics: Dict) -> str:
        """
        Suggest hyperparameter adjustments based on current performance.

        Args:
            current_params: Current hyperparameters
            metrics: Current model metrics

        Returns:
            str: Suggested hyperparameter adjustments
        """
        prompt = f"""
Eres un experto en optimización de hiperparámetros para modelos de Deep Learning médico.

HIPERPARÁMETROS ACTUALES:
{json.dumps(current_params, indent=2)}

MÉTRICAS ACTUALES:
{json.dumps(metrics, indent=2)}

TAREA:
Sugiere ajustes específicos a los hiperparámetros para mejorar el rendimiento del modelo.

CONSIDERA:
1. Learning Rate: ¿Es apropiado? ¿Usar learning rate decay/scheduling?
2. Batch Size: ¿Es óptimo para el dataset?
3. Epochs: ¿Más o menos épocas? ¿Early stopping?
4. Regularización: Dropout rate, weight decay
5. Optimizer: ¿Adam es la mejor opción? ¿Probar otros?
6. Data Augmentation: ¿Parámetros apropiados para imágenes médicas?

FORMATO DE RESPUESTA:
Para cada hiperparámetro, proporciona:
- Valor actual
- Valor sugerido
- Justificación breve
- Impacto esperado

Prioriza las recomendaciones por impacto esperado.
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar sugerencias: {str(e)}"

    def comprehensive_analysis(self,
                              history_dict: Dict,
                              metrics_dict: Dict,
                              image_paths: List[str],
                              hyperparams: Dict) -> Dict[str, str]:
        """
        Perform comprehensive analysis of all aspects.

        Args:
            history_dict: Training history
            metrics_dict: Evaluation metrics
            image_paths: Paths to visualization images
            hyperparams: Current hyperparameters

        Returns:
            Dict with different analysis sections
        """
        print("Ejecutando análisis integral con Gemini AI...")

        results = {}

        # 1. Training History Analysis
        print("  - Analizando historial de entrenamiento...")
        results['training_analysis'] = self.analyze_training_history(history_dict)

        # 2. Evaluation Metrics Analysis
        print("  - Analizando métricas de evaluación...")
        results['evaluation_analysis'] = self.analyze_evaluation_metrics(metrics_dict)

        # 3. Visual Analysis
        if image_paths:
            print("  - Analizando visualizaciones...")
            results['visual_analysis'] = self.analyze_with_images(image_paths, metrics_dict)

        # 4. Hyperparameter Suggestions
        print("  - Generando sugerencias de hiperparámetros...")
        results['hyperparameter_suggestions'] = self.suggest_hyperparameter_adjustments(
            hyperparams, metrics_dict
        )

        print("Análisis completo generado exitosamente.")
        return results

    def generate_executive_summary(self, comprehensive_results: Dict[str, str]) -> str:
        """
        Generate an executive summary from all analyses.

        Args:
            comprehensive_results: Dictionary with all analysis results

        Returns:
            str: Executive summary
        """
        prompt = f"""
Basándote en los siguientes análisis detallados de un modelo de detección de COVID-19,
genera un RESUMEN EJECUTIVO conciso y accionable para el equipo médico y técnico.

ANÁLISIS DETALLADO:
{json.dumps(comprehensive_results, indent=2)}

RESUMEN EJECUTIVO DEBE INCLUIR:
1. ESTADO GENERAL DEL MODELO (2-3 líneas): ¿Está listo para uso clínico como herramienta de apoyo?
2. MÉTRICAS CLAVE (bullet points): Las 3-4 métricas más importantes y su interpretación
3. PRINCIPALES HALLAZGOS (bullet points): 3 hallazgos más relevantes
4. RIESGOS IDENTIFICADOS (bullet points): Cualquier riesgo clínico o técnico
5. TOP 3 RECOMENDACIONES: Las 3 acciones más impactantes a tomar

TONO: Profesional, claro, orientado a acción
LONGITUD: Máximo 500 palabras
FORMATO: Markdown con secciones claras
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error al generar resumen ejecutivo: {str(e)}"


if __name__ == '__main__':
    print("Módulo de análisis Gemini AI para COVID-19 Detection System")
    print("\nEste módulo requiere una API key de Google Gemini.")
    print("Configura la variable de entorno GEMINI_API_KEY antes de usar.")
    print("\nEjemplo de uso:")
    print("""
    from gemini_analyzer import GeminiAnalyzer

    # Inicializar
    analyzer = GeminiAnalyzer(api_key='tu-api-key')

    # Analizar métricas
    analysis = analyzer.analyze_evaluation_metrics(metrics_dict)
    print(analysis)
    """)
