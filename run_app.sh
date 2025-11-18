#!/bin/bash

# Script de inicio rápido para el Sistema de Detección de COVID-19
# Diplomatura en IA - UTN

echo "=========================================="
echo "🩺 COVID-19 Detection System"
echo "Diplomatura en Inteligencia Artificial - UTN"
echo "=========================================="
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "app/app_complete.py" ]; then
    echo "❌ Error: No se encuentra el archivo app_complete.py"
    echo "Por favor, ejecuta este script desde el directorio raíz del proyecto"
    exit 1
fi

# Verificar que existe el entorno virtual (opcional)
if [ -d "venv" ]; then
    echo "✅ Entorno virtual encontrado"
    echo "Activando entorno virtual..."
    source venv/bin/activate
else
    echo "⚠️  No se encontró entorno virtual"
    echo "Recomendación: Crea un entorno virtual con 'python -m venv venv'"
fi

# Verificar que streamlit está instalado
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit no está instalado"
    echo "Instalando dependencias..."
    pip install -r requirements.txt
fi

# Verificar que existe el dataset
if [ ! -d "/Users/amgarcia71/Downloads/Dataset/COVID" ] || [ ! -d "/Users/amgarcia71/Downloads/Dataset/Normal" ]; then
    echo ""
    echo "⚠️  ADVERTENCIA: Dataset no encontrado en /Users/amgarcia71/Downloads/Dataset/"
    echo "Asegúrate de que existan las carpetas:"
    echo "  - /Users/amgarcia71/Downloads/Dataset/COVID/"
    echo "  - /Users/amgarcia71/Downloads/Dataset/Normal/"
    echo ""
fi

echo ""
echo "🚀 Iniciando aplicación..."
echo ""
echo "La aplicación se abrirá en tu navegador en:"
echo "  👉 http://localhost:8501"
echo ""
echo "Para detener la aplicación, presiona Ctrl+C"
echo ""
echo "=========================================="
echo ""

# Ejecutar la aplicación
streamlit run app/app_complete.py
