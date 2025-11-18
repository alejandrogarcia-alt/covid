"""
Script para convertir modelo H5 a formato SavedModel compatible.
"""
import tensorflow as tf
from tensorflow import keras
import os
import sys

def convert_h5_to_savedmodel(h5_path, output_dir):
    """
    Convierte un modelo .h5 a formato SavedModel.
    """
    print(f"Cargando modelo desde: {h5_path}")

    try:
        # Intentar cargar con compile=False
        model = keras.models.load_model(h5_path, compile=False)
        print("✅ Modelo cargado exitosamente")

        # Recompilar el modelo
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Modelo recompilado")

        # Guardar en formato SavedModel
        model.save(output_dir, save_format='tf')
        print(f"✅ Modelo guardado en formato SavedModel: {output_dir}")

        # También guardar en formato .keras (Keras 3)
        keras_path = output_dir.replace('_savedmodel', '.keras')
        model.save(keras_path, save_format='keras')
        print(f"✅ Modelo guardado en formato .keras: {keras_path}")

        return True

    except Exception as e:
        print(f"❌ Error al convertir modelo: {e}")
        return False

if __name__ == '__main__':
    # Ruta al modelo H5
    h5_model_path = 'training_results_20251117_195038/final_model.h5'

    # Ruta de salida para SavedModel
    savedmodel_path = 'training_results_20251117_195038/model_savedmodel'

    if not os.path.exists(h5_model_path):
        print(f"❌ No se encontró el modelo en: {h5_model_path}")
        sys.exit(1)

    print("="*60)
    print("CONVERSIÓN DE MODELO H5 A SAVEDMODEL")
    print("="*60)

    success = convert_h5_to_savedmodel(h5_model_path, savedmodel_path)

    if success:
        print("\n" + "="*60)
        print("✅ CONVERSIÓN COMPLETADA EXITOSAMENTE")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ LA CONVERSIÓN FALLÓ")
        print("="*60)
        sys.exit(1)
