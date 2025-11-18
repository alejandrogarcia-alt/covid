import tensorflow as tf
import numpy as np
import cv2 # OpenCV for image manipulation
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import os
import sys
from contextlib import contextmanager

# Configure TensorFlow logging to suppress messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.get_logger().setLevel('ERROR')

# --- Constants ---
MODEL_PATH = 'covid_detection_model.h5'
IMAGE_SIZE = (224, 224)

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            os.close(fd)

def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Loads and preprocesses a single image for model prediction.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a Grad-CAM heatmap.
    Handles both top-level and nested layer names (e.g., "mobilenetv2/Conv_1").
    """
    # Convert input to tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Get the conv layer, handling nested models
    if "/" in last_conv_layer_name:
        # Nested layer: "base_model_name/conv_layer_name"
        base_name, conv_name = last_conv_layer_name.split("/", 1)
        base_model = model.get_layer(base_name)
        conv_layer = base_model.get_layer(conv_name)

        # Create a submodel that outputs BOTH conv and final output
        conv_output_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[conv_layer.output, base_model.output]
        )

        # Manual forward pass within GradientTape
        # This ensures conv_output and preds are in the same computational graph
        with tf.GradientTape() as tape:
            x = img_tensor

            # Process through each layer
            for layer in model.layers:
                if layer.name == base_name:
                    # Get both conv output and base_model output in ONE call
                    last_conv_layer_output, base_out = conv_output_model(x)
                    # Watch the conv output for gradient computation
                    tape.watch(last_conv_layer_output)
                    # Continue with base_model output
                    x = base_out
                elif not isinstance(layer, tf.keras.layers.InputLayer):
                    # Apply other layers
                    x = layer(x)

            # x is now the final predictions
            preds = x
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Compute gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)

    else:
        # Top-level layer - standard approach
        conv_layer = model.get_layer(last_conv_layer_name)

        # Create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [conv_layer.output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_tensor)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_saliency_map(image_path, img_array, model, predicted_class):
    """
    Genera un mapa de saliencia (Saliency Map) como alternativa a Grad-CAM.
    Más simple y robusto - calcula gradientes de la salida respecto a la entrada.
    """
    # Convertir a tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Calcular gradientes con respecto a la imagen de entrada
    # Suppress output to avoid BrokenPipeError
    with suppress_stdout_stderr():
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = model(img_tensor, training=False)

            # Obtener el score de la clase predicha
            if predicted_class == "COVID":
                target_class_score = 1 - predictions[0][0]  # COVID es cuando la predicción es baja
            else:
                target_class_score = predictions[0][0]  # Normal es cuando la predicción es alta

        # Calcular gradientes
        gradients = tape.gradient(target_class_score, img_tensor)

    # Procesar gradientes para mejor visualización
    gradients = tf.abs(gradients)

    # Opción 1: Usar la suma de los gradientes en todos los canales (más intenso)
    gradients = tf.reduce_sum(gradients, axis=-1)[0]

    # Convertir a numpy
    gradients = gradients.numpy()

    # Normalizar con percentiles para aumentar contraste
    # Esto hace que los valores extremos sean más visibles
    p2, p98 = np.percentile(gradients, (2, 98))
    gradients = np.clip(gradients, p2, p98)
    gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

    # Aumentar contraste aplicando una función de potencia
    gradients = np.power(gradients, 0.7)  # Reduce el exponente para más contraste

    # Cargar imagen original
    original_img = cv2.imread(image_path)

    # Redimensionar el mapa de saliencia al tamaño de la imagen original
    saliency_map = cv2.resize(gradients, (original_img.shape[1], original_img.shape[0]))

    # Aplicar suavizado para mejor visualización
    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)

    # Convertir a mapa de calor con escala de 0-255
    heatmap = np.uint8(255 * saliency_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superponer con balance 30/70 entre mapa de calor y radiografía
    alpha = 0.3  # 30% de peso al heatmap, 70% a la imagen original
    superimposed = cv2.addWeighted(original_img, 1-alpha, heatmap, alpha, 0)

    return superimposed

def overlay_heatmap(original_img_path, heatmap, alpha=0.4):
    """
    Overlays the heatmap on the original image.
    """
    # Load the original image
    img = cv2.imread(original_img_path)

    # Resize the heatmap to be the same size as the image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

def predict_and_visualize(image_path, model_path=MODEL_PATH):
    """
    Makes a prediction on an image and generates a Grad-CAM visualization.
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}.\n"
            "Please train a model first using the 'Entrenamiento' section."
        )

    # Try to load the model with custom object scope to handle TrueDivide compatibility issues
    try:
        # Import the actual TrueDivide operation class from TensorFlow
        from tensorflow.python.ops.math_ops import truediv

        # Use custom_object_scope to register the 'TrueDivide' operation
        # This handles models saved with older TensorFlow versions that used preprocess_input
        # Suppress stdout/stderr to avoid BrokenPipeError in Streamlit
        with suppress_stdout_stderr():
            with custom_object_scope({'TrueDivide': truediv}):
                model = load_model(model_path, compile=False)
    except (ValueError, OSError) as e:
        # If loading still fails, provide a helpful error message
        if "Unknown layer" in str(e) or "TrueDivide" in str(e):
            raise ValueError(
                "The saved model could not be loaded.\n\n"
                "SOLUTION: Please train a new model using the 'Entrenamiento' section.\n"
                "Newer models use a compatible preprocessing layer (Rescaling) and won't have this issue."
            ) from e
        else:
            raise
    
    # Find the name of the last convolutional layer
    # For transfer learning models, Conv2D layers are often in nested base models
    last_conv_layer_name = None

    # First, try to find Conv2D in top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    # If not found, search inside nested models (like MobileNetV2)
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, 'layers'):  # This is a nested model
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        # Use the full path: base_model_name/conv_layer_name
                        last_conv_layer_name = f"{layer.name}/{nested_layer.name}"
                        break
                if last_conv_layer_name:
                    break

    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv2D layer in the model.")

    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make prediction with suppressed output
    with suppress_stdout_stderr():
        prediction = model.predict(img_array, verbose=0)[0][0]

    class_names = ['COVID', 'Normal'] # Assuming 0 is COVID, 1 is Normal based on binary output

    # The label is often inferred from the directory structure, let's assume
    # 'COVID' is class 0 and 'Normal' is class 1.
    # A sigmoid > 0.5 means class 1.
    if prediction > 0.5:
        predicted_class = "Normal"
        confidence = prediction
    else:
        predicted_class = "COVID"
        confidence = 1 - prediction

    # Generar visualización alternativa con Saliency Map
    try:
        saliency_img = generate_saliency_map(image_path, img_array, model, predicted_class)
        return predicted_class, confidence, saliency_img
    except Exception as e:
        print(f"Warning: Could not generate saliency map: {e}")
        # Si falla, retornar la imagen original
        original_img = cv2.imread(image_path)
        return predicted_class, confidence, original_img

if __name__ == '__main__':
    # Example usage:
    # You need to provide a path to an image from your dataset.
    # Let's find one from the user's dataset path.
    import os
    
    dataset_dir = '/Users/amgarcia71/Downloads/Dataset'
    # Pick a sample image to test
    sample_image_path = os.path.join(dataset_dir, 'COVID', 'COVID-1.png')

    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}. Please check the path.")
    else:
        # Run the prediction and visualization
        predicted_class, confidence, superimposed_image = predict_and_visualize(sample_image_path)
        
        # Save the output image
        output_path = 'prediction_with_heatmap.png'
        cv2.imwrite(output_path, superimposed_image)
        print(f"Saved visualization to {output_path}")
        
        # Display the image (optional, works in environments like Jupyter)
        # from matplotlib import pyplot as plt
        # plt.imshow(cv2.cvtColor(superimposed_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
        # plt.axis('off')
        # plt.show()
