import tensorflow as tf
from tensorflow.keras import layers, models

# --- Constants ---
IMG_SHAPE = (224, 224, 3)

def get_data_augmentation():
    """
    Returns a Sequential model with data augmentation layers.
    This helps prevent overfitting by creating modified versions of the training data.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

def create_model(input_shape=IMG_SHAPE, fine_tune=False):
    """
    Creates the CNN model using transfer learning with MobileNetV2.

    Args:
        input_shape (tuple): The shape of the input images.
        fine_tune (bool): If True, unfreezes the top layers of the base model
                          for fine-tuning.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    # --- 1. Base Model (MobileNetV2) ---
    # Load the pre-trained MobileNetV2 model without its top classification layer.
    # We will use the weights learned from the large ImageNet dataset.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model by default
    base_model.trainable = False

    # --- Fine-Tuning Setup ---
    # If fine_tune is True, unfreeze some of the later layers.
    # The layers at the end of the base model are more specialized. By retraining
    # them on our dataset, the model can learn features specific to X-rays.
    if fine_tune:
        base_model.trainable = True
        # We only want to fine-tune the top layers, not the entire network.
        # Let's freeze all layers before the 100th layer.
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # --- 2. Create the New "Head" of the Model ---
    # This part of the model will take the output of the base_model and
    # produce our final binary classification (COVID vs. Normal).
    inputs = tf.keras.Input(shape=input_shape)

    # Apply data augmentation only during training
    x = get_data_augmentation()(inputs)

    # Preprocess the inputs for the MobileNetV2 model
    # Manual preprocessing to avoid TrueDivide layer compatibility issues
    # This scales pixel values from [0, 255] to [-1, 1] (same as preprocess_input)
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x)

    # Pass the preprocessed inputs to the base model
    x = base_model(x, training=False) # Set training=False for frozen layers

    # --- 3. Classification Head ---
    # Convert the features from the base model into a single prediction.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x) # Regularization to prevent overfitting
    # Final prediction layer with a sigmoid activation for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # --- 4. Compile the Model ---
    # The learning rate for fine-tuning should be much lower than for initial training.
    learning_rate = 0.0001 if fine_tune else 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

if __name__ == '__main__':
    # This block is for testing the script directly
    print("--- Creating Initial Model (Feature Extraction) ---")
    model = create_model()
    model.summary()
    print(f"\nNumber of trainable variables: {len(model.trainable_variables)}")

    print("\n" + "="*50 + "\n")

    print("--- Creating Model for Fine-Tuning ---")
    fine_tune_model = create_model(fine_tune=True)
    fine_tune_model.summary()
    print(f"\nNumber of trainable variables for fine-tuning: {len(fine_tune_model.trainable_variables)}")
