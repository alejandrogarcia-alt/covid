import tensorflow as tf
import os

# --- Constants ---
# Using the absolute path provided by the user
DATA_DIR = '/Users/amgarcia71/Downloads/Dataset'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def get_datasets(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, seed=SEED):
    """
    Loads the dataset from the specified directory and splits it into training,
    validation, and test sets.

    Args:
        data_dir (str): The path to the dataset directory, which should contain
                        subdirectories for each class (e.g., 'COVID', 'Normal').
        image_size (tuple): The target size for the images (height, width).
        batch_size (int): The batch size for the datasets.
        seed (int): Random seed for shuffling and transformations.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets
               as `tf.data.Dataset` objects.
               (train_ds, val_ds, test_ds)
    """
    # Validation split: 20% of the data
    # Of the remaining 80%, we'll use 25% for testing (which is 20% of the total)
    # and 75% for training (which is 60% of the total).
    # Total: 60% train, 20% validation, 20% test.

    # First, create the training and validation sets
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary' # For 'Normal' vs 'COVID'
    )

    # Further split the training set to create a test set
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 2)
    val_ds = val_ds.skip(val_batches // 2)

    print(f"Found {train_ds.cardinality() * batch_size} images for training.")
    print(f"Found {val_ds.cardinality() * batch_size} images for validation.")
    print(f"Found {test_ds.cardinality() * batch_size} images for testing.")

    # --- Configure for Performance ---
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def get_class_names():
    """
    Gets the class names from the directory structure.
    
    Returns:
        list: A list of class names (e.g., ['COVID', 'Normal']).
    """
    return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])


if __name__ == '__main__':
    # This block is for testing the script directly
    print("--- Loading Datasets ---")
    train_dataset, val_dataset, test_dataset = get_datasets()

    class_names = get_class_names()
    print(f"Class names: {class_names}")

    # Print the shape and type of a batch
    for images, labels in train_dataset.take(1):
        print("\n--- Batch Info ---")
        print(f"Images batch shape: {images.shape}")
        print(f"Labels batch shape: {labels.shape}")
        print(f"Image data type: {images.dtype}")
        print(f"Label data type: {labels.dtype}")
