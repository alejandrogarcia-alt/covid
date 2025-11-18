import tensorflow as tf
from data_preprocessing import get_datasets
from model import create_model
import matplotlib.pyplot as plt
import os

# --- Constants ---
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
MODEL_SAVE_PATH = 'covid_detection_model.h5'
HISTORY_SAVE_DIR = 'training_history'

def plot_history(history, initial_epochs, title_prefix=''):
    """Plots the training and validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title(f'{title_prefix} Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title(f'{title_prefix} Training and Validation Loss')
    
    plt.suptitle(f'{title_prefix} Model Performance', fontsize=16)
    
    # Save the plot
    if not os.path.exists(HISTORY_SAVE_DIR):
        os.makedirs(HISTORY_SAVE_DIR)
    plt.savefig(os.path.join(HISTORY_SAVE_DIR, f'{title_prefix.lower().replace(" ", "_")}_performance.png'))
    plt.show()


def main():
    """Main training script."""
    # 1. Load Data
    print("--- Loading datasets ---")
    train_ds, val_ds, test_ds = get_datasets()

    # 2. Create Initial Model (Feature Extraction)
    print("\n--- Creating and compiling the initial model (feature extraction) ---")
    model = create_model(fine_tune=False)

    # 3. Phase 1: Train the Head
    print(f"\n--- Starting Phase 1: Training the head for {INITIAL_EPOCHS} epochs ---")
    history = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds
    )

    # 4. Create and Compile Fine-Tuning Model
    print("\n--- Creating and compiling the model for fine-tuning ---")
    # We re-use the same architecture but with fine-tuning enabled
    model_fine_tune = create_model(fine_tune=True)
    
    # We need to load the weights from the previous phase
    # This is a simplified approach; for robustness, one would save and load weights.
    # However, since we are in the same script, we can just continue training
    # on a re-compiled model. Let's re-create the model and train from scratch
    # for simplicity in this script, but the concept is to continue.
    # A better approach is shown below with `model.fit` continuation.

    # Let's actually continue training on the same model, but re-compile with a lower learning rate
    # This is more efficient.
    print("\n--- Re-compiling model for fine-tuning with a lower learning rate ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )


    # 5. Phase 2: Fine-Tuning
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    print(f"\n--- Starting Phase 2: Fine-tuning for {FINE_TUNE_EPOCHS} epochs ---")
    history_fine_tune = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1], # Continue from where we left off
        validation_data=val_ds
    )
    
    # Combine histories
    history.history['accuracy'].extend(history_fine_tune.history['accuracy'])
    history.history['val_accuracy'].extend(history_fine_tune.history['val_accuracy'])
    history.history['loss'].extend(history_fine_tune.history['loss'])
    history.history['val_loss'].extend(history_fine_tune.history['val_loss'])


    # 6. Evaluate the Final Model
    print("\n--- Evaluating the final model on the test set ---")
    loss, accuracy, precision, recall = model.evaluate(test_ds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')

    # 7. Save the Final Model
    print(f"\n--- Saving the final model to {MODEL_SAVE_PATH} ---")
    model.save(MODEL_SAVE_PATH)
    print("Model saved successfully.")

    # 8. Plot and Save History
    print("\n--- Plotting and saving training history ---")
    plot_history(history, INITIAL_EPOCHS, title_prefix='Full Training')


if __name__ == '__main__':
    main()
