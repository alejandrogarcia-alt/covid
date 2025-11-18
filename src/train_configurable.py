"""
Configurable training script for COVID-19 detection model.
Allows customization of hyperparameters and generates comprehensive metrics.
"""

import tensorflow as tf
from data_preprocessing import get_datasets
from model import create_model
from metrics import ModelEvaluator
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime


class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self,
                 initial_epochs=10,
                 fine_tune_epochs=10,
                 initial_learning_rate=0.001,
                 fine_tune_learning_rate=0.0001,
                 batch_size=32,
                 image_size=(224, 224),
                 dropout_rate=0.2,
                 data_augmentation=True,
                 model_architecture='MobileNetV2',
                 fine_tune_at=100,
                 optimizer='adam',
                 early_stopping_patience=5,
                 reduce_lr_patience=3):
        """
        Initialize training configuration.

        Args:
            initial_epochs: Number of epochs for initial training (feature extraction)
            fine_tune_epochs: Number of epochs for fine-tuning
            initial_learning_rate: Learning rate for initial training
            fine_tune_learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
            image_size: Input image size (height, width)
            dropout_rate: Dropout rate for regularization
            data_augmentation: Whether to use data augmentation
            model_architecture: Base model architecture ('MobileNetV2' or 'ResNet50')
            fine_tune_at: Layer index to start fine-tuning from
            optimizer: Optimizer to use ('adam', 'sgd', 'rmsprop')
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
        """
        self.initial_epochs = initial_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.initial_learning_rate = initial_learning_rate
        self.fine_tune_learning_rate = fine_tune_learning_rate
        self.batch_size = batch_size
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.data_augmentation = data_augmentation
        self.model_architecture = model_architecture
        self.fine_tune_at = fine_tune_at
        self.optimizer = optimizer
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'initial_epochs': self.initial_epochs,
            'fine_tune_epochs': self.fine_tune_epochs,
            'initial_learning_rate': self.initial_learning_rate,
            'fine_tune_learning_rate': self.fine_tune_learning_rate,
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'dropout_rate': self.dropout_rate,
            'data_augmentation': self.data_augmentation,
            'model_architecture': self.model_architecture,
            'fine_tune_at': self.fine_tune_at,
            'optimizer': self.optimizer,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


def plot_training_history(history, initial_epochs, save_dir='training_results'):
    """
    Plot and save training history.

    Args:
        history: Training history object
        initial_epochs: Number of initial epochs (for marking on plot)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(16, 6))

    # Plot 1: Accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    if initial_epochs > 0:
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--',
                   label='Start Fine-Tuning', linewidth=2)
    plt.legend(loc='lower right', fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.grid(alpha=0.3)

    # Plot 2: Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    if initial_epochs > 0:
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--',
                   label='Start Fine-Tuning', linewidth=2)
    plt.legend(loc='upper right', fontsize=10)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(alpha=0.3)

    # Plot 3: Precision and Recall
    plt.subplot(1, 3, 3)
    if 'precision' in history.history:
        plt.plot(epochs_range, history.history['precision'],
                label='Training Precision', linewidth=2)
        plt.plot(epochs_range, history.history['val_precision'],
                label='Validation Precision', linewidth=2)
    if 'recall' in history.history:
        plt.plot(epochs_range, history.history['recall'],
                label='Training Recall', linewidth=2, linestyle='--')
        plt.plot(epochs_range, history.history['val_recall'],
                label='Validation Recall', linewidth=2, linestyle='--')
    if initial_epochs > 0:
        plt.axvline(x=initial_epochs-1, color='r', linestyle='--',
                   label='Start Fine-Tuning', linewidth=2)
    plt.legend(loc='lower right', fontsize=9)
    plt.title('Precision and Recall', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Score', fontsize=11)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train_model(config: TrainingConfig, save_dir='training_results'):
    """
    Train the COVID-19 detection model with given configuration.

    Args:
        config: TrainingConfig object with hyperparameters
        save_dir: Directory to save results

    Returns:
        tuple: (trained_model, history, metrics_dict)
    """
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{save_dir}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    print(f"Configuration saved to {config_path}")

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    for key, value in config.to_dict().items():
        print(f"{key:30s}: {value}")
    print("="*70 + "\n")

    # Load datasets with configured batch size
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(
        batch_size=config.batch_size,
        image_size=config.image_size
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        input_shape=(*config.image_size, 3),
        fine_tune=False
    )

    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.reduce_lr_patience,
            verbose=1,
            min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Phase 1: Feature Extraction
    print(f"\n{'='*70}")
    print(f"PHASE 1: FEATURE EXTRACTION ({config.initial_epochs} epochs)")
    print("="*70 + "\n")

    history_phase1 = model.fit(
        train_ds,
        epochs=config.initial_epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Phase 2: Fine-Tuning (if configured)
    if config.fine_tune_epochs > 0:
        print(f"\n{'='*70}")
        print(f"PHASE 2: FINE-TUNING ({config.fine_tune_epochs} epochs)")
        print("="*70 + "\n")

        # Unfreeze base model layers for fine-tuning
        # Instead of creating a new model, we'll modify the existing one
        print("Unfreezing base model layers for fine-tuning...")

        # Get the base model from the current model
        base_model = None
        for layer in model.layers:
            if 'mobilenet' in layer.name.lower():
                base_model = layer
                break

        if base_model is not None:
            base_model.trainable = True
            # Freeze early layers, only train the last ones
            fine_tune_at = 100
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            print(f"Unfroze layers from index {fine_tune_at} onwards in base model")

        # Recompile with lower learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.fine_tune_learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )

        total_epochs = config.initial_epochs + config.fine_tune_epochs

        history_phase2 = model.fit(
            train_ds,
            epochs=total_epochs,
            initial_epoch=config.initial_epochs,
            validation_data=val_ds,
            callbacks=callbacks
        )

        # Combine histories
        combined_history = history_phase1.history.copy()
        for key in history_phase2.history.keys():
            if key in combined_history:
                combined_history[key].extend(history_phase2.history[key])
            else:
                combined_history[key] = history_phase2.history[key]

        # Create a mock History object
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict

        history = CombinedHistory(combined_history)
    else:
        history = history_phase1

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, config.initial_epochs, save_dir)

    # Save training history as JSON
    history_json_path = os.path.join(save_dir, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    print(f"Training history saved to {history_json_path}")

    # Evaluate on test set
    print(f"\n{'='*70}")
    print("FINAL EVALUATION ON TEST SET")
    print("="*70 + "\n")

    # Generate comprehensive metrics
    print("Generating comprehensive evaluation metrics...")
    evaluator = ModelEvaluator(model, test_ds, save_dir=save_dir)
    metrics_dict = evaluator.generate_all_metrics()

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"All results saved to: {save_dir}")
    print("="*70 + "\n")

    return model, history, metrics_dict, save_dir


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train COVID-19 Detection Model')

    parser.add_argument('--initial_epochs', type=int, default=10,
                       help='Number of initial training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--fine_tune_lr', type=float, default=0.0001,
                       help='Fine-tuning learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        initial_learning_rate=args.initial_lr,
        fine_tune_learning_rate=args.fine_tune_lr,
        batch_size=args.batch_size,
        dropout_rate=args.dropout
    )

    # Train model
    model, history, metrics, save_dir = train_model(config)

    print("\nTraining completed successfully!")
    print(f"Results saved to: {save_dir}")
