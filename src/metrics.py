"""
Advanced metrics module for COVID-19 detection model evaluation.
Generates confusion matrix, ROC curve, AUC, and other performance metrics.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import os
import json

class ModelEvaluator:
    """Class to evaluate model performance and generate visualizations."""

    def __init__(self, model, test_dataset, save_dir='evaluation_results'):
        """
        Initialize the evaluator.

        Args:
            model: Trained Keras model
            test_dataset: Test dataset (tf.data.Dataset)
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.test_dataset = test_dataset
        self.save_dir = save_dir
        self.class_names = ['COVID', 'Normal']

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Get predictions and true labels
        self.y_true, self.y_pred_probs, self.y_pred = self._get_predictions()

    def _get_predictions(self):
        """Get predictions and true labels from the test dataset."""
        y_true = []
        y_pred_probs = []

        print("Generating predictions on test set...")
        for images, labels in self.test_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred_probs.extend(predictions.flatten())
            y_true.extend(labels.numpy().flatten())  # Flatten labels to ensure 1D

        y_true = np.array(y_true).flatten()  # Ensure 1D array
        y_pred_probs = np.array(y_pred_probs).flatten()  # Ensure 1D array
        y_pred = (y_pred_probs > 0.5).astype(int)

        return y_true, y_pred_probs, y_pred

    def plot_confusion_matrix(self, save=True):
        """
        Generate and plot confusion matrix.

        Returns:
            matplotlib.figure.Figure: The confusion matrix figure
        """
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'},
                   ax=ax)

        # Add percentage annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                             ha="center", va="center", color="red", fontsize=10)

        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curve(self, save=True):
        """
        Generate and plot ROC curve.

        Returns:
            tuple: (matplotlib.figure.Figure, float) - Figure and AUC score
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random Classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.save_dir, 'roc_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        return fig, roc_auc

    def plot_precision_recall_curve(self, save=True):
        """
        Generate and plot Precision-Recall curve.

        Returns:
            matplotlib.figure.Figure: The PR curve figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_probs)
        avg_precision = average_precision_score(self.y_true, self.y_pred_probs)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.save_dir, 'precision_recall_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")

        return fig

    def generate_classification_report(self, save=True):
        """
        Generate detailed classification report.

        Returns:
            dict: Classification metrics
        """
        report = classification_report(self.y_true, self.y_pred,
                                      target_names=self.class_names,
                                      output_dict=True)

        if save:
            save_path = os.path.join(self.save_dir, 'classification_report.json')
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Classification report saved to {save_path}")

        # Print report to console
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(self.y_true, self.y_pred,
                                   target_names=self.class_names))

        return report

    def plot_metrics_summary(self, save=True):
        """
        Create a comprehensive metrics summary visualization.

        Returns:
            matplotlib.figure.Figure: Summary figure
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Calculate metrics
        accuracy = accuracy_score(self.y_true, self.y_pred)
        precision = precision_score(self.y_true, self.y_pred)
        recall = recall_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred)

        # Calculate per-class metrics
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Metrics Summary', fontsize=16, fontweight='bold')

        # 1. Overall Metrics Bar Chart
        ax1 = axes[0, 0]
        metrics_names = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'F1-Score', 'Specificity']
        metrics_values = [accuracy, precision, sensitivity, f1, specificity]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        bars = ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_title('Overall Performance Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Confusion Matrix Heatmap
        ax2 = axes[0, 1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_ylabel('True Label', fontsize=11)
        ax2.set_xlabel('Predicted Label', fontsize=11)
        ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

        # 3. Per-Class Performance
        ax3 = axes[1, 0]
        class_metrics = {
            'COVID': {
                'Precision': cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0,
                'Recall': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            },
            'Normal': {
                'Precision': cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
                'Recall': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
            }
        }

        x = np.arange(len(self.class_names))
        width = 0.35

        precisions = [class_metrics[cls]['Precision'] for cls in self.class_names]
        recalls = [class_metrics[cls]['Recall'] for cls in self.class_names]

        ax3.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.7)
        ax3.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c', alpha=0.7)

        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('Per-Class Performance', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.class_names)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1.1])

        # 4. Prediction Distribution
        ax4 = axes[1, 1]
        ax4.hist(self.y_pred_probs[self.y_true == 0], bins=30, alpha=0.6,
                label='COVID (True)', color='red', edgecolor='black')
        ax4.hist(self.y_pred_probs[self.y_true == 1], bins=30, alpha=0.6,
                label='Normal (True)', color='green', edgecolor='black')
        ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        ax4.set_xlabel('Predicted Probability (Normal)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.save_dir, 'metrics_summary.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics summary saved to {save_path}")

        return fig

    def generate_all_metrics(self):
        """
        Generate all evaluation metrics and visualizations.

        Returns:
            dict: Dictionary containing all metrics and file paths
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE MODEL EVALUATION")
        print("="*60 + "\n")

        # Generate all visualizations
        cm_fig = self.plot_confusion_matrix()
        roc_fig, roc_auc = self.plot_roc_curve()
        pr_fig = self.plot_precision_recall_curve()
        summary_fig = self.plot_metrics_summary()
        report = self.generate_classification_report()

        # Calculate additional metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics_dict = {
            'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
            'precision': float(precision_score(self.y_true, self.y_pred)),
            'recall': float(recall_score(self.y_true, self.y_pred)),
            'f1_score': float(f1_score(self.y_true, self.y_pred)),
            'auc': float(roc_auc),
            'classification_report': report,
            'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist(),
            'save_directory': self.save_dir
        }

        # Save metrics summary
        save_path = os.path.join(self.save_dir, 'metrics_summary.json')
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {self.save_dir}")
        print(f"{'='*60}\n")

        return metrics_dict


if __name__ == '__main__':
    # Example usage
    print("This module is meant to be imported and used with a trained model.")
    print("Example usage:")
    print("""
    from tensorflow.keras.models import load_model
    from data_preprocessing import get_datasets
    from metrics import ModelEvaluator

    # Load model and data
    model = load_model('covid_detection_model.h5')
    _, _, test_ds = get_datasets()

    # Create evaluator and generate metrics
    evaluator = ModelEvaluator(model, test_ds)
    metrics = evaluator.generate_all_metrics()
    """)
