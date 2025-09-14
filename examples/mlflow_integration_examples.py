# examples/mlflow_integration_examples.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from caracal import ModelConfig, ExperimentRunner, MLflowLogger
from caracal.data import TabularDataHandler


# Example 1: Basic MLflow logging
def basic_mlflow_example():
    """Basic example of using MLflow logger"""

    # Create logger with experiment name
    logger = MLflowLogger(
        run_name="basic_test_run",
        experiment_name="caracal_basic_tests",
        verbose=True
    )

    # Log parameters
    logger.log_params({
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam'
    })

    # Log some metrics over time (simulating training)
    for epoch in range(10):
        train_loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
        val_loss = 1.1 - (epoch * 0.08) + np.random.normal(0, 0.08)

        logger.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': 1 - train_loss,
            'val_accuracy': 1 - val_loss
        }, step=epoch)

    # Log system information
    logger.log_system_info()

    # Set some tags
    logger.set_tags({
        'model_type': 'neural_network',
        'dataset': 'synthetic',
        'experiment_phase': 'testing'
    })

    # End the run
    logger.end_run()
    print(f"View results at: {logger.get_run_url()}")


# Example 2: Variability study with MLflow
def variability_study_with_mlflow():
    """Example of running a variability study with MLflow logging"""

    # Setup MLflow experiment
    from caracal.loggers import setup_mlflow
    experiment_id = setup_mlflow("caracal_variability_study")

    # Create a parent run for the entire study
    parent_logger = MLflowLogger(
        run_name="variability_study_parent",
        experiment_name="caracal_variability_study"
    )

    # Log study parameters
    parent_logger.log_params({
        'num_runs': 5,
        'model_type': 'simple_nn',
        'study_type': 'variability_analysis'
    })

    parent_logger.set_tags({
        'experiment_type': 'variability_study',
        'purpose': 'reproducibility_analysis'
    })

    # Individual run results
    final_accuracies = []

    # Simulate individual runs
    for run_i in range(5):
        # Create child run
        child_logger = MLflowLogger(
            run_name=f"variability_run_{run_i + 1}",
            experiment_name="caracal_variability_study"
        )

        # Log run-specific parameters
        child_logger.log_params({
            'run_number': run_i + 1,
            'random_seed': run_i * 42,
            'parent_run_id': parent_logger.run_id
        })

        # Simulate training
        final_acc = np.random.normal(0.85, 0.03)  # Simulate some variability
        final_accuracies.append(final_acc)

        child_logger.log_metric('final_accuracy', final_acc)
        child_logger.end_run()

    # Log summary statistics to parent run
    parent_logger.log_metrics({
        'mean_accuracy': np.mean(final_accuracies),
        'std_accuracy': np.std(final_accuracies),
        'min_accuracy': np.min(final_accuracies),
        'max_accuracy': np.max(final_accuracies)
    })

    # Create and log summary plot
    plt.figure(figsize=(10, 6))
    plt.hist(final_accuracies, bins=10, alpha=0.7)
    plt.axvline(np.mean(final_accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(final_accuracies):.3f}')
    plt.xlabel('Final Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Accuracies Across Runs')
    plt.legend()

    parent_logger.log_figure(plt.gcf(), 'variability_summary.png')
    plt.close()

    # Log summary DataFrame
    summary_df = pd.DataFrame({
        'run_id': range(1, 6),
        'final_accuracy': final_accuracies
    })
    parent_logger.log_dataframe(summary_df, 'run_summary.csv')

    parent_logger.end_run()


# Example 3: Model comparison with hypothesis testing
def model_comparison_with_mlflow():
    """Example comparing two models with statistical testing"""

    logger = MLflowLogger(
        run_name="model_comparison",
        experiment_name="caracal_model_comparison"
    )

    # Simulate two model performance distributions
    np.random.seed(42)
    model_a_scores = np.random.normal(0.85, 0.02, 20)
    model_b_scores = np.random.normal(0.87, 0.025, 20)

    # Log comparison parameters
    logger.log_params({
        'model_a_type': 'random_forest',
        'model_b_type': 'gradient_boosting',
        'num_runs_each': 20,
        'test_type': 'mann_whitney'
    })

    # Perform statistical test (using your caracal function)
    from caracal.analysis import mann_whitney_test
    test_result = mann_whitney_test(
        pd.Series(model_a_scores),
        pd.Series(model_b_scores),
        alternative='two-sided'
    )

    # Log test results
    logger.log_metrics({
        'model_a_mean': np.mean(model_a_scores),
        'model_b_mean': np.mean(model_b_scores),
        'mann_whitney_statistic': test_result['statistic'],
        'mann_whitney_pvalue': test_result['p-value']
    })

    # Log the conclusion as a tag
    logger.set_tags({
        'statistical_conclusion': test_result['conclusion'][:100],  # Truncate for MLflow
        'significant_difference': str(test_result['p-value'] < 0.05)
    })

    # Create comparison plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.boxplot([model_a_scores, model_b_scores],
                labels=['Model A', 'Model B'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')

    plt.subplot(1, 2, 2)
    plt.hist(model_a_scores, alpha=0.6, label='Model A', bins=10)
    plt.hist(model_b_scores, alpha=0.6, label='Model B', bins=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Performance Distributions')

    plt.tight_layout()
    logger.log_figure(plt.gcf(), 'model_comparison.png')
    plt.close()

    # Log detailed results
    results_df = pd.DataFrame({
        'model_a_scores': model_a_scores,
        'model_b_scores': model_b_scores
    })
    logger.log_dataframe(results_df, 'comparison_data.csv')

    logger.end_run()


# Example 4: Integration with ExperimentRunner
class MLflowExperimentRunner(ExperimentRunner):
    """Extended ExperimentRunner with MLflow integration"""

    def __init__(self, model_builder, data_handler, model_config,
                 experiment_name="caracal_experiment"):
        # Create MLflow logger instead of base logger
        logger = MLflowLogger(
            run_name="variability_study",
            experiment_name=experiment_name
        )
        super().__init__(model_builder, data_handler, model_config, logger)

        # Log system info and study setup
        self.logger.log_system_info()
        self.logger.set_tags({
            'study_type': 'variability_analysis',
            'data_type': data_handler.data_type,
            'return_format': data_handler.return_format
        })

    def _run_single_fit(self, run_id: int):
        """Override to add more detailed MLflow logging per run"""
        # Log run start
        self.logger.log_metric('runs_completed', run_id - 1, step=run_id)

        result = super()._run_single_fit(run_id)

        if result is not None:
            # Log detailed training history
            for idx, row in result.iterrows():
                step = row.get('epoch', idx)
                metrics = {}
                for col in result.columns:
                    if col not in ['run_num', 'epoch'] and pd.notna(row[col]):
                        # Prefix metrics with run number to avoid conflicts
                        metrics[f"{col}_run_{run_id}"] = row[col]

                if metrics:
                    self.logger.log_metrics(metrics, step=step)

        return result

    def run_study(self):
        """Enhanced run study with comprehensive MLflow logging"""
        results = super().run_study()

        # Log summary statistics
        if self.final_val_accuracies:
            self.logger.log_metrics({
                'study_mean_val_accuracy': np.mean(self.final_val_accuracies),
                'study_std_val_accuracy': np.std(self.final_val_accuracies),
                'study_min_val_accuracy': np.min(self.final_val_accuracies),
                'study_max_val_accuracy': np.max(self.final_val_accuracies)
            })

        # Create and log summary visualizations
        self._log_study_visualizations()

        return results

    def _log_study_visualizations(self):
        """Create and log summary plots"""
        if not self.final_val_accuracies:
            return

        # Accuracy distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.final_val_accuracies, bins=min(10, len(self.final_val_accuracies)),
                 alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(self.final_val_accuracies), color='red',
                    linestyle='--', label=f'Mean: {np.mean(self.final_val_accuracies):.3f}')
        plt.xlabel('Final Validation Accuracy')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Final Accuracies ({len(self.final_val_accuracies)} runs)')
        plt.legend()

        self.logger.log_figure(plt.gcf(), 'accuracy_distribution.png')
        plt.close()

        # Training curves (if we have detailed history)
        if self.all_runs_metrics:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            for i, run_data in enumerate(self.all_runs_metrics):
                if 'train_accuracy' in run_data.columns:
                    plt.plot(run_data['epoch'], run_data['train_accuracy'],
                             alpha=0.6, label='_nolegend_' if i > 0 else 'Training')

            if 'train_accuracy' in self.all_runs_metrics[0].columns:
                plt.xlabel('Epoch')
                plt.ylabel('Training Accuracy')
                plt.title('Training Accuracy Across Runs')
                plt.legend()

            plt.subplot(1, 2, 2)
            for i, run_data in enumerate(self.all_runs_metrics):
                if 'val_accuracy' in run_data.columns:
                    plt.plot(run_data['epoch'], run_data['val_accuracy'],
                             alpha=0.6, label='_nolegend_' if i > 0 else 'Validation')

            if 'val_accuracy' in self.all_runs_metrics[0].columns:
                plt.xlabel('Epoch')
                plt.ylabel('Validation Accuracy')
                plt.title('Validation Accuracy Across Runs')
                plt.legend()

            plt.tight_layout()
            self.logger.log_figure(plt.gcf(), 'training_curves.png')
            plt.close()


if __name__ == "__main__":
    print("Running MLflow integration examples...")

    # Run examples
    basic_mlflow_example()
    print("\n" + "=" * 50 + "\n")

    variability_study_with_mlflow()
    print("\n" + "=" * 50 + "\n")

    model_comparison_with_mlflow()

    print("\nAll examples completed!")
    print("To view results, run: mlflow ui")
    print("Then open http://localhost:5000 in your browser")