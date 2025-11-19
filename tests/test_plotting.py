"""Test plotting functions."""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from caracal.plotting import (
    plot_confusion_matrix,
    plot_training_history,
    plot_variability_summary,
    plot_multiple_comparisons,
    plot_pairwise_comparison_matrix,
    plot_training_stability,
    plot_autocorr_vs_lag,
    plot_averaged_autocorr
)


class TestBasicPlots:
    """Test basic plotting functions."""
    
    def setup_method(self):
        """Create sample data for tests."""
        self.history = pd.DataFrame({
            'epoch': range(10),
            'train_accuracy': np.linspace(0.5, 0.9, 10),
            'val_accuracy': np.linspace(0.45, 0.85, 10),
            'loss': np.linspace(1.0, 0.1, 10)
        })
        
        self.all_histories = [self.history, self.history]
        self.final_accuracies = pd.Series([0.85, 0.88, 0.83, 0.89, 0.86])
        
        # Confusion matrix data
        self.cm_df = pd.DataFrame(
            [[10, 2], [1, 15]], 
            index=['True 0', 'True 1'],
            columns=['Pred 0', 'Pred 1']
        )
    
    def teardown_method(self):
        """Close all figures after each test."""
        plt.close('all')
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plot."""
        fig = plot_confusion_matrix(self.cm_df, title="Test CM", show=False)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
    
    def test_plot_training_history(self):
        """Test training history plot."""
        fig = plot_training_history(
            self.history, 
            title="Test History",
            metrics=['train_accuracy', 'val_accuracy'],
            show=False
        )
        assert fig is not None
        assert isinstance(fig, plt.Figure)
    
    def test_plot_variability_summary(self):
        """Test variability summary plot."""
        fig = plot_variability_summary(
            self.all_histories,
            self.final_accuracies,
            metric='accuracy',
            show=False
        )
        assert fig is not None
    
    def test_plot_autocorr_vs_lag(self):
        """Test autocorrelation plot."""
        data = pd.Series(np.random.randn(100))
        fig = plot_autocorr_vs_lag(data, max_lag=20, show=False)
        assert fig is not None
    
    def test_plot_averaged_autocorr(self):
        """Test averaged autocorrelation plot."""
        lags = list(range(1, 21))
        mean_autocorr = np.random.rand(20)
        std_autocorr = np.random.rand(20) * 0.1
        
        fig = plot_averaged_autocorr(
            lags, mean_autocorr, std_autocorr, 
            title="Test", show=False
        )
        assert fig is not None


class TestStatisticalPlots:
    """Test statistical visualization functions."""
    
    def test_plot_multiple_comparisons(self):
        """Test multiple comparisons plot."""
        from caracal.analysis import StatisticalTestResult
        
        # Mock comparison results
        comparison_results = {
            'overall_test': StatisticalTestResult(
                test_name="Kruskal-Wallis",
                statistic=12.5,
                p_value=0.002
            ),
            'pairwise_comparisons': {},
            'correction_method': 'holm'
        }
        
        fig = plot_multiple_comparisons(comparison_results, show=False)
        assert fig is not None
    
    def test_plot_training_stability(self):
        """Test training stability plot."""
        stability_results = {
            'n_runs': 5,
            'common_length': 10,
            'final_loss_mean': 0.1,
            'final_loss_std': 0.02,
            'final_loss_cv': 0.2,
            'convergence_rate': 0.8,
            'converged_runs': 4,
            'stability_assessment': 'moderate',
            'final_losses_list': [0.08, 0.09, 0.10, 0.11, 0.12]
        }
        
        fig = plot_training_stability(stability_results, show=False)
        assert fig is not None
