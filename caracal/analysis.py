from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import warnings

# Optional scipy imports
try:
    from scipy.stats import (
        beta, f_oneway, kruskal, mannwhitneyu, shapiro, wilcoxon,
        ttest_ind, ttest_rel, levene, normaltest, jarque_bera
    )
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional sklearn import (only for confusion matrix)
try:
    from sklearn.metrics import confusion_matrix

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class StatisticalTestResult:
    """Comprehensive statistical test result with validation, effect sizes, and interpretation."""

    test_name: str
    statistic: float
    p_value: float

    # Effect size information
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    effect_size_interpretation: Optional[str] = None

    # Sample and power information
    sample_sizes: Optional[Dict[str, int]] = None

    # Assumption checking
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    assumption_details: Dict[str, Any] = field(default_factory=dict)

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Multiple comparison correction
    corrected_p_value: Optional[float] = None
    correction_method: Optional[str] = None

    # Interpretation
    conclusion: str = ""
    detailed_interpretation: str = ""

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        p_val = self.corrected_p_value if self.corrected_p_value is not None else self.p_value
        return p_val < alpha

    def get_summary(self) -> str:
        """Get a concise summary of the test result."""
        sig_marker = "***" if self.is_significant(0.001) else "**" if self.is_significant(
            0.01) else "*" if self.is_significant(0.05) else "ns"

        summary = f"{self.test_name}: {self.statistic:.3f}, p={self.p_value:.4f} {sig_marker}"

        if self.effect_size is not None:
            summary += f", {self.effect_size_name}={self.effect_size:.3f}"

        if self.warnings:
            summary += f" (⚠ {len(self.warnings)} warnings)"

        return summary


def _check_scipy():
    """Check scipy availability."""
    if not HAS_SCIPY:
        raise ImportError("scipy required for statistical tests. Install with: pip install scipy")


def _check_sklearn():
    """Check sklearn availability."""
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for confusion matrix. Install with: pip install scikit-learn")


# VALIDATION FUNCTIONS

def validate_sample_sizes(data: Union[pd.Series, List[pd.Series]],
                          min_size: int, test_name: str) -> Tuple[bool, List[str]]:
    """Validate sample sizes are adequate for statistical testing."""
    warnings_list = []

    if isinstance(data, pd.Series):
        sizes = [len(data)]
    else:
        sizes = [len(series) for series in data]

    inadequate_sizes = [i for i, size in enumerate(sizes) if size < min_size]

    if inadequate_sizes:
        warnings_list.append(
            f"{test_name} requires at least {min_size} samples per group. "
            f"Groups {inadequate_sizes} have insufficient data."
        )
        return False, warnings_list

    return True, warnings_list


def check_normality(data: pd.Series, alpha: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    """Test for normality using multiple methods."""
    _check_scipy()

    results = {}

    if len(data) < 3:
        return False, {"error": "Insufficient data for normality testing"}

    # Shapiro-Wilk test (best for small samples)
    if len(data) <= 5000:  # Shapiro-Wilk has sample size limits
        try:
            shapiro_stat, shapiro_p = shapiro(data)
            results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
        except Exception:
            pass

    # D'Agostino and Pearson test (better for larger samples)
    if len(data) >= 20:
        try:
            dagostino_stat, dagostino_p = normaltest(data)
            results['dagostino'] = {'statistic': dagostino_stat, 'p_value': dagostino_p}
        except Exception:
            pass

    if not results:
        return False, {"error": "Could not perform normality tests"}

    # Consider normal if any test fails to reject normality
    is_normal = any(test['p_value'] > alpha for test in results.values())

    return is_normal, results


def check_equal_variances(data1: pd.Series, data2: pd.Series,
                          alpha: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    """Test for equal variances using Levene's test."""
    _check_scipy()

    try:
        stat, p_val = levene(data1, data2)
        equal_vars = p_val > alpha

        return equal_vars, {
            'levene_statistic': stat,
            'levene_p': p_val,
            'conclusion': 'equal' if equal_vars else 'unequal'
        }
    except Exception as e:
        return False, {'error': str(e)}


def check_independence(data: pd.Series, max_lag: int = 5,
                       alpha: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    """Check for autocorrelation that suggests non-independence."""

    autocorr_results = {}
    significant_lags = []

    for lag in range(1, min(max_lag + 1, len(data) // 4)):
        try:
            autocorr = data.autocorr(lag)
            if not np.isnan(autocorr):
                autocorr_results[f'lag_{lag}'] = autocorr

                # Rough significance test (assuming normal distribution)
                se = 1.0 / np.sqrt(len(data))
                if abs(autocorr) > 1.96 * se:  # 95% confidence
                    significant_lags.append(lag)
        except Exception:
            continue

    is_independent = len(significant_lags) == 0

    details = {
        'autocorrelations': autocorr_results,
        'significant_lags': significant_lags,
        'max_autocorr': max(autocorr_results.values()) if autocorr_results else 0
    }

    return is_independent, details


# EFFECT SIZE CALCULATIONS

def cohens_d(group1: pd.Series, group2: pd.Series,
             pooled: bool = True) -> Tuple[float, str]:
    """Calculate Cohen's d effect size."""

    mean1, mean2 = group1.mean(), group2.mean()

    if pooled:
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(), group2.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    else:
        # Control group standard deviation
        std2 = group2.std()
        d = (mean1 - mean2) / std2 if std2 > 0 else 0

    interpretation = _interpret_cohens_d(abs(d))

    return d, interpretation


def _interpret_cohens_d(abs_d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def rank_biserial_correlation(group1: pd.Series, group2: pd.Series) -> Tuple[float, str]:
    """Calculate rank-biserial correlation (effect size for Mann-Whitney test)."""
    _check_scipy()

    n1, n2 = len(group1), len(group2)

    try:
        U, _ = mannwhitneyu(group1, group2, alternative='two-sided')
        # Convert to rank-biserial correlation
        r = 1 - (2 * U) / (n1 * n2)
    except Exception:
        r = 0.0

    interpretation = _interpret_rank_biserial(abs(r))

    return r, interpretation


def _interpret_rank_biserial(abs_r: float) -> str:
    """Interpret rank-biserial correlation magnitude."""
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "small"
    elif abs_r < 0.5:
        return "medium"
    else:
        return "large"


def eta_squared(groups: List[pd.Series]) -> Tuple[float, str]:
    """Calculate eta-squared effect size for ANOVA from group data."""

    # Calculate sums of squares
    all_data = pd.concat(groups, ignore_index=True)
    grand_mean = all_data.mean()

    # Total sum of squares
    ss_total = ((all_data - grand_mean) ** 2).sum()

    # Between-group sum of squares
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)

    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    interpretation = _interpret_eta_squared(eta_sq)

    return eta_sq, interpretation


def _interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared effect size magnitude."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


# MULTIPLE COMPARISON CORRECTIONS

def apply_multiple_comparison_correction(p_values: List[float],
                                         method: str = 'holm') -> Tuple[List[float], str]:
    """Apply multiple comparison correction."""

    p_array = np.array(p_values)
    n = len(p_array)

    if method == 'bonferroni':
        corrected = p_array * n
        corrected = np.minimum(corrected, 1.0)
        description = f"Bonferroni correction (α adjusted by factor of {n})"

    elif method == 'holm':
        # Sort p-values with original indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Apply Holm correction
        corrected_sorted = np.minimum.accumulate(sorted_p * (n - np.arange(n)))
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        # Restore original order
        corrected = np.empty_like(corrected_sorted)
        corrected[sorted_indices] = corrected_sorted

        description = "Holm step-down correction (less conservative than Bonferroni)"

    elif method == 'fdr_bh':  # Benjamini-Hochberg FDR
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # BH procedure
        corrected_sorted = sorted_p * n / (np.arange(n) + 1)
        corrected_sorted = np.minimum.accumulate(corrected_sorted[::-1])[::-1]
        corrected_sorted = np.minimum(corrected_sorted, 1.0)

        corrected = np.empty_like(corrected_sorted)
        corrected[sorted_indices] = corrected_sorted

        description = "Benjamini-Hochberg FDR correction (controls false discovery rate)"

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return corrected.tolist(), description


# STATISTICAL TESTS

def mann_whitney_test(model1_metrics: pd.Series, model2_metrics: pd.Series,
                      alternative: str = 'two-sided',
                      alpha: float = 0.05) -> StatisticalTestResult:
    """Mann-Whitney U test with comprehensive validation, effect sizes, and interpretation."""
    _check_scipy()

    result = StatisticalTestResult(test_name="Mann-Whitney U Test")

    # Validate inputs
    if not isinstance(model1_metrics, pd.Series) or not isinstance(model2_metrics, pd.Series):
        raise TypeError("Inputs must be pandas Series.")

    # Clean data
    clean1 = model1_metrics.dropna()
    clean2 = model2_metrics.dropna()

    result.sample_sizes = {'group1': len(clean1), 'group2': len(clean2)}

    # Sample size validation
    adequate_size, size_warnings = validate_sample_sizes([clean1, clean2], 5, "Mann-Whitney U test")
    result.warnings.extend(size_warnings)
    result.assumptions_met['adequate_sample_size'] = adequate_size

    # Independence check
    is_indep1, indep_details1 = check_independence(clean1)
    is_indep2, indep_details2 = check_independence(clean2)

    if not is_indep1 or not is_indep2:
        result.warnings.append("Data shows evidence of autocorrelation, violating independence assumption")

    result.assumptions_met['independence'] = is_indep1 and is_indep2
    result.assumption_details['independence'] = {'group1': indep_details1, 'group2': indep_details2}

    # Perform test
    try:
        statistic, p_value = mannwhitneyu(clean1, clean2, alternative=alternative)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

        # Effect size
        r, r_interpretation = rank_biserial_correlation(clean1, clean2)
        result.effect_size = r
        result.effect_size_name = "rank-biserial correlation"
        result.effect_size_interpretation = r_interpretation

    except Exception as e:
        result.warnings.append(f"Test failed: {str(e)}")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    # Generate interpretation
    result.conclusion = _generate_mann_whitney_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    # Recommendations
    if not adequate_size:
        result.recommendations.append("Collect more data for more reliable results")
    if result.effect_size is not None and abs(result.effect_size) < 0.1:
        result.recommendations.append("Consider if this small effect size is practically meaningful")

    return result


def wilcoxon_signed_rank_test(model_metrics: pd.Series, null_value: float = 0.5,
                              alternative: str = 'two-sided',
                              alpha: float = 0.05) -> StatisticalTestResult:
    """Wilcoxon signed-rank test with comprehensive validation and effect sizes."""
    _check_scipy()

    result = StatisticalTestResult(test_name="Wilcoxon Signed-Rank Test")

    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    # Clean data and center on null value
    clean_data = model_metrics.dropna()
    centered_data = clean_data - null_value

    # Remove zeros (ties at the null hypothesis value)
    non_zero_data = centered_data[centered_data != 0]

    result.sample_sizes = {'total': len(clean_data), 'non_zero': len(non_zero_data)}

    # Sample size validation
    adequate_size, size_warnings = validate_sample_sizes([non_zero_data], 6, "Wilcoxon signed-rank test")
    result.warnings.extend(size_warnings)
    result.assumptions_met['adequate_sample_size'] = adequate_size

    # Symmetry assumption check (approximate)
    if len(centered_data) > 0:
        skewness = centered_data.skew()
        if abs(skewness) > 1:
            result.warnings.append("Data appears highly skewed, violating symmetry assumption")
            result.assumptions_met['symmetry'] = False
        else:
            result.assumptions_met['symmetry'] = True

        result.assumption_details['skewness'] = skewness

    # Perform test
    try:
        if len(non_zero_data) < 6:
            raise ValueError("Insufficient non-zero differences for Wilcoxon test")

        statistic, p_value = wilcoxon(non_zero_data, alternative=alternative)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

        # Effect size (r = Z / sqrt(N))
        n = len(non_zero_data)
        if n > 0:
            z_score = (statistic - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            r = abs(z_score) / np.sqrt(n)

            result.effect_size = r
            result.effect_size_name = "r (effect size)"
            result.effect_size_interpretation = _interpret_wilcoxon_r(r)

    except Exception as e:
        result.warnings.append(f"Test failed: {str(e)}")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    # Generate interpretation
    result.conclusion = _generate_wilcoxon_conclusion(result, null_value, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def anova_test(model_metrics: Dict[str, pd.Series],
               alpha: float = 0.05) -> StatisticalTestResult:
    """One-way ANOVA test with comprehensive validation, effect sizes, and interpretation."""
    _check_scipy()

    result = StatisticalTestResult(test_name="One-Way ANOVA")

    if len(model_metrics) < 2:
        raise ValueError("ANOVA requires at least two groups to compare.")

    # Clean data
    clean_groups = []
    group_names = []
    for name, series in model_metrics.items():
        clean_data = series.dropna()
        if len(clean_data) > 0:
            clean_groups.append(clean_data)
            group_names.append(name)

    result.sample_sizes = {name: len(group) for name, group in zip(group_names, clean_groups)}

    # Validation
    adequate_size, size_warnings = validate_sample_sizes(clean_groups, 3, "ANOVA")
    result.warnings.extend(size_warnings)
    result.assumptions_met['adequate_sample_size'] = adequate_size

    # Normality check for each group
    normality_results = {}
    all_normal = True
    for i, (name, group) in enumerate(zip(group_names, clean_groups)):
        is_normal, norm_details = check_normality(group)
        normality_results[name] = norm_details
        if not is_normal:
            all_normal = False

    result.assumptions_met['normality'] = all_normal
    result.assumption_details['normality'] = normality_results

    if not all_normal:
        result.warnings.append("Some groups violate normality assumption - consider Kruskal-Wallis test")
        result.recommendations.append("Consider using Kruskal-Wallis test for non-normal data")

    # Equal variance check (Levene's test for multiple groups)
    try:
        levene_stat, levene_p = levene(*clean_groups)
        equal_vars = levene_p > alpha
        result.assumptions_met['equal_variances'] = equal_vars
        result.assumption_details['variance_test'] = {
            'levene_statistic': levene_stat,
            'levene_p': levene_p
        }

        if not equal_vars:
            result.warnings.append("Groups have unequal variances")
    except Exception:
        result.warnings.append("Could not test equal variances assumption")

    # Perform ANOVA
    try:
        statistic, p_value = f_oneway(*clean_groups)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

        # Effect size (eta-squared)
        eta_sq, eta_interpretation = eta_squared(clean_groups)
        result.effect_size = eta_sq
        result.effect_size_name = "eta-squared"
        result.effect_size_interpretation = eta_interpretation

    except Exception as e:
        result.warnings.append(f"ANOVA failed: {str(e)}")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    # Generate interpretation
    result.conclusion = _generate_anova_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def kruskal_wallis_test(model_metrics: Dict[str, pd.Series],
                        alpha: float = 0.05) -> StatisticalTestResult:
    """Kruskal-Wallis H-test with comprehensive validation and effect sizes."""
    _check_scipy()

    if len(model_metrics) < 2:
        raise ValueError("Kruskal-Wallis test requires at least two groups to compare.")

    # Clean data
    clean_groups = []
    group_names = []
    for name, series in model_metrics.items():
        clean_data = series.dropna()
        if len(clean_data) > 0:
            clean_groups.append(clean_data)
            group_names.append(name)

    # NEW: Check for empty groups after cleaning
    if not clean_groups or len(clean_groups) < 2:
        raise ValueError(
            "Insufficient data after cleaning. Kruskal-Wallis test requires at least two non-empty groups.")

    # Validation
    adequate_size, size_warnings = validate_sample_sizes(clean_groups, 3, "Kruskal-Wallis test")

    try:
        statistic, p_value = kruskal(*clean_groups)

        # --- FIX STARTS HERE ---
        # Create the result object after computing statistic and p_value
        result = StatisticalTestResult(
            test_name="Kruskal-Wallis H-Test",
            statistic=float(statistic),
            p_value=float(p_value)
        )

        result.sample_sizes = {name: len(group) for name, group in zip(group_names, clean_groups)}
        result.warnings.extend(size_warnings)
        result.assumptions_met['adequate_sample_size'] = adequate_size

        # Effect size (epsilon-squared)
        N = sum(len(group) for group in clean_groups)
        k = len(clean_groups)
        if N > k:
            epsilon_sq = (statistic - k + 1) / (N - k)
            epsilon_sq = max(0, epsilon_sq)  # Ensure non-negative
        else:
            epsilon_sq = 0

        result.effect_size = epsilon_sq
        result.effect_size_name = "epsilon-squared"
        result.effect_size_interpretation = _interpret_eta_squared(epsilon_sq)  # Similar interpretation
        # --- FIX ENDS HERE ---

    except Exception as e:
        # Create a result object even on failure for consistent return type
        result = StatisticalTestResult(
            test_name="Kruskal-Wallis H-Test",
            statistic=float('nan'),
            p_value=float('nan')
        )
        result.warnings.append(f"Kruskal-Wallis test failed: {str(e)}")
        return result

    # Generate interpretation
    result.conclusion = _generate_kruskal_wallis_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    return result


def shapiro_wilk_test(model_metrics: pd.Series, alpha: float = 0.05) -> StatisticalTestResult:
    """Shapiro-Wilk test for normality with comprehensive validation."""
    _check_scipy()

    result = StatisticalTestResult(test_name="Shapiro-Wilk Normality Test")

    if not isinstance(model_metrics, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    clean_data = model_metrics.dropna()
    result.sample_sizes = {'total': len(clean_data)}

    if len(clean_data) < 3:
        result.warnings.append("Shapiro-Wilk test requires at least 3 observations")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    if len(clean_data) > 5000:
        result.warnings.append("Shapiro-Wilk test may not be reliable for very large samples (n>5000)")

    try:
        statistic, p_value = shapiro(clean_data)
        result.statistic = float(statistic)
        result.p_value = float(p_value)

    except Exception as e:
        result.warnings.append(f"Shapiro-Wilk test failed: {str(e)}")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    # Generate interpretation
    result.conclusion = _generate_shapiro_conclusion(result, alpha)
    result.detailed_interpretation = _generate_detailed_interpretation(result, alpha)

    # Add recommendation about sample size
    if len(clean_data) < 20:
        result.recommendations.append("Consider collecting more data for more reliable normality assessment")
    elif len(clean_data) > 1000:
        result.recommendations.append(
            "For large samples, consider visual methods (Q-Q plots) alongside statistical tests")

    return result


# HIGH-LEVEL COMPARISON FUNCTIONS

def compare_two_models(model1_results: pd.Series, model2_results: pd.Series,
                       paired: bool = False, alpha: float = 0.05) -> StatisticalTestResult:
    """Compare two models with intelligent test selection based on data properties."""

    # Clean data first
    if paired:
        # For paired data, remove rows where either value is missing
        combined = pd.DataFrame({'model1': model1_results, 'model2': model2_results})
        combined_clean = combined.dropna()
        clean1 = combined_clean['model1']
        clean2 = combined_clean['model2']
    else:
        clean1 = model1_results.dropna()
        clean2 = model2_results.dropna()

    # Check sample sizes
    if len(clean1) < 6 or len(clean2) < 6:
        result = StatisticalTestResult(test_name="Insufficient Data")
        result.warnings.append("Insufficient data for reliable statistical testing")
        result.recommendations.append("Collect more data (at least 6 samples per group)")
        result.statistic = float('nan')
        result.p_value = float('nan')
        return result

    if paired:
        # For paired comparisons, test the differences
        differences = clean1 - clean2
        result = wilcoxon_signed_rank_test(differences, null_value=0, alpha=alpha)
        result.test_name = "Paired Comparison (Wilcoxon Signed-Rank)"
    else:
        # For independent comparisons, use Mann-Whitney
        result = mann_whitney_test(clean1, clean2, alpha=alpha)
        result.test_name = "Independent Comparison (Mann-Whitney U)"

    return result


def compare_multiple_models(model_results: Dict[str, pd.Series],
                            alpha: float = 0.05,
                            correction_method: str = 'holm') -> Dict[str, Any]:
    """Compare multiple models with automatic multiple comparison correction."""

    model_names = list(model_results.keys())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError("Need at least 2 models to compare")

    # Overall test first (Kruskal-Wallis - more robust than ANOVA)
    overall_result = kruskal_wallis_test(model_results, alpha=alpha)

    results = {
        'overall_test': overall_result,
        'pairwise_comparisons': {},
        'correction_method': correction_method,
        'family_wise_error_rate': alpha,
        'n_comparisons': n_models * (n_models - 1) // 2
    }

    if overall_result.is_significant(alpha):
        # Perform pairwise comparisons
        pairwise_tests = []
        pairwise_names = []

        for i in range(n_models):
            for j in range(i + 1, n_models):
                name1, name2 = model_names[i], model_names[j]
                comparison_name = f"{name1}_vs_{name2}"

                pairwise_result = mann_whitney_test(
                    model_results[name1],
                    model_results[name2],
                    alpha=alpha
                )

                pairwise_tests.append(pairwise_result)
                pairwise_names.append(comparison_name)

                results['pairwise_comparisons'][comparison_name] = pairwise_result

        # Apply multiple comparison correction
        p_values = [test.p_value for test in pairwise_tests]
        corrected_p_values, correction_description = apply_multiple_comparison_correction(
            p_values, method=correction_method
        )

        # Update results with corrected p-values
        for i, (name, test) in enumerate(zip(pairwise_names, pairwise_tests)):
            test.corrected_p_value = corrected_p_values[i]
            test.correction_method = correction_method

        results['correction_description'] = correction_description
        results['significant_comparisons'] = [
            name for name, test in results['pairwise_comparisons'].items()
            if test.is_significant(alpha)
        ]
    else:
        results['message'] = "Overall test not significant - no pairwise comparisons performed"
        results['significant_comparisons'] = []

    return results


# UTILITY FUNCTIONS FOR INTERPRETATION

def _generate_mann_whitney_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Mann-Whitney test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Mann-Whitney U test indicates a statistically significant difference between groups (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Mann-Whitney U test shows no statistically significant difference between groups (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_wilcoxon_conclusion(result: StatisticalTestResult, null_value: float, alpha: float) -> str:
    """Generate conclusion for Wilcoxon test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Wilcoxon signed-rank test indicates the median differs significantly from {null_value} (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Wilcoxon signed-rank test shows no significant difference from {null_value} (p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_anova_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for ANOVA test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"One-way ANOVA indicates significant differences between group means (F={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"One-way ANOVA shows no significant differences between group means (F={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_kruskal_wallis_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Kruskal-Wallis test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Kruskal-Wallis test indicates significant differences between group distributions (H={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f" with {result.effect_size_interpretation} effect size ({result.effect_size_name}={result.effect_size:.3f})"
    else:
        conclusion = f"Kruskal-Wallis test shows no significant differences between group distributions (H={result.statistic:.3f}, p={p_val:.4f})"
        if result.effect_size is not None:
            conclusion += f". Effect size is {result.effect_size_interpretation} ({result.effect_size_name}={result.effect_size:.3f})"

    return conclusion


def _generate_shapiro_conclusion(result: StatisticalTestResult, alpha: float) -> str:
    """Generate conclusion for Shapiro-Wilk test."""
    p_val = result.corrected_p_value if result.corrected_p_value is not None else result.p_value

    if p_val < alpha:
        conclusion = f"Shapiro-Wilk test indicates the sample is not normally distributed (W={result.statistic:.4f}, p={p_val:.4f})"
    else:
        conclusion = f"Shapiro-Wilk test is consistent with normal distribution (W={result.statistic:.4f}, p={p_val:.4f})"

    return conclusion


def _generate_detailed_interpretation(result: StatisticalTestResult, alpha: float) -> str:
    """Generate detailed interpretation of statistical test result."""
    interpretation = [result.conclusion]

    # Sample size information
    if result.sample_sizes:
        sample_info = ", ".join([f"{k}: {v}" for k, v in result.sample_sizes.items()])
        interpretation.append(f"Sample sizes: {sample_info}")

    # Assumption violations
    if result.warnings:
        interpretation.append("Warnings:")
        for warning in result.warnings:
            interpretation.append(f"  - {warning}")

    # Multiple comparison correction
    if result.corrected_p_value is not None:
        interpretation.append(f"P-value corrected for multiple comparisons using {result.correction_method}")

    # Recommendations
    if result.recommendations:
        interpretation.append("Recommendations:")
        for rec in result.recommendations:
            interpretation.append(f"  - {rec}")

    return "\n".join(interpretation)


def _interpret_wilcoxon_r(r: float) -> str:
    """Interpret Wilcoxon r effect size."""
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


# CONVERGENCE ASSESSMENT (enhanced existing functions)

def calculate_autocorr(data: Union[pd.Series, List[float]], lag: int = 1) -> Optional[float]:
    """Calculate autocorrelation of a time series at a specific lag. Enhanced with better error handling."""
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) <= lag:
        return None

    try:
        return data.autocorr(lag)
    except Exception:
        return None


def calculate_averaged_autocorr(histories: List[pd.Series], max_lag: int = 20) -> Tuple[
    List[float], List[float], List[float]]:
    """Compute mean and standard deviation of autocorrelation functions across multiple histories. Enhanced with better validation."""
    if not histories:
        raise ValueError("The list of histories cannot be empty.")

    # Filter out histories that are too short
    valid_histories = [h for h in histories if len(h) > max_lag]

    if not valid_histories:
        raise ValueError(f"No histories long enough for max_lag={max_lag}")

    autocorrs = []
    for history in valid_histories:
        autocorr_values = []
        for lag in range(1, max_lag + 1):
            autocorr = calculate_autocorr(history, lag)
            autocorr_values.append(autocorr if autocorr is not None else np.nan)
        autocorrs.append(autocorr_values)

    autocorrs_np = np.array(autocorrs)
    mean_autocorr = np.nanmean(autocorrs_np, axis=0)
    std_autocorr = np.nanstd(autocorrs_np, axis=0)

    lags = list(range(1, max_lag + 1))

    return lags, mean_autocorr.tolist(), std_autocorr.tolist()


def check_convergence(data: Union[pd.Series, List[float]], window_size: int = 5,
                      autocorr_threshold: float = 0.1) -> bool:
    """Check for convergence using autocorrelation and variance criteria. Enhanced with multiple convergence indicators."""
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    if len(data) < window_size + 1:
        return False

    # Get recent window
    recent_data = data.iloc[-window_size:]

    # Primary criterion: low autocorrelation
    autocorr = calculate_autocorr(recent_data, lag=1)
    if autocorr is None:
        return False

    low_autocorr = abs(autocorr) < autocorr_threshold

    # Secondary criterion: low variance in recent window relative to overall
    if len(data) > window_size:
        recent_variance = recent_data.var()
        overall_variance = data.var()

        if overall_variance > 0:
            low_recent_variance = recent_variance < 0.1 * overall_variance
        else:
            low_recent_variance = True
    else:
        low_recent_variance = False

    return low_autocorr or low_recent_variance


# CONFUSION MATRIX FUNCTION (enhanced)

def get_confusion_matrix_df(predictions: np.ndarray, true_labels: np.ndarray,
                            class_names: Dict[int, str]) -> pd.DataFrame:
    """Generate confusion matrix DataFrame from predictions and true labels."""
    _check_sklearn()

    if len(predictions) != len(true_labels):
        raise ValueError("predictions and true_labels must have the same length")

    class_indices = sorted(class_names.keys())
    cm = confusion_matrix(true_labels, predictions, labels=class_indices)

    return pd.DataFrame(
        cm,
        index=[class_names[i] for i in class_indices],
        columns=[class_names[i] for i in class_indices]
    )


# SUMMARY AND REPORTING FUNCTIONS

def generate_statistical_summary(results: List[StatisticalTestResult]) -> str:
    """Generate a comprehensive summary of multiple statistical test results."""
    if not results:
        return "No statistical test results to summarize."

    summary_lines = ["Statistical Analysis Summary", "=" * 30, ""]

    # Count significant results
    significant_results = [r for r in results if r.is_significant()]

    summary_lines.append(f"Tests performed: {len(results)}")
    summary_lines.append(
        f"Significant results: {len(significant_results)} ({100 * len(significant_results) / len(results):.1f}%)")
    summary_lines.append("")

    # Individual test summaries
    for result in results:
        summary_lines.append(result.get_summary())

        # Add effect size context
        if result.effect_size is not None:
            effect_context = f"  Effect size ({result.effect_size_name}): {result.effect_size:.3f} ({result.effect_size_interpretation})"
            summary_lines.append(effect_context)

        # Add major warnings
        if result.warnings:
            major_warnings = [w for w in result.warnings if "assumption" in w.lower() or "insufficient" in w.lower()]
            for warning in major_warnings:
                summary_lines.append(f"  ⚠ {warning}")

        summary_lines.append("")

    # Overall recommendations
    all_recommendations = [r for result in results for r in result.recommendations]

    if all_recommendations:
        summary_lines.append("Key Recommendations:")
        # Get unique recommendations
        unique_recs = list(set(all_recommendations))
        for rec in unique_recs:
            summary_lines.append(f"  - {rec}")
        summary_lines.append("")

    return "\n".join(summary_lines)


def create_results_dataframe(results: List[StatisticalTestResult]) -> pd.DataFrame:
    """Create a pandas DataFrame summarizing statistical test results."""
    if not results:
        return pd.DataFrame()

    data = []
    for result in results:
        row = {
            'test_name': result.test_name,
            'statistic': result.statistic,
            'p_value': result.p_value,
            'significant': result.is_significant(),
            'effect_size': result.effect_size,
            'effect_size_name': result.effect_size_name,
            'effect_size_interpretation': result.effect_size_interpretation,
            'n_warnings': len(result.warnings),
            'assumptions_met': all(result.assumptions_met.values()) if result.assumptions_met else None
        }

        # Add sample sizes
        if result.sample_sizes:
            for key, value in result.sample_sizes.items():
                row[f'n_{key}'] = value

        # Add corrected p-value if available
        if result.corrected_p_value is not None:
            row['corrected_p_value'] = result.corrected_p_value
            row['correction_method'] = result.correction_method

        data.append(row)

    return pd.DataFrame(data)


# TRAINING STABILITY ASSESSMENT

def assess_training_stability(loss_histories: List[pd.Series],
                              window_size: int = 10) -> Dict[str, Any]:
    """Assess stability of training across multiple runs."""
    if not loss_histories or len(loss_histories) < 2:
        return {'error': 'Need at least 2 training histories for stability assessment'}

    # Find common length (minimum across all histories)
    min_length = min(len(history) for history in loss_histories)
    if min_length < window_size:
        return {'error': f'Training histories too short for window_size={window_size}'}

    # Truncate all histories to same length
    truncated_histories = [history.iloc[:min_length] for history in loss_histories]

    # Convert to array for easier computation
    loss_array = np.array([history.values for history in truncated_histories])

    # Compute stability metrics
    final_losses = loss_array[:, -1]
    final_window_losses = loss_array[:, -window_size:]

    results = {
        'n_runs': len(loss_histories),
        'common_length': min_length,
        'final_loss_mean': np.mean(final_losses),
        'final_loss_std': np.std(final_losses),
        'final_loss_cv': np.std(final_losses) / np.mean(final_losses) if np.mean(final_losses) > 0 else float('inf'),
        'final_window_std_mean': np.mean([np.std(run_window) for run_window in final_window_losses]),
        'between_run_variance': np.var(np.mean(final_window_losses, axis=1)),
        'within_run_variance_mean': np.mean([np.var(run_window) for run_window in final_window_losses])
    }

    # Convergence assessment for each run
    convergence_results = []
    for i, history in enumerate(truncated_histories):
        converged = check_convergence(pd.Series(history), window_size=window_size)
        convergence_results.append(converged)

    results['convergence_rate'] = sum(convergence_results) / len(convergence_results)
    results['converged_runs'] = sum(convergence_results)

    # Stability interpretation
    cv = results['final_loss_cv']
    if cv < 0.05:
        results['stability_assessment'] = 'high'
    elif cv < 0.15:
        results['stability_assessment'] = 'moderate'
    else:
        results['stability_assessment'] = 'low'

    return results