import logging
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_PATH
from main.eda_pipeline import load_and_clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def test_gender_income(df: pd.DataFrame) -> Dict[str, Any]:
    """Chi-squared test: gender vs income."""
    contingency = pd.crosstab(df["sex"], df["income"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    # Cramer's V effect size
    n = contingency.sum().sum()
    cramers_v = (chi2 / (n * (min(contingency.shape) - 1))) ** 0.5
    result = "Reject H0" if p < 0.05 else "Fail to reject H0"
    logger.info(f"Gender vs Income — chi2={chi2:.4f}, p={p:.4f}, Cramer's V={cramers_v:.4f}")
    return {
        "test": "Chi-squared",
        "variable": "gender vs income",
        "statistic": chi2,
        "p_value": p,
        "effect_size": cramers_v,
        "effect_size_metric": "Cramer's V",
        "result": result
    }


def test_education_hours(df: pd.DataFrame) -> Dict[str, Any]:
    """T-test: hours per week — Bachelors vs HS-grad."""
    valid_educations = df["education"].unique()
    for edu in ["Bachelors", "HS-grad"]:
        if edu not in valid_educations:
            raise ValueError(f"Education value '{edu}' not found in data. Available: {list(valid_educations)}")
    group1 = df[df["education"] == "Bachelors"]["hours_per_week"]
    group2 = df[df["education"] == "HS-grad"]["hours_per_week"]
    t_stat, p = stats.ttest_ind(group1, group2)
    # Cohen's d effect size
    pooled_std = ((group1.std() ** 2 + group2.std() ** 2) / 2) ** 0.5
    cohens_d = (group1.mean() - group2.mean()) / pooled_std
    result = "Reject H0" if p < 0.05 else "Fail to reject H0"
    logger.info(f"Education Hours — t={t_stat:.4f}, p={p:.4f}, Cohen's d={cohens_d:.4f}")
    return {
        "test": "T-test",
        "variable": "Bachelors vs HS-grad hours",
        "statistic": t_stat,
        "p_value": p,
        "effect_size": cohens_d,
        "effect_size_metric": "Cohen's d",
        "result": result
    }


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run all hypothesis tests and return results as a DataFrame."""
    results = [
        test_gender_income(df),
        test_education_hours(df),
    ]
    results_df = pd.DataFrame(results)
    logger.info("\n" + results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    df = load_and_clean_data()
    results = run_hypothesis_tests(df)
    results.to_csv(Path(__file__).parent.parent / "data" / "hypothesis_results.csv", index=False)
    logger.info("Hypothesis results saved to data/hypothesis_results.csv")
