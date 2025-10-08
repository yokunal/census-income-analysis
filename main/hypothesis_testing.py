# hypothesis_testing.py

import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from eda_pipeline import load_and_clean_data

def run_hypothesis_tests(df):
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)

    # 1. Gender vs. Income (Chi-Squared Test)
    print("\nTest 1: Is there an association between gender and income group?")
    contingency = pd.crosstab(df['sex'], df['income'])
    print("Contingency Table:\n", contingency)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"Chi2 statistic: {chi2:.3f}, p-value: {p:.4f}")
    if p < 0.05:
        print("Result: Reject H0. Significant association between gender and income group.\n")
    else:
        print("Result: Fail to reject H0. No significant association found.\n")

    # 2. Hours worked: Bachelors vs HS-grad (T-test)
    print("Test 2: Do Bachelors degree holders work more hours per week than HS graduates?")
    if 'Bachelors' in df['education'].values and 'HS-grad' in df['education'].values:
        bachelor = df[df['education'] == 'Bachelors']['hours_per_week']
        highschool = df[df['education'] == 'HS-grad']['hours_per_week']
        stat, p = ttest_ind(bachelor, highschool, equal_var=False)
        print(f"t-statistic: {stat:.3f}, p-value: {p:.4f}")
        print(f"Mean hours - Bachelors: {bachelor.mean():.2f}, HS-grad: {highschool.mean():.2f}")
        if p < 0.05:
            print("Result: Reject H0. Significant difference in hours between education levels.\n")
        else:
            print("Result: Fail to reject H0. No significant difference in hours between groups.\n")
    else:
        print("Required education categories not found in data.\n")

if __name__ == "__main__":
    # Choose your data file path
    filepath = r'D:\kunal\others\codes\working project of data analysis\data\raw.csv'

    # Load and clean data
    df = load_and_clean_data(filepath)

    # Run hypothesis tests on cleaned data
    run_hypothesis_tests(df)
