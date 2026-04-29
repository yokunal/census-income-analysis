# Census Income Analysis

This project delivers a production-grade machine learning pipeline that predicts whether an individual's annual income exceeds $50,000 based on demographic and employment features from census data. The analysis enables HR professionals, policy makers, and financial analysts to identify which factors — education, occupation, age, work hours — drive higher earnings, and to act on those patterns for workforce planning, compensation strategy, and talent development.

## Dataset

- **Source**: UCI Machine Learning Repository — Adult Census Income Dataset
- **Size**: 32,561 records (18,991 after cleaning and outlier removal)
- **Features**: 14 demographic and employment attributes (age, education, occupation, work hours, marital status, native country, and others)
- **Target**: Binary classification — income ≤ $50K or > $50K

**Key fields:**
- `age`, `fnlwgt`, `education_num`, `hours_per_week` — numeric demographic and employment measures
- `education`, `workclass`, `occupation`, `relationship`, `race`, `sex`, `native_country`, `marital_status` — categorical workforce attributes
- `income` — binary target variable (>50K / ≤50K)

## Business Questions Answered

1. Which education levels are most strongly associated with high-income outcomes?
2. Do gender and occupation create measurable disparities in income and work hours?
3. At what age does income peak, and how does it vary across age groups?
4. Which occupation categories produce the highest proportion of >50K earners?
5. How do work hours correlate with income across different demographic segments?
6. What workforce or policy actions can meaningfully shift income trajectories?

## Project Architecture

```
config.py             — Centralized project constants: file paths, feature lists,
                        hyperparameters, and preprocessing rules. Ensures all modules
                        reference a single source of truth, eliminating hardcoded paths.
main/
  eda_pipeline.py     — Loads raw census data, standardizes columns, imputes missing
                        values, removes outliers, and generates 20+ EDA visualizations
                        with business commentary.
  hypothesis_testing.py — Runs chi-squared tests and t-tests to validate whether
                        education, gender, and age groups are statistically associated
                        with income differences.
  ml_modeling.py      — Trains and compares 5 models (Logistic Regression, Random
                        Forest, Gradient Boosting, KNN, SVM) via stratified K-fold
                        cross-validation, tunes the best model, and saves artifacts.
  predictive_visualization.py — Loads the trained model to generate prediction
                        distributions, confusion matrices, and feature importance plots.
  streamlit_app.py    — Interactive dashboard for exploring EDA results, statistical
                        findings, model metrics, and uploading new data for predictions.
main.py               — CLI entry point: python main.py --steps [eda|hypothesis|model|
                        visualize|all] runs one or all pipeline stages in sequence.
```

## Key Findings

- **Education is the single strongest predictor of income.** Individuals with bachelor's degrees or higher are approximately 3 times more likely to earn >50K compared to high school graduates — a statistically significant relationship (p < 0.001) confirmed across all modeling approaches.
- **The workforce is imbalanced: 76–78% of individuals earn ≤50K.** This majority represents the largest opportunity for targeted interventions in workforce policy, financial product design, and education access programs.
- **Peak earning ages are 40–59.** Income rises through the 30s and 40s, plateaus in the 50s, and declines thereafter. This suggests mid-career professionals carry the highest retention risk and replacement cost if lost.
- **Occupational sorting contributes to gender income disparities.** Chi-squared testing (p < 0.001) confirms that gender, occupation, and income are not independent. Pay equity audits must examine job family-level gaps rather than overall gender averages.
- **Work hours correlate with income but vary by occupation.** Some professions show long hours without proportionate high earnings — a signal for HR to benchmark workload expectations against industry compensation standards.
- **Workclass and native country add measurable signal.** Both features are retained through preprocessing and appear in the feature set, adding nuance to predictions beyond education and age alone.

## Analyst Recommendations

1. **Expand employer-sponsored education and skills programs.** The strong education-to-income correlation means tuition assistance, apprenticeships, and professional certifications carry high ROI for both recruitment and retention — especially for populations with HS-grad or some-college backgrounds where the income gap is widest.
2. **Design targeted retention strategies for the 40–59 age cohort.** With peak earning concentrated in this group, succession planning, mentorship programs, and competitive compensation packages for mid-career professionals yield outsized business value compared to early-career investment alone.
3. **Conduct pay equity audits at the job family level, not overall gender averages.** The data confirms structural occupational sorting by gender. Analyzing compensation gaps within occupation categories — not just across all employees — surfaces the root causes of inequity that aggregate statistics obscure.
4. **Benchmark workload expectations by occupation before setting compensation bands.** The correlation between hours worked and income is strong but varies by profession. HR teams should ensure overtime culture and actual workload in high-hour roles are matched with appropriate pay to reduce retention risk.
5. **Track occupation-level trends for strategic workforce planning.** Certain job categories consistently produce >50K incomes regardless of individual demographics. Monitoring which occupations are growing, declining, or shifting in skill requirements informs training investments and long-term talent strategy.

## SQL Analysis

Documented SQL queries against the cleaned dataset reproduce the key EDA and statistical findings and can be run directly to explore business questions:

- Income distribution by education level
- High-income rate by occupation
- Average work hours by gender and income bracket
- Income breakdown by age group
- Top 10 occupations by high-income rate with statistical relevance filters

These queries provide a complementary analytical surface for stakeholders who prefer SQL over Python.

## Technical Stack

- **pandas** — data loading, cleaning, and transformation of structured census records
- **numpy** — numerical operations underlying all statistical and ML computations
- **scikit-learn** — preprocessing pipelines (ColumnTransformer), model training, cross-validation, and hyperparameter tuning (RandomizedSearchCV)
- **scipy** — statistical hypothesis testing (chi-squared, t-tests, effect size estimation)
- **streamlit** — interactive dashboard for non-technical stakeholders to explore findings
- **joblib** — serialization of trained model pipelines for deployment without environment re-training

## How to Run

```bash
# Step 1: Clone the repository
git clone https://github.com/yokunal/census-income-analysis.git
cd census-income-analysis

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the full analysis pipeline (EDA → hypothesis → model → visualize)
python main.py --steps all

# Step 4: Launch the Streamlit dashboard
streamlit run main/streamlit_app.py
```

Individual pipeline stages can be run independently:
```bash
python main.py --steps eda        # Exploratory data analysis
python main.py --steps hypothesis # Statistical tests
python main.py --steps model      # Train and tune ML models
python main.py --steps visualize  # Prediction analysis and plots
```

## Testing

A pytest suite validates core pipeline behavior:

```bash
pytest tests/ -v
```

Tests cover:
- Hypothesis test result schema (correct column names, data types, and statistical interpretation)
- Imputation robustness (raises on missing values in critical feature columns)
- Feature loading integrity (raises on unexpected column sets)
- Model prediction interface (raises on missing required input fields)

All 4 tests pass with the current codebase.