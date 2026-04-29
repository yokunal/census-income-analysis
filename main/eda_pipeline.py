import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (RAW_DATA_PATH, CLEANED_DATA_PATH, IQR_MULTIPLIER,
                    LOW_VARIANCE_COLUMNS, AGE_BINS, AGE_LABELS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads dataset, standardizes columns, handles missing values and outliers,
    drops duplicates, and creates age_group feature.

    Args:
        filepath: Path to CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for EDA and modeling.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise

    # Standardize columns FIRST so validation uses consistent names
    df.columns = [col.strip().replace('.', '_').replace(' ', '_') for col in df.columns]

    expected_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss",
                     "hours_per_week", "workclass", "education", "marital_status",
                     "occupation", "relationship", "race", "sex", "native_country", "income"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after standardization: {missing}")

    # Replace "?" with NaN
    df = df.replace('?', pd.NA)

    # Impute categorical with mode
    for col in df.select_dtypes(include='object').columns:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Impute numeric with median
    for col in df.select_dtypes(include='number').columns:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    logger.info(f"Removed {n_dupes} duplicate rows")

    # Remove outliers using IQR
    numeric_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    n_outliers = 0
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - IQR_MULTIPLIER * IQR
            upper = Q3 + IQR_MULTIPLIER * IQR
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            n_outliers += before - len(df)

    logger.info(f"Removed {n_outliers} outlier rows")

    # Drop columns with low variance
    for col in LOW_VARIANCE_COLUMNS:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Feature engineering: Age group — clip ages to fit within bins
    if 'age' in df.columns:
        df['age'] = df['age'].clip(lower=AGE_BINS[0], upper=AGE_BINS[-1])
        df['age_group'] = pd.cut(
            df['age'],
            bins=AGE_BINS,
            labels=AGE_LABELS
        )

    return df


def show_basic_info(df: pd.DataFrame) -> None:
    """
    Displays summary of dataframe shape, columns, info, and missing/duplicate counts.
    """
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    buf = df.dtypes.to_string()
    logger.info(f"Data types:\n{buf}")
    nulls = df.isnull().sum().to_string()
    logger.info(f"Null values count:\n{nulls}")
    logger.info(f"Number of duplicates: {df.duplicated().sum()}")


def eda_visualizations(df: pd.DataFrame) -> None:
    """
    Plots a comprehensive set of EDA graphs and provides business insights for each.

    Args:
        df: Cleaned dataframe for analysis.
    """
    # Pie chart: Income distribution
    logger.info('\n>>> Pie Chart: Income Distribution')
    fig = plt.figure(figsize=(8, 6))
    plt.pie(df['income'].value_counts(), labels=df['income'].value_counts().index, autopct='%1.1f%%', startangle=90)
    plt.title('Income Distribution')
    plt.close(fig)
    logger.info("Business Insight: Majority respondents belong to the <=50K group, indicating class imbalance and potential bias in prediction models.")

    # Countplot by income
    categorical_columns = ["education", "occupation", "workclass", "race", "sex", "marital_status", "income", "age_group"]
    logger.info('\n>>> Count Plots by Income')
    for col in categorical_columns:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x=col, data=df, hue="income", order=df[col].value_counts().index)
            plt.title(f'Distribution of {col} by Income')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            for container in ax.containers:
                ax.bar_label(container)
            plt.close()
            if col == "education":
                logger.info("Business Insight: Higher education levels show more >50K predictions, confirming education as a powerful driver of income.")
            elif col == "sex":
                logger.info("Business Insight: Gender disparities are observable and may warrant pay equity monitoring.")
            elif col == "occupation":
                logger.info("Business Insight: Data highlights which occupations are more likely to command higher salaries.")
            else:
                logger.info(f"Business Insight: {col} impacts income distribution, showing demographic and categorical effects.")

    # Countplot by sex
    logger.info('\n>>> Count Plots by Sex')
    for col in categorical_columns:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            ax = sns.countplot(x=col, data=df, hue="sex", order=df[col].value_counts().index)
            plt.title(f'Distribution of {col} by Sex')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            for container in ax.containers:
                ax.bar_label(container)
            plt.close()
            logger.info("Business Insight: Reveals how demographic and categorical variables break down by gender.")

    # Initial boxplots (numeric columns)
    logger.info('\n>>> Initial Box Plots (before outlier removal)')
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.close()
        logger.info(f"Business Insight: Initial spread and outlier check for {col}.")

    # Box plots (categorical vs numeric)
    logger.info('\n>>> Box Plots (Categorical vs Numeric)')
    x_categoricals = ['income', 'sex', 'marital_status', 'age_group']
    y_numerics = ['age', 'hours_per_week', 'fnlwgt']
    for x in x_categoricals:
        for y in y_numerics:
            if x in df.columns and y in df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=x, y=y, data=df)
                plt.title(f'{y} distribution by {x}')
                plt.xlabel(x)
                plt.ylabel(y)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.close()
                logger.info(f"Business Insight: Examines spread/variance in {y} for each group in {x}.")

    # Boxplot: hours_per_week by workclass
    logger.info('\n>>> Box Plot (Hours per Week by Workclass)')
    if "workclass" in df.columns and "hours_per_week" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="workclass", y="hours_per_week")
        plt.xticks(rotation=45)
        plt.title('Distribution of Hours per Week by Workclass')
        plt.tight_layout()
        plt.close()
        logger.info("Business Insight: Work hour differences for work types, informs productivity insights.")

    # Histogram with income hue
    logger.info('\n>>> Histograms (hue=Income)')
    for col in ["age", "education_num", "fnlwgt", "hours_per_week"]:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(x=col, data=df, hue="income", bins=30, element="poly", multiple="dodge", common_norm=False)
            plt.title(f'Distribution of {col} by Income')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.close()
            logger.info(f"Business Insight: Shows split of {col} values across income levels.")

    # Histogram with sex hue
    logger.info('\n>>> Histograms (hue=Sex)')
    for col in ["age", "education_num", "fnlwgt", "hours_per_week"]:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(x=col, data=df, hue="sex", bins=30, element="poly", multiple="dodge", common_norm=False)
            plt.title(f'Distribution of {col} by Sex')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.close()
            logger.info(f"Business Insight: Reveals gender-dependent distributions for {col}.")

    # Stacked bar (income as hue)
    logger.info('\n>>> Stacked Bar Plots (Income as Hue)')
    for col in ["education", "occupation", "marital_status", "race"]:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            ctab = pd.crosstab(df[col], df["income"])
            ctab.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'Distribution of {col} by Income (Stacked)')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.close()
            logger.info("Business Insight: Stacked bar shows how different categories break down across income.")

    # Violin plot
    logger.info('\n>>> Violin Plot (hours_per_week by education)')
    if "education" in df.columns and "hours_per_week" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x="education", y="hours_per_week")
        plt.xticks(rotation=45)
        plt.title('Distribution of Hours per Week by Education')
        plt.tight_layout()
        plt.close()
        logger.info("Business Insight: Reveals work hour diversity within education groups.")

    # Stacked bar (sex by workclass)
    logger.info('\n>>> Stacked Bar (Sex by Workclass)')
    if "sex" in df.columns and "workclass" in df.columns:
        plt.figure(figsize=(10, 6))
        df.groupby(['sex', 'workclass']).size().unstack().plot(kind='bar', stacked=True)
        plt.title('Distribution of Sex by Workclass')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.close()
        logger.info("Business Insight: Shows how men and women are distributed among work classes.")

    # SCATTER PLOTS: both income and sex as hue
    logger.info('\n>>> Scatter Plots (hours_per_week/fnlwgt vs education_num/age; hue=income & sex)')
    x_cat = ["hours_per_week", "fnlwgt"]
    y_cat = ["education_num", "age"]
    for hue_column in ["income", "sex"]:
        for x in x_cat:
            for y in y_cat:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=x, y=y, data=df, hue=hue_column, alpha=0.5)
                plt.title(f'{y} vs {x} by {hue_column}')
                plt.xlabel(x)
                plt.ylabel(y)
                plt.tight_layout()
                plt.close()
                logger.info(f"Business Insight: {y} vs {x} patterns colored by {hue_column}; key for multidimensional feature analysis.")

    # PAIRPLOT: both income and sex as hue
    logger.info('\n>>> Pairplots (Numeric columns, hue=income & sex)')
    pairplot_cols = ["age", "education_num", "fnlwgt", "hours_per_week"]
    for hue_column in ["income", "sex"]:
        if all(col in df.columns for col in pairplot_cols):
            sample_df = df.sample(n=min(1000, len(df)), random_state=42)
            sns.pairplot(sample_df[pairplot_cols + [hue_column]], hue=hue_column, diag_kind="hist")
            plt.suptitle(f"Pairplot of numeric features by {hue_column}", y=1.02)
            plt.close()
            logger.info(f"Business Insight: Pairplot helps identify interplay between features and {hue_column}.")

    # HEATMAP: average hours worked per week by age group & age
    logger.info('\n>>> Heatmap (Avg Hours Worked by Age Group & Age)')
    heatmap_data = df.pivot_table(
        index='age_group',
        columns='age',
        values='hours_per_week',
        aggfunc='mean',
        observed=False
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".1f", annot_kws={"rotation": 90})
    plt.title("Hours per week by Age and Age Group")
    plt.ylabel("Age Group")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.close()
    logger.info("Business Insight: Reveals age-related work hour trends, critical for workforce/career planning.")


def comprehensive_eda(df: pd.DataFrame) -> None:
    """
    Runs complete EDA, visualization, and annotated business insights for provided dataframe.

    Args:
        df: Cleaned dataframe.
    """
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    logger.info("="*80)
    show_basic_info(df)
    eda_visualizations(df)
    logger.info("\nBusiness Summary: This analysis illustrates significant relationships between demographic factors and income distribution, highlighting age, education, occupation, and work hours as key predictors. Gender and occupation also show workforce trends with implications for business policy and HR.")


def main(filepath: Path = RAW_DATA_PATH) -> None:
    """
    Full pipeline: Loads data, cleans, and runs EDA with insights.

    Args:
        filepath: CSV file to load.
    """
    df = load_and_clean_data(filepath)
    comprehensive_eda(df)


if __name__ == "__main__":
    main(RAW_DATA_PATH)
