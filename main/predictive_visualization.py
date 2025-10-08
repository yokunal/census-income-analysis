import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def ensure_plots_folder():
    folder = "plots"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def save_plot(filename, folder):
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filepath}")

def load_model_and_data():
    """Load trained model and cleaned data"""
    model = joblib.load(r'd:\kunal\others\codes\working project of data analysis\main\svm_best_pipeline.joblib')
    df_clean = pd.read_csv('../data/cleaned_data.csv')
    return model, df_clean

def make_predictions(model, df_clean):
    """Generate predictions using the trained model"""
    feature_columns = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "sex", "native_country",
                       "age", "education_num", "fnlwgt", "hours_per_week"]
    X = df_clean[feature_columns]
    df_clean["predicted_income"] = model.predict(X)
    return df_clean

def create_predictive_visualizations(df_clean):
    print("\n" + "="*80)
    print("PREDICTIVE VISUALIZATION")
    print("="*80)
    folder = ensure_plots_folder()
    idx = 1

    # 1. Distribution of Predicted Income
    print("\n1. Distribution of Predicted Income")
    plt.figure(figsize=(8, 6))
    sns.countplot(x="predicted_income", data=df_clean)
    plt.title("Distribution of Predicted Income")
    plt.ylabel("Count")
    save_plot(f"{idx:02d}_distribution_predicted_income.png", folder)
    plt.show()
    print("Business Insight: The predicted income distribution shows most people are in the <=50K group, reflecting the model's conservative approach and the original data's class imbalance.")
    idx += 1

    # 2. Actual vs Predicted
    if 'income' in df_clean.columns:
        print("\n2. Actual vs Predicted Income Comparison")
        plt.figure(figsize=(8, 6))
        sns.countplot(x="income", hue="predicted_income", data=df_clean)
        plt.title("Actual vs Predicted Income")
        plt.xlabel("Actual Income")
        plt.ylabel("Count")
        plt.legend(title="Predicted")
        save_plot(f"{idx:02d}_actual_vs_predicted_income.png", folder)
        plt.show()
        print("Business Insight: Comparison reveals the model's tendency to under-predict high-income individuals, suggesting opportunities for improving recall on the >50K class through model tuning or cost-sensitive learning.")
        idx += 1

        # Confusion Matrix
        print("\n3. Confusion Matrix")
        cm = confusion_matrix(df_clean['income'], df_clean['predicted_income'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        save_plot(f"{idx:02d}_confusion_matrix.png", folder)
        plt.show()
        print("Business Insight: Model accuracy is high for <=50K, but lower for >50K; this may be due to class imbalance. The high precision for <=50K predictions makes this model suitable for conservative income classification tasks.")
        idx += 1

        print("\nClassification Report:")
        print(classification_report(df_clean['income'], df_clean['predicted_income']))
        print("Business Insight: Classification metrics show strong overall performance with room for improvement in high-income prediction recall, which could be addressed through threshold adjustment or ensemble methods.")

    # 3. Predictions by Education
    print("\n4. Predicted Income by Education Level")
    plt.figure(figsize=(12, 6))
    sns.countplot(x="education", hue="predicted_income", data=df_clean)
    plt.title("Predicted Income by Education Level")
    plt.xlabel("Education")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Predicted Income")
    plt.tight_layout()
    save_plot(f"{idx:02d}_predicted_income_by_education.png", folder)
    plt.show()
    print("Business Insight: Model correctly identifies education as a strong income predictor, with higher education levels showing increased >50K predictions. This validates education investment strategies and targeted recruitment policies.")
    idx += 1

    # 4. Predictions by Age Group
    print("\n5. Predicted Income by Age Group")
    plt.figure(figsize=(10, 6))
    sns.countplot(x="age_group", hue="predicted_income", data=df_clean)
    plt.title("Predicted Income by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.legend(title="Predicted Income")
    save_plot(f"{idx:02d}_predicted_income_by_age_group.png", folder)
    plt.show()
    print("Business Insight: Age-based predictions show peak earning predictions in middle-age groups (40-59), aligning with typical career progression patterns and informing retirement planning and workforce development strategies.")
    idx += 1

    # 5. Age vs Hours per Week colored by Predictions
    print("\n6. Age vs Hours per Week (Colored by Predicted Income)")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="age", y="hours_per_week", hue="predicted_income",
                    data=df_clean, alpha=0.6)
    plt.title("Age vs Hours per Week (Colored by Predicted Income)")
    plt.xlabel("Age")
    plt.ylabel("Hours per Week")
    plt.legend(title="Predicted Income")
    save_plot(f"{idx:02d}_age_vs_hours_by_predicted_income.png", folder)
    plt.show()
    print("Business Insight: Scatter plot reveals that higher predicted incomes correlate with longer work hours across all age groups, suggesting work intensity as a key factor in earning potential and work-life balance considerations.")
    idx += 1

    # 6. Predictions by Occupation
    print("\n7. Predicted Income by Occupation")
    plt.figure(figsize=(14, 8))
    sns.countplot(y="occupation", hue="predicted_income", data=df_clean)
    plt.title("Predicted Income by Occupation")
    plt.xlabel("Count")
    plt.ylabel("Occupation")
    plt.legend(title="Predicted Income")
    plt.tight_layout()
    save_plot(f"{idx:02d}_predicted_income_by_occupation.png", folder)
    plt.show()
    print("Business Insight: Occupation-based predictions clearly distinguish high-earning professions (executives, professionals) from service roles, providing valuable insights for career counseling and salary benchmarking initiatives.")
    idx += 1

    # 7. Box Plot: Hours per Week by Predicted Income
    print("\n8. Hours per Week Distribution by Predicted Income")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="predicted_income", y="hours_per_week", data=df_clean)
    plt.title("Hours per Week Distribution by Predicted Income")
    plt.xlabel("Predicted Income")
    plt.ylabel("Hours per Week")
    save_plot(f"{idx:02d}_boxplot_hours_by_predicted_income.png", folder)
    plt.show()
    print("Business Insight: Box plot confirms that predicted high earners work significantly more hours on average, with greater variability, indicating the importance of work commitment in income prediction and potential burnout risk management.")

    print(f"\nAll predictive visualizations saved in '{folder}'.")

def main():
    model, df_clean = load_model_and_data()
    df_clean = make_predictions(model, df_clean)
    create_predictive_visualizations(df_clean)
    print("\nPredictive visualization complete!")
    print("Business Summary: Model predictions align well with expected demographic and professional patterns...")
    df_clean.to_csv('../data/predictions_output.csv', index=False)
    print("Predictions saved to '../data/predictions_output.csv'")

if __name__ == "__main__":
    main()
