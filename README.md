# Income Prediction Analysis - Machine Learning Pipeline

A comprehensive data science project that predicts income levels using census data through exploratory data analysis, statistical hypothesis testing, and machine learning modeling.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline to predict whether an individual's income exceeds $50,000 annually based on demographic and employment features. The analysis includes comprehensive exploratory data analysis, statistical hypothesis testing, and predictive modeling with business insights.

### Key Features
- **Data Cleaning & Preprocessing**: Automated pipeline for missing value imputation and outlier removal
- **Exploratory Data Analysis**: 20+ visualizations revealing income patterns across demographics
- **Statistical Testing**: Hypothesis tests examining relationships between gender, education, and income
- **Machine Learning**: Multiple model comparison with hyperparameter tuning
- **Business Insights**: Actionable interpretations for each analysis step

## 🎯 Business Problem

Understanding income distribution patterns is crucial for:
- **Policy Making**: Informing economic and social policies
- **Market Research**: Identifying target demographics for financial products
- **HR Analytics**: Salary benchmarking and compensation planning
- **Social Research**: Analyzing income inequality factors

## 📁 Project Structure
```
income-prediction-analysis/
├── data/
│ ├── raw.csv # Original dataset
│ ├── cleaned_data.csv # Processed dataset
│ └── predictions_output.csv # Model predictions
├── main/
│ ├── eda_pipeline.py # Data cleaning & EDA
│ ├── hypothesis_testing.py # Statistical analysis
│ ├── ml_modeling.py # Model training & tuning
│ ├── predictive_visualization.py # Prediction analysis
│ └── svm_best_pipeline.joblib # Trained model
├── notebooks/
│ └── final.ipynb # Complete analysis demo
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── LICENSE # MIT license
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. -**Clone the repository**

git clone https://github.com/yourusername/income-prediction-analysis.git
cd income-prediction-analysis


2. **Install dependencies**

pip install -r requirements.txt


3. **Run the complete analysis**

Data cleaning and EDA
python main/eda_pipeline.py

Statistical hypothesis testing
python main/hypothesis_testing.py

Machine learning modeling
python main/ml_modeling.py

Predictive visualization
python main/predictive_visualization.py


### Jupyter Notebook Demo

jupyter notebook notebooks/final.ipynb


## 📈 Key Results

### Model Performance
- **Best Model**: Support Vector Machine (SVM)
- **Accuracy**: 85%
- **Precision (<=50K)**: 88%
- **Precision (>50K)**: 71%
- **F1-Score**: 0.84 (weighted average)

### Statistical Findings
- **Education Impact**: Strong correlation between education level and income (p < 0.001)
- **Gender Analysis**: Significant differences in work hours and income distribution by gender
- **Age Patterns**: Peak earning predictions in 40-59 age groups

### Business Insights
- 78% of predictions fall into <=50K category, reflecting class imbalance
- Higher education levels show 3x higher rates of >50K predictions
- Work hours strongly correlate with income predictions across all demographics

## 🛠️ Technical Implementation

### Data Processing
- **Missing Value Handling**: Mode imputation for categorical, median for numerical
- **Outlier Removal**: IQR-based filtering for numerical features
- **Feature Engineering**: Age group categorization, column standardization

### Machine Learning Pipeline
- **Models Tested**: Random Forest, SVM, Logistic Regression, Gradient Boosting, KNN
- **Validation**: Stratified K-Fold cross-validation
- **Hyperparameter Tuning**: RandomizedSearchCV with 100 iterations
- **Feature Selection**: 12 key demographic and employment features

### Visualization Suite
- **EDA**: 15+ plot types (distributions, correlations, comparisons)
- **Statistical Tests**: Chi-square, t-tests, ANOVA with interpretations
- **Model Analysis**: Confusion matrix, feature importance, prediction distributions

## 📊 Dataset Information

**Source**: Adult Census Income Dataset
- **Size**: ~32,000 records (after cleaning)
- **Features**: 14 attributes including age, education, occupation, hours worked
- **Target**: Binary classification (<=50K, >50K)
- **Missing Data**: Handled through imputation strategies

### Key Features
- `age`: Age of individual
- `education`: Highest education level achieved
- `occupation`: Job category
- `hours_per_week`: Weekly work hours
- `sex`: Gender
- `marital_status`: Relationship status

## 🔍 Analysis Highlights

### Exploratory Data Analysis
- Income distribution shows 76% in <=50K category
- Education strongly correlates with income levels
- Work hours vary significantly between income groups
- Gender-based patterns in occupation and income

### Hypothesis Testing Results
- **H1**: Education level significantly affects income (✓ Confirmed)
- **H2**: Gender influences work hours and income distribution (✓ Confirmed)
- **H3**: Age groups show different income patterns (✓ Confirmed)

### Model Insights
- SVM achieves best balance of precision and recall
- Model conservative in high-income predictions (addressing class imbalance)
- Feature importance: education, occupation, age, hours_per_week

## 💡 Future Improvements

- **Model Enhancement**: Ensemble methods, deep learning approaches
- **Feature Engineering**: Interaction terms, polynomial features
- **Class Balancing**: SMOTE, cost-sensitive learning
- **Deployment**: REST API, web dashboard
- **Real-time Updates**: Streaming data pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- LinkedIn: [Kunal Jha](https://www.linkedin.com/in/kunal-jha-743863282/)
- GitHub: [@yokunal](http://github.com/yokunal)
- Email: jhakunal471@gmail.com

## 🙏 Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Inspiration: Real-world income analysis challenges
- Tools: Python data science ecosystem (pandas, scikit-learn, matplotlib)

---

⭐ **If you found this project helpful, please give it a star!**
