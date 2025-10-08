# # ml model traing and scoring and tuning the bet model script 1 and 2 combined
# # Define features and target
cat_features = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
num_features = ["age", "education_num", "fnlwgt", "hours_per_week"]
target = "income"
feature_columns = cat_features + num_features
X = df[feature_columns]
y = df[target]

# Data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features)
])

# Models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(random_state=42)
}

results = []

# Train and score all models, keep their pipelines too
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Score': test_score,
        'Pipeline': pipeline
    })
    print(f"{name}:  CV Score: {cv_scores.mean():.4f}, Test Score: {test_score:.4f}")

results_df = pd.DataFrame(results).sort_values("Test Score", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_pipeline = results_df.iloc[0]["Pipeline"]
print(f"\nBest Model: {best_model_name}")

# Define hyperparameter grid for tuning
param_grids = {
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__bootstrap": [True, False]
    },
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"]
    },
    "SVM": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__kernel": ["linear", "rbf", "poly"],
        "classifier__gamma": ["scale", "auto"]
    }
}

# Fine-tune only the best model
search = RandomizedSearchCV(
    best_pipeline,
    param_distributions=param_grids[best_model_name],
    n_iter=10,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1
)
search.fit(X_train, y_train)
test_score = search.best_estimator_.score(X_test, y_test)
print(f"\nFine-tuned {best_model_name}:")
print(f"  Best CV score: {search.best_score_:.4f}")
print(f"  Test Score: {test_score:.4f}")
print(f"  Best params: {search.best_params_}")
