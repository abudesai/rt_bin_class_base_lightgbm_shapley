LightGBM Classifier with Shapley explanations for Binary Classification - Base problem category as per Ready Tensor specifications.

- lightgbm
- ensemble
- binary classification
- XAI
- interpretable
- Shap
- Shapley values
- python
- pandas
- numpy
- sklearn
- scikit-optimize
- HPT
- flask
- nginx
- uvicorn
- docker

This is an explainable version of LightGBM classifier, with Shapley values for model interpretability.

Model explainability is provided using Shapley values. Local explanations at each instance can be obtained.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) is conducted on LightGBM's 4 hyperparameters: boosting_type, n_estimators, num_leaves, and learning_rate.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, shap package for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides three endpoints- /ping for health check, /infer for predictions in real time and /explain to generate local explanations.
