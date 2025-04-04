# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This binary classification model was developed in April 2025 as part of the Deploying a Scalable ML Pipeline with FastAPI project. The model was built using the scikit-learn `RandomForestClassifier`, with slight hyperparameters tuning specified within `param_grid` in `train_model.py`. It is the first official version of the model (v1.0.0).

The model is designed to predict whether an individual earns more than $50K annually, based on a combination of demographic and socio-economic features. It uses a mix of categorical and continuous variables sourced from the U.S. Census dataset. Categorical features are transformed using one-hot encoding with scikit-learn’s `OneHotEncoder`, while the target label is binarized using `LabelBinarizer`. These transformations are implemented in a reusable `process_data` function, which ensures consistency between the training and inference phases.

No explicit fairness constraints were applied during training, but the model pipeline includes functionality to evaluate performance across slices of the data—such as by education level or race—allowing for basic fairness analysis and bias detection. Overall, the model emphasizes modular design, reproducibility, and readiness for scalable deployment.
## Intended Use
This model is intended to be used for educational and demonstration purposes to show the construction, evaluation, and deployment of machine learning models using production-grade tools. It is not intended for real-world financial, hiring, or policy decisions without further ethical, legal, and performance evaluations.

## Training Data
The training data is derived from the UCI Adult Census Income dataset. It contains demographic features such as age, education, workclass, marital status, occupation, relationship, race, sex, and native country. The target label is whether the individual's income exceeds $50K annually.

Sensitive attributes such as race and sex are included for the purpose of fairness auditing and performance analysis across slices.

## Evaluation Data
Evaluation data consists of a held-out test split from the same UCI dataset, ensuring consistent feature distribution and label representation. Slice-based evaluations were also conducted across categorical values of features like workclass, education, marital-status, occupation, relationship, race, sex, and native-country

Preprocessing:

The dataset was processed using a dedicated process_data function, which:

- Encodes categorical features using `OneHotEncoder`.

- Normalizes labels using `LabelBinarizer`.

- Concatenates numerical and encoded categorical features into a final feature matrix. The same preprocessing pipeline is used at inference time to ensure consistency.

## Metrics
The model was evaluated using standard classification metrics:

- Precision: 0.7904

- Recall: 0.6084

- F1-score: 0.6876

In addition to overall performance, slice-based performance analysis was conducted to evaluate fairness and robustness, stored within `slice_output.txt`:

Examples of slice performance:

- Workclass = Private

  - F1-score: 0.6717

- Education = Bachelors

  - F1-score: 0.7883

- Sex = Female

  - F1-score: 0.6954

- Sex = Male

  - F1-score: 0.6862

- Race = White

  - F1-score: 0.6926

- Occupation = Exec-managerial

  - F1-score: 0.8041

Extreme values like F1 = 0.0 or 1.0 were observed for slices with very small counts and should be interpreted with caution.

## Ethical Considerations
**Bias and Fairness:**

Performance varies across demographic slices, especially for underrepresented groups. For example, individuals with education levels such as "7th-8th" or "Married-AF-spouse" had F1 scores of 0.0 due to low recall or very small sample sizes. This suggests the model may not generalize well to certain groups.

**Privacy:** 

The model was trained on publicly available anonymized data, but care should be taken when deploying similar models on sensitive personal data.

**Misuse:**

The model should not be used to make decisions affecting people's livelihoods without significant improvements, fairness audits, and regulatory compliance.

## Caveats and Recommendations
1. Small Sample Slices: 
   - Slices such as "Without-pay", "Preschool", and "Cambodia" show perfect scores (F1 = 1.0), likely due to having only a few samples. These results are not statistically significant.

2. Feature Drift:
   - In real-world deployment, changes in demographic distributions over time could reduce model performance. Regular re-evaluation is recommended.

3. Improvement Areas: 
   - Techniques such as upsampling underrepresented slices, fairness-aware training, and additional model interpretability methods could improve trust and robustness.