Cardiovascular Disease Prediction (CVD)

Predict risk of Heart_Disease from lifestyle and clinical survey features.
The notebook includes EDA, visualizations, and ML models (Logistic Regression, Random Forest, XGBoost) with a Colab-friendly pipeline.


Dataset

File: CVD_cleaned.csv 

Target: Heart_Disease (Yes/No, converted to 1/0)

Example features:

Categorical: General_Health, Checkup, Exercise, Skin_Cancer, Other_Cancer, Depression, Diabetes, Arthritis, Sex, Age_Category, Smoking_History

Numerical: Height_(cm), Weight_(kg), BMI, Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption

Repository structure

├─ notebooks/
│  └─ Cardiovascular_disease_prediction.ipynb   # main Colab notebook
├─ data/                                        # put CVD_cleaned.csv here (local/Colab)
├─ .github/workflows/ci.yml                     # optional smoke check
├─ requirements.txt
├─ .gitignore
└─ README.md



Problem & target

Binary classification: predict Heart_Disease where:

Heart_Disease: “Yes” → 1, “No” → 0

📊 Data (expected columns)

The notebook auto-detects categorical vs numeric columns. It supports columns typically seen in CVD datasets, e.g.:

Categorical examples: General_Health, Age_Category, Smoking, Alcohol_Drinking, Diabetic, Physical_Activity, Sex, etc.

Numeric examples: BMI, Height_(cm), Weight_(kg), Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption, etc.

Note: If your CSV uses different names, update the notebook’s cleaning cell accordingly.

 Methods

Preprocessing

OneHotEncoder(handle_unknown='ignore') on categoricals

Pass-through on numerics via ColumnTransformer

Models

Baseline: Logistic Regression

Main: Random Forest (RandomizedSearchCV, 3-fold CV, 12 trials)

Optional: XGBoost (pipeline included in the notebook)

Train/Test Split

Stratified split with random_state=42, test size = 0.2

Metrics

Primary: ROC-AUC

Also: precision, recall, F1, confusion matrix, ROC curve


RESULTS

Dataset size: 308,854 rows × 19 columns after cleaning (no missing values in modeled columns).

Target imbalance: Heart_Disease = Yes is only 8.085% of the data → severely imbalanced.

Baseline model (first run): ROC-AUC = 0.8129. High overall accuracy but very low recall for the positive class at the default 0.5 threshold.

Tuned Random Forest (CV search):

Best CV AUC = 0.8242

Test AUC = 0.8313

With a lower decision threshold, positive-class (Yes) recall improved to 0.48 (precision 0.29), overall accuracy ≈ 0.86.

XGBoost (default threshold):

Test AUC ≈ 0.8401 (best AUC among the tested models)

But at threshold 0.5 it shows accuracy ≈ 0.92 with recall ≈ 0.05 for the positive class → great ranking power, poor sensitivity without threshold tuning.

Implication: Because of class imbalance, AUC is a better metric than accuracy, and you should tune the decision threshold (or use class weighting / resampling) to reach a clinically useful recall.

Details you can paste into your README
 Data overview

Rows: 308,854

Columns: 19

Example numeric stats:

BMI mean 28.63, median 27.44

Target distribution: No = 91.915%, Yes = 8.085%

🔬 Model results (this run)
Model/Setting	Metric	Score
Baseline (first run)	ROC-AUC	0.8129
Random Forest (best CV)	CV AUC	0.8242
Random Forest (test)	ROC-AUC	0.8313
Random Forest (lower threshold)	Recall (Yes)	0.48
Random Forest (lower threshold)	Precision (Yes)	0.29
Random Forest (lower threshold)	Accuracy	0.86
XGBoost (default 0.5 threshold)	ROC-AUC (test)	~0.8401
XGBoost (default 0.5 threshold)	Recall (Yes)	~0.05
XGBoost (default 0.5 threshold)	Accuracy	~0.92

Note: The RF “lower threshold” results reflect a trade-off you made in the notebook to boost sensitivity on the minority class; accuracy drops (as expected) while recall improves significantly.

 What this means? 

Accuracy is misleading on this dataset—most samples are “No,” so a model can look “good” while missing positives.

Your models have good ranking ability (AUC ~0.83–0.84).

To make the model useful for screening:

Tune the threshold to hit a target recall (e.g., ≥0.70) and report the paired precision.

Consider class_weight='balanced', focal loss (if supported), or SMOTE/undersampling.

Calibrate probabilities (e.g., Platt/Isotonic) before thresholding.

Visuals generated

Class distribution plot (shows strong imbalance)

ROC curves (per model)

Confusion matrices (default vs adjusted threshold)

Feature importance plots for tree models (see notebook cells)
