🚀 Third Place Winner – CEB SOLE Biohackathon Track

📌 Overview
This project was developed as part of the SOLE (Science Operation Leaders in Egypt) competition, organized by CEB, in the Biohackathon track. The objective was to analyze genomic data and predict clades using advanced machine-learning techniques.

The approach involved data preprocessing, feature engineering, and model training. Multiple and selwere then evaluated, andst-performing one. The f was selectedinal model was deployed for inference on unseen data.

📂 Project Structure

📁 Biohackathon-Project/
│── 📄 train_model.py       # Training and saving the best model  
│── 📄 validate_model.py    # Evaluating the model performance  
│── 📄 test_model.py        # Making predictions on new data  
│── 📄 ensemble_submission.py  # Combining predictions from multiple models  
│── 📄 requirements.txt     # Required dependencies  
│── 📄 README.md            # Detailed documentation  
│── 📁 data/                # Dataset files (train, validation, test)  
│── 📁 models/              # Saved machine learning models  
│── 📁 submissions/         # Final prediction files  
🛠️ Steps Implemented in the Project
1️⃣ Data Preprocessing & Feature Engineering
Loaded the dataset containing genomic sequences and corresponding clades.
Checked for missing values and performed necessary data cleaning.
Encoded categorical variables (e.g., Clade, other genomic features) using LabelEncoder.
Removed unnecessary columns (e.g., sample IDs that do not contribute to classification).

📌 Example Code:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/train.csv")

# Encode categorical features
encoder = LabelEncoder()
df['Clade'] = encoder.fit_transform(df['Clade'])

# Save encoder for later use
import pickle
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
2️⃣ Model Training
Used LightGBM due to its high efficiency with large datasets.
Trained the model using optimized hyperparameters.
Saved the trained model for later use.
📌 Example Code:


import lightgbm as lgb
import pickle

# Prepare training data
X_train = df.drop(columns=['Clade'])
y_train = df['Clade']

# Initialize and train the model
model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train)

# Save the trained model
with open("models/lightgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)
3️⃣ Model Validation & Performance Evaluation
Loaded the trained model and applied it to the validation dataset.
Computed performance metrics:
F1-score: Measures precision & recall balance.
AUC-ROC: Measures classification performance across all classes.


📌 Example Code:

from sklearn.metrics import f1_score, roc_auc_score

# Load model
with open("models/lightgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load validation data
df_val = pd.read_csv("data/val.csv")
X_val = df_val.drop(columns=['Clade'])
y_val = df_val['Clade']

# Make predictions
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)

# Evaluate performance
f1 = f1_score(y_val, y_pred, average="weighted")
auc = roc_auc_score(y_val, y_pred_proba, multi_class="ovr")

print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
4️⃣ Making Predictions on New Data
Loaded the trained model and applied it to unseen test data.
Ensured all preprocessing steps were consistent with training.
Saved predictions for submission.

📌 Example Code:

# Load test data
df_test = pd.read_csv("data/test.csv")
X_test = df_test.drop(columns=['ID'])

# Predict clades
y_test_pred = model.predict(X_test)

# Save predictions
df_predictions = pd.DataFrame({
    "ID": df_test["ID"],
    "Predicted_Clade": y_test_pred
})
df_predictions.to_csv("submissions/test_predictions.csv", index=False)
5️⃣ Ensemble Learning (Combining Multiple Models' Predictions)
Since different models capture different aspects of the data, an ensemble approach was applied:

Trained multiple models: LightGBM, XGBoost, and Random Forest.
Averaged predictions from the different models to improve accuracy.
Generated final submission file using majority voting.

📌 Example Code:

import pandas as pd
import numpy as np

# Load individual model predictions
df_lgb = pd.read_csv("submissions/lgb_predictions.csv")
df_xgb = pd.read_csv("submissions/xgb_predictions.csv")
df_rf = pd.read_csv("submissions/rf_predictions.csv")

# Majority voting
final_pred = np.round((df_lgb['Predicted_Clade'] + df_xgb['Predicted_Clade'] + df_rf['Predicted_Clade']) / 3).astype(int)

# Create a submission file
df_submission = pd.DataFrame({
    "ID": df_lgb["ID"],
    "Predicted_Clade": final_pred
})
df_submission.to_csv("submissions/final_submission.csv", index=False)
📊 Final Model Performance
Metric	Score
F1 Score	0.92
AUC-ROC	0.96



📌 How to Use This Repository
🔹 Cloning the Repository

git clone https://github.com/IbraheemMustafa7/MetawithAI.git
cd MetawithAI

or have a look at 
https://colab.research.google.com/drive/1Arp9K0BoCmCSeYuw4p9nUojqdpMYK4bH

🔹 Installing Dependencies
Ensure you have Python installed, then run:


pip install -r requirements.txt
🔹 Training the Model

python train_model.py
🔹 Validating the Model

python validate_model.py
🔹 Running Predictions on Test Data

python test_model.py
🛠️ Technologies Used
Python 🐍
Pandas & NumPy 📊
LightGBM 🌲
Scikit-learn 🧠
Ensemble Learning 🎯
💡 Key Takeaways
✅ Successfully trained a high-performing model on genomic data.
✅ Applied advanced feature engineering techniques.
✅ Achieved excellent performance on F1-score and AUC-ROC metrics.
✅ Leveraged ensemble learning to enhance predictions.
✅ Secured Third Place in the SOLE Biohackathon! 🎉

📢 Acknowledgments
Thanks to CEB for organizing this competition and providing an incredible learning experience! 🙌

And huge Thanks to all the judges for mentoring, and guiding us through an incredible learning experience! 🙌
