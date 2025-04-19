📚 Student Performance Prediction System
🌟 Project Overview
Problem: Predicting student performance in exams based on demographic and academic factors can help educators identify at-risk students and tailor interventions. However, such predictions require handling mixed data types (categorical and numerical) and selecting the best model for accuracy and interpretability.Solution: A machine learning pipeline combining preprocessing (TF-IDF vectorization and scaling) with multiple regression models (Random Forest, XGBoost, CatBoost, etc.), optimized via hyperparameter tuning with GridSearchCV.Key Achievement: Achieved an R2 score of 78% on the test set, with a user-friendly Flask web app for real-time predictions.
This project aims to assist educators and institutions by predicting a student’s math score based on factors like gender, race/ethnicity, parental education, lunch type, test preparation, and reading/writing scores, enabling data-driven educational support.
📊 Dataset
Source: Students Performance in Exams dataset from Kaggle

Link: Students Performance in Exams on Kaggle
Size: 1,000 student records (~150KB uncompressed)
Columns:
gender: Student’s gender (e.g., female, male).
race_ethnicity: Student’s race/ethnicity group (e.g., group B, group C).
parental_level_of_education: Parent’s education level (e.g., bachelor's degree, some college).
lunch: Type of lunch (e.g., standard, free/reduced).
test_preparation_course: Whether the student completed a test prep course (e.g., none, completed).
math_score, reading_score, writing_score: Exam scores (0–100).


Preprocessing:
Split into 80% training and 20% test sets.
Encoded categorical variables (e.g., gender, race_ethnicity) using OneHotEncoder.
Scaled numerical features (reading_score, writing_score) using StandardScaler.


Location: Stored in notebook/data/stud.csv.

🧠 Why This Architecture?



Component
Choice
Rationale



Preprocessing
ColumnTransformer + Pipeline
Handles mixed data types (categorical + numerical) efficiently


Models
Random Forest, XGBoost, CatBoost
Ensemble methods excel at capturing complex patterns in educational data


Benchmark
Linear Regression
Shows 15% R2 score improvement over baseline


Key Advantage: The pipeline ensures robust preprocessing and model selection, balancing accuracy and interpretability. Ensemble models like Random Forest and CatBoost were chosen for their ability to handle non-linear relationships in the data, which are common in educational performance prediction.
⚙️ Hyperparameter Tuning
Optimal Configuration

Random Forest: n_estimators=128
XGBoost: learning_rate=0.1, n_estimators=256
CatBoost: depth=8, learning_rate=0.05, iterations=100
GridSearchCV: Used 3-fold cross-validation to tune parameters like learning_rate, n_estimators, and depth.

Performance Impact



Metric
Before Tuning
After Tuning



R2 Score
0.72
0.78


Training Time
45s
52s


Model Size
38MB
42MB


Tuning improved the R2 score by 6%, making predictions more accurate, though it slightly increased training time due to GridSearchCV.
🚀 Getting Started

Installation

Create Conda environment:
conda create -p "E:\Coding\ML and DL Projects\Student Performance Project\venv" python=3.8 -y


Activate environment:
conda activate "E:\Coding\ML and DL Projects\Student Performance Project\venv"


Install dependencies:
pip install -r requirements.txt




How to Run

Training: Train the model and generate artifacts:
python src/pipeline/train_pipeline.py


Web App: Start the Flask app to access the prediction interface:
python app.py


Access the homepage at http://localhost:5000/ and the prediction page at http://localhost:5000/predictdata.





📈 Results



Model
R2 Score (Test)
Training Time



Best Model (CatBoost)
0.78
52s


Linear Regression Baseline
0.63
10s


The best model (CatBoost) achieved a 15% improvement in R2 score over the baseline, providing reliable predictions for student math scores.
💡 Future Roadmap



Feature
Status
Target Impact



Feature Engineering
Planned
+5% R2 score


User Feedback
Research
Improve UX


API Deployment
Backlog
Enable mobile access


