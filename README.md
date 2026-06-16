# 🌱 NayePankh Foundation — Donor Repeat Donation Predictor

## About the Project
A Machine Learning web app that predicts whether a donor will donate again to NayePankh Foundation based on their engagement profile.

## Live Demo
[Click here to view the live app](YOUR_STREAMLIT_LINK_HERE)

## ML Models Used
- Logistic Regression — 78.3% accuracy
- Decision Tree — 82.5% accuracy
- Random Forest — 91.7% accuracy (best)
- XGBoost — 89.2% accuracy

## Features Used for Prediction
- Previous donation history
- Email open rate
- Campaigns participated
- Volunteer history
- Social media following
- Last gift amount

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Plotly
- Pandas
- NumPy

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
nayepankh-ml-internship/

├── app.py                    # Streamlit web app

├── donor_model.pkl           # Trained Random Forest model

├── requirements.txt          # Dependencies

├── NayePankh_ML_Project.ipynb # Google Colab notebook

└── README.md                 # Project description

## Internship
Built as part of Machine Learning Internship at NayePankh Foundation via Internshala.