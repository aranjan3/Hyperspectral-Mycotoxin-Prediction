# Hyperspectral-Mycotoxin-Prediction
This project predicts mycotoxin (DON) concentration in corn samples using hyperspectral imaging data. The dataset consists of spectral reflectance data across multiple wavelengths. The goal is to preprocess this data, reduce its dimensionality, and train ML models (Random Forest, XGBoost, and CNN) to make accurate predictions.
**#Project_Overview**
Dataset: TASK-ML-INTERN.csv (Hyperspectral Imaging Data)
Dimensionality Reduction: PCA (448 → 3 features)
Models Used:
Random Forest
XGBoost
CNN
Deployment: Interactive Streamlit App (app.py)

Repository Structure
Hyperspectral-Mycotoxin-Prediction
Hyperspectral-Mycotoxin-Prediction
├── final_task.ipynb    # Jupyter Notebook (Preprocessing, PCA, ML models)
├── app.py              # Streamlit app for model inference
├── TASK-ML-INTERN.csv  # Hyperspectral dataset
├── report.pdf          # Final report summarizing results and analysis
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

 Dataset Details
The dataset contains hyperspectral imaging data for corn samples. It includes 448 spectral reflectance features captured across different wavelength bands. The target variable is mycotoxin concentration (DON level), which we predict using machine learning models.

**Setup Instructions**
1. Create a Virtual Environment
Before running the project, create an isolated Python environment:
python -m venv env
Activate the environment:
Windows:
env\Scripts\activate
Mac/Linux:
source env/bin/activate

2.Install Dependencies
Install all required packages using:
  pip install -r requirements.txt
 
3. Running the Jupyter Notebook
To explore data preprocessing, PCA, and model training, run the Jupyter Notebook:
jupyter notebook final_task.ipynb

5. Running the Streamlit App
After training the models, launch the Streamlit web app to interactively predict mycotoxin levels:
streamlit run app.py
This will open a browser window where you can upload new spectral data and get predictions.


Future Improvements
✔️ Hyperparameter tuning for CNN to reduce overfitting
✔️ Feature engineering to extract better spectral insights
✔️ Try t-SNE instead of PCA for dimensionality reduction
✔️ Experiment with ensemble models combining RF & XGBoost

 Credits
Dataset Source: Provided as part of the Machine Learning Internship Task
Developed By: Apoorv Ranjan
GitHub Repo: https://github.com/aranjan3/Hyperspectral-Mycotoxin-Prediction
🚀 Happy Coding! 🎯




