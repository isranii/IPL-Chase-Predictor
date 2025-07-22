# 🏏 IPL Chase Predictor

## Project Overview
This project is an **IPL Cricket Match Chase Predictor** built using machine learning. It's designed to predict the outcome of an IPL match (specifically, whether the chasing team will win) based on various match parameters and historical team performance data.

The application provides an interactive Streamlit interface where users can input match details and get real-time predictions and historical insights.

## Features
* **Predictive Model:** Utilizes a RandomForestClassifier to predict chase outcomes.
* **Interactive UI:** Built with Streamlit for easy input and result visualization.
* **Real-time Match Simulation:** Allows input of live match progress (powerplay/death over scores).
* **Historical Insights:** Displays overall team performance and head-to-head statistics.
* **Run Rate & Season Performance Charts:** Visualizes trends for estimated run rates and team win percentages across seasons.

## How it Works
The model is trained on historical IPL match data (`matches.csv` and `deliveries.csv`). It considers features such as:
* First innings score
* Match date (year, month, day, day of week)
* Historical win percentages and matches played for both teams
* Head-to-head records between the two competing teams
* Runs and wickets in Powerplay (0-6 overs) and Death Overs (15-20 overs) for both innings.

## Technologies Used
* Python
* Streamlit (for web application)
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning model)
* Joblib (for model serialization)
* Matplotlib & Plotly (for data visualization)

## Data Source
The historical IPL data used for training the model is sourced from Kaggle.
(You can add a link to the specific Kaggle dataset here if you wish, e.g., `[IPL Data (Kaggle)](https://www.kaggle.com/datasets/nowke9/ipldata)`)

## Getting Started (Run Locally)
To run this application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/isranii/IPL-Chase-Predictor.git](https://github.com/isranii/IPL-Chase-Predictor.git)
    cd IPL-Chase-Predictor
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On Mac/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure data and model files are present:**
    Make sure `ipl_chase_prediction_model_tuned.joblib` is in the root and `matches.csv`, `deliveries.csv` are in `archive/data/`.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Deployment
This application is deployed using Streamlit Community Cloud and can be accessed at:
(Once deployed, you can paste your live app URL here)

---
**Developed by:** Jahnavi Israni
