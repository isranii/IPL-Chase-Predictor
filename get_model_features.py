#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Jahnavi Israni
#
# Created:     21-07-2025
# Copyright:   (c) Jahnavi Israni 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import joblib
import pandas as pd
import os
model_file_name = 'ipl_chase_prediction_model_tuned.joblib'
model_path = os.path.join(os.path.dirname(__file__), model_file_name)
try:
    # Load your trained model
    model = joblib.load(model_path)

    if hasattr(model, 'feature_names_in_'):
        correct_feature_names = model.feature_names_in_.tolist()
        print("\n--- START COPYING FROM HERE ---")
        print(correct_feature_names)
        print("--- STOP COPYING HERE ---")
        print("\nInstructions:")
        print("1. Copy the entire list (including square brackets []) printed between 'START' and 'STOP' lines.")
        print("2. Open your 'app.py' file.")
        print("3. Find the line that starts with 'expected_model_columns = [...]'.")
        print("4. Replace the entire list on that line with the list you just copied.")
        print("5. Save 'app.py' and run it again: 'streamlit run app.py'")
    else:
        print("Error: Your loaded model does not have a 'feature_names_in_' attribute.")
        print("This means the model might be an older version or a different type.")
        print("You will need to go back to your original model training script.")
        print("In that script, after you trained your model (e.g., `model.fit(X_train, y_train)`),")
        print("add a line to print the column names of your training data (e.g., `print(X_train.columns.tolist())`).")
        print("Run your training script, copy that output, and use it to replace `expected_model_columns` in `app.py`.")

except FileNotFoundError:
    print(f"Error: Model file '{model_file_name}' not found at '{model_path}'")
    print("Please double-check the 'model_path' in this script and ensure the model file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print("Please ensure 'joblib' and 'pandas' are installed (`pip install joblib pandas`).")