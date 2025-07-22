import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Set up your Python Environment and Load Data ---
print("--- Starting Step 1: Loading Data ---")
script_dir = os.path.dirname(__file__)
matches_file_path = os.path.join(script_dir, 'archive', 'data', 'matches.csv')
deliveries_file_path = os.path.join(script_dir, 'archive', 'data', 'deliveries.csv')

try:
    matches_data = pd.read_csv(matches_file_path)
    deliveries_data = pd.read_csv(deliveries_file_path)
except FileNotFoundError:
    print("FATAL ERROR: Files not found! Check folder structure.")
    exit()

matches = matches_data.copy()
deliveries = deliveries_data.copy()
print("Step 1: Data Loading Complete. Matches Shape:", matches.shape, "Deliveries Shape:", deliveries.shape)

# --- Step 2: Initial Data Preprocessing ---
print("\n--- Starting Step 2: Initial Data Preprocessing ---")
matches.dropna(subset=['winner'], inplace=True)
matches['player_of_match'] = matches['player_of_match'].fillna('Unknown') # Corrected for FutureWarning

columns_to_drop_matches = ['city', 'umpire3']
matches.drop(columns=columns_to_drop_matches, inplace=True)
print("Step 2: Initial Data Preprocessing Complete for matches. Shape:", matches.shape)

# --- Step 3: Merging Dataframes and Initial Feature Creation ---
print("\n--- Starting Step 3: Merging Dataframes and Initial Feature Creation ---")
merged_data = pd.merge(deliveries, matches, left_on='match_id', right_on='id')

first_innings_scores = merged_data[merged_data['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
first_innings_scores.columns = ['match_id', 'first_innings_score']

matches_with_second_inning = merged_data[merged_data['inning'] == 2]['match_id'].unique()
matches_for_chase_prediction = matches[matches['id'].isin(matches_with_second_inning)].copy()

matches_with_first_innings_score = pd.merge(matches_for_chase_prediction, first_innings_scores,
                                            left_on='id', right_on='match_id', how='left')
matches_with_first_innings_score.drop('match_id', axis=1, inplace=True)
print("Step 3: Merging and First Innings Score Complete. Shape:", matches_with_first_innings_score.shape)

# --- Step 4: Advanced Feature Engineering (Date Features & Team Standardization) ---
print("\n--- Starting Step 4: Advanced Feature Engineering (Date & Team Names) ---")
matches_with_first_innings_score['date'] = pd.to_datetime(matches_with_first_innings_score['date'])
matches_with_first_innings_score['match_year'] = matches_with_first_innings_score['date'].dt.year
matches_with_first_innings_score['match_month'] = matches_with_first_innings_score['date'].dt.month
matches_with_first_innings_score['match_day'] = matches_with_first_innings_score['date'].dt.day
matches_with_first_innings_score['match_dayofweek'] = matches_with_first_innings_score['date'].dt.dayofweek

# The team_name_mapping from your model script (ENSURE THIS IS COMPLETE AND ACCURATE)
team_name_mapping = {
    'Mumbai Indians': 'Mumbai Indians', 'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Lions', 'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad', 'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings', 'Chennai Super Kings': 'Chennai Super Kings',
    'Rajasthan Royals': 'Rajasthan Royals', 'Deccan Chargers': 'Deccan Chargers', # Keeping separate based on latest error
    'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala', 'Pune Warriors': 'Pune Warriors',
    'Rising Pune Supergiants': 'Rising Pune Supergiants', 'Delhi Capitals': 'Delhi Capitals',
    'Punjab Kings': 'Punjab Kings'
    # Add any new teams from matches.csv here if they are present in the raw data
    # and you want their OHE features in the model.
    # e.g., 'Gujarat Titans': 'Gujarat Titans', 'Lucknow Super Giants': 'Lucknow Super Giants'
}

matches_with_first_innings_score['team1'] = matches_with_first_innings_score['team1'].replace(team_name_mapping)
matches_with_first_innings_score['team2'] = matches_with_first_innings_score['team2'].replace(team_name_mapping)
matches_with_first_innings_score['winner'] = matches_with_first_innings_score['winner'].replace(team_name_mapping)
matches_with_first_innings_score['toss_winner'] = matches_with_first_innings_score['toss_winner'].replace(team_name_mapping) # IMPORTANT: Ensure toss_winner mapping is done consistently
deliveries['batting_team'] = deliveries['batting_team'].replace(team_name_mapping)
deliveries['bowling_team'] = deliveries['bowling_team'].replace(team_name_mapping)
print("Step 4: Date features and team names standardized. Shape:", matches_with_first_innings_score.shape)

# --- Step 5: Advanced Feature Engineering (Historical Team-Level Statistics) ---
print("\n--- Starting Step 5: Advanced FE (Historical Team Stats) ---")
cumulative_team_stats = {}
match_features_with_history = []
matches_sorted = matches_with_first_innings_score.sort_values(by='date').copy() # Use the already processed DF

for index, match in matches_sorted.iterrows():
    team1 = match['team1']
    team2 = match['team2']
    winner = match['winner']
    match_id = match['id']

    if team1 not in cumulative_team_stats:
        cumulative_team_stats[team1] = {'played': 0, 'won': 0}
    if team2 not in cumulative_team_stats:
        cumulative_team_stats[team2] = {'played': 0, 'won': 0}

    team1_played_before = cumulative_team_stats[team1]['played']
    team1_won_before = cumulative_team_stats[team1]['won']
    team2_played_before = cumulative_team_stats[team2]['played']
    team2_won_before = cumulative_team_stats[team2]['won']

    team1_win_pct_before = (team1_won_before / team1_played_before) * 100 if team1_played_before > 0 else 0
    team2_win_pct_before = (team2_won_before / team2_played_before) * 100 if team2_played_before > 0 else 0

    match_features_with_history.append({
        'id': match_id,
        'team1_matches_played_hist': team1_played_before,
        'team1_win_pct_hist': team1_win_pct_before,
        'team2_matches_played_hist': team2_played_before,
        'team2_win_pct_hist': team2_win_pct_before,
    })

    # Update stats after the match
    cumulative_team_stats[team1]['played'] += 1
    cumulative_team_stats[team2]['played'] += 1
    if winner == team1:
        cumulative_team_stats[team1]['won'] += 1
    elif winner == team2:
        cumulative_team_stats[team2]['won'] += 1

historical_features_df = pd.DataFrame(match_features_with_history)
matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, historical_features_df, on='id', how='left')
print("Step 5: Historical team features added to main DF. New shape:", matches_with_first_innings_score.shape)

# --- Step 6: Advanced Feature Engineering (Match Segment Statistics - Powerplay/Death Overs) ---
print("\n--- Starting Step 6: Advanced FE (Match Segment Stats) ---")
powerplay_data = merged_data[merged_data['over'] <= 6].copy()
powerplay_stats = powerplay_data.groupby(['match_id', 'inning']).agg(
    powerplay_runs=('total_runs', 'sum'),
    powerplay_wickets=('player_dismissed', lambda x: x.notna().sum())
).reset_index()

powerplay_inning1 = powerplay_stats[powerplay_stats['inning'] == 1].rename(columns={
    'powerplay_runs': 'inning1_powerplay_runs', 'powerplay_wickets': 'inning1_powerplay_wickets'}).drop(columns='inning')
powerplay_inning2 = powerplay_stats[powerplay_stats['inning'] == 2].rename(columns={
    'powerplay_runs': 'inning2_powerplay_runs', 'powerplay_wickets': 'inning2_powerplay_wickets'}).drop(columns='inning')

matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, powerplay_inning1,
                                            left_on='id', right_on='match_id', how='left')
matches_with_first_innings_score.drop(columns='match_id', inplace=True)
matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, powerplay_inning2,
                                            left_on='id', right_on='match_id', how='left')
matches_with_first_innings_score.drop(columns='match_id', inplace=True)

matches_with_first_innings_score['inning1_powerplay_runs'] = matches_with_first_innings_score['inning1_powerplay_runs'].fillna(0)
matches_with_first_innings_score['inning1_powerplay_wickets'] = matches_with_first_innings_score['inning1_powerplay_wickets'].fillna(0)
matches_with_first_innings_score['inning2_powerplay_runs'] = matches_with_first_innings_score['inning2_powerplay_runs'].fillna(0)
matches_with_first_innings_score['inning2_powerplay_wickets'] = matches_with_first_innings_score['inning2_powerplay_wickets'].fillna(0)

death_overs_data = merged_data[merged_data['over'] >= 15].copy()
death_overs_stats = death_overs_data.groupby(['match_id', 'inning']).agg(
    death_overs_runs=('total_runs', 'sum'),
    death_overs_wickets=('player_dismissed', lambda x: x.notna().sum())
).reset_index()

death_overs_inning1 = death_overs_stats[death_overs_stats['inning'] == 1].rename(columns={
    'death_overs_runs': 'inning1_death_overs_runs', 'death_overs_wickets': 'inning1_death_overs_wickets'}).drop(columns='inning')
death_overs_inning2 = death_overs_stats[death_overs_stats['inning'] == 2].rename(columns={
    'death_overs_runs': 'inning2_death_overs_runs', 'death_overs_wickets': 'inning2_death_overs_wickets'}).drop(columns='inning')

matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, death_overs_inning1,
                                            left_on='id', right_on='match_id', how='left')
matches_with_first_innings_score.drop(columns='match_id', inplace=True)
matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, death_overs_inning2,
                                            left_on='id', right_on='match_id', how='left')
matches_with_first_innings_score.drop(columns='match_id', inplace=True)

matches_with_first_innings_score['inning1_death_overs_runs'] = matches_with_first_innings_score['inning1_death_overs_runs'].fillna(0)
matches_with_first_innings_score['inning1_death_overs_wickets'] = matches_with_first_innings_score['inning1_death_overs_wickets'].fillna(0)
matches_with_first_innings_score['inning2_death_overs_runs'] = matches_with_first_innings_score['inning2_death_overs_runs'].fillna(0)
matches_with_first_innings_score['inning2_death_overs_wickets'] = matches_with_first_innings_score['inning2_death_overs_wickets'].fillna(0)
print("Step 6: Match Segment Stats Complete. Shape:", matches_with_first_innings_score.shape)

# --- Step 7: Advanced Feature Engineering (Head-to-Head Records) ---
print("\n--- Starting Step 7: Advanced FE (Head-to-Head Records) ---")
cumulative_head_to_head_stats = {}
head_to_head_features = []
matches_sorted_for_h2h = matches_with_first_innings_score.sort_values(by='date').copy()

for index, match in matches_sorted_for_h2h.iterrows():
    team1 = match['team1']
    team2 = match['team2']
    winner = match['winner']
    match_id = match['id']

    h2h_key = tuple(sorted((team1, team2)))

    if h2h_key not in cumulative_head_to_head_stats:
        cumulative_head_to_head_stats[h2h_key] = {'played': 0, 'team1_wins_h2h': 0, 'team2_wins_h2h': 0}

    h2h_played_before = cumulative_head_to_head_stats[h2h_key]['played']
    key_team1_in_h2h = h2h_key[0]
    key_team2_in_h2h = h2h_key[1]

    team1_h2h_wins_hist = 0
    team2_h2h_wins_hist = 0

    if team1 == key_team1_in_h2h:
        team1_h2h_wins_hist = cumulative_head_to_head_stats[h2h_key]['team1_wins_h2h']
        team2_h2h_wins_hist = cumulative_head_to_head_stats[h2h_key]['team2_wins_h2h']
    else: # team1 must be key_team2_in_h2h
        team1_h2h_wins_hist = cumulative_head_to_head_stats[h2h_key]['team2_wins_h2h']
        team2_h2h_wins_hist = cumulative_head_to_head_stats[h2h_key]['team1_wins_h2h']

    head_to_head_features.append({
        'id': match_id,
        'h2h_matches_played_hist': h2h_played_before,
        'team1_h2h_wins_hist': team1_h2h_wins_hist,
        'team2_h2h_wins_hist': team2_h2h_wins_hist
    })

    # Update stats after the match
    cumulative_head_to_head_stats[h2h_key]['played'] += 1
    if winner == team1:
        if team1 == key_team1_in_h2h:
            cumulative_head_to_head_stats[h2h_key]['team1_wins_h2h'] += 1
        else:
            cumulative_head_to_head_stats[h2h_key]['team2_wins_h2h'] += 1
    elif winner == team2:
        if team2 == key_team1_in_h2h: # If team2 is the first in the sorted key
            cumulative_head_to_head_stats[h2h_key]['team1_wins_h2h'] += 1
        else: # If team2 is the second in the sorted key
            cumulative_head_to_head_stats[h2h_key]['team2_wins_h2h'] += 1

head_to_head_df = pd.DataFrame(head_to_head_features)
matches_with_first_innings_score = pd.merge(matches_with_first_innings_score, head_to_head_df, on='id', how='left')

matches_with_first_innings_score['h2h_matches_played_hist'] = matches_with_first_innings_score['h2h_matches_played_hist'].fillna(0)
matches_with_first_innings_score['team1_h2h_wins_hist'] = matches_with_first_innings_score['team1_h2h_wins_hist'].fillna(0)
matches_with_first_innings_score['team2_h2h_wins_hist'] = matches_with_first_innings_score['team2_h2h_wins_hist'].fillna(0)
print("Step 7: Head-to-Head records added to main DF. New shape:", matches_with_first_innings_score.shape)

# --- Step 8: Define Target Variable and Prepare Categorical Features ---
print("\n--- Starting Step 8: Define Target & One-Hot Encoding ---")
matches_with_first_innings_score['chase_successful'] = 0
matches_with_first_innings_score.loc[matches_with_first_innings_score['team2'] == matches_with_first_innings_score['winner'], 'chase_successful'] = 1

features_for_model_final = [
    'id',
    'first_innings_score',
    'match_year', 'match_month', 'match_day', 'match_dayofweek',
    'team1_matches_played_hist', 'team1_win_pct_hist',
    'team2_matches_played_hist', 'team2_win_pct_hist',
    'inning1_powerplay_runs', 'inning1_powerplay_wickets',
    'inning2_powerplay_runs', 'inning2_powerplay_wickets',
    'inning1_death_overs_runs', 'inning1_death_overs_wickets',
    'inning2_death_overs_runs', 'inning2_death_overs_wickets',
    'h2h_matches_played_hist',
    'team1_h2h_wins_hist',
    'team2_h2h_wins_hist',
    'team1', 'team2', 'toss_winner', 'toss_decision', 'venue',
]

data_for_model_final = matches_with_first_innings_score[features_for_model_final + ['chase_successful']].copy()

categorical_cols_final = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']

# --- Get all unique categories for consistent One-Hot Encoding ---
# Use the original matches_data for getting all unique values
all_teams_in_raw_data = sorted(list(matches_data['team1'].unique()) + \
                               list(matches_data['team2'].unique()) + \
                               list(matches_data['toss_winner'].unique()))
# Ensure that all mapped team names are considered for OHE features
all_possible_team_categories = sorted(list(set(team_name_mapping.get(t, t) for t in all_teams_in_raw_data)))

all_venues_in_raw_data = sorted(matches_data['venue'].unique().tolist())
all_toss_decisions = ['bat', 'field'] # Explicitly define

# --- CRITICAL NEW PART: Create a reference DataFrame with ALL possible OHE columns ---
# This ensures consistent column creation and order.
reference_df_for_ohe = pd.DataFrame(index=[0])
for col_name in categorical_cols_final:
    if col_name == 'team1' or col_name == 'team2' or col_name == 'toss_winner':
        for team in all_possible_team_categories:
            reference_df_for_ohe[f'{col_name}_{team}'] = 0
    elif col_name == 'toss_decision':
        for decision in all_toss_decisions:
            if decision != 'bat': # Assuming 'bat' is the dropped one if drop_first=True
                reference_df_for_ohe[f'toss_decision_{decision}'] = 0
    elif col_name == 'venue':
        for venue in all_venues_in_raw_data:
            reference_df_for_ohe[f'venue_{venue}'] = 0

# Convert dummy variables to int type
reference_df_for_ohe = reference_df_for_ohe.astype(int)

# Extract only the features for encoding from data_for_model_final
X_pre_encoded_for_dummies = data_for_model_final[categorical_cols_final].copy()

# Apply get_dummies to the actual data.
# This will produce columns for categories present in this data, respecting drop_first.
X_encoded_actual_data = pd.get_dummies(X_pre_encoded_for_dummies, columns=categorical_cols_final, drop_first=True, dtype=int)


# Now, combine numerical features with the encoded categorical features.
numerical_cols = [col for col in data_for_model_final.drop(columns=['id', 'chase_successful'] + categorical_cols_final).columns]
X_numerical_part = data_for_model_final[numerical_cols].reset_index(drop=True)

# Concatenate numerical and one-hot encoded parts
X_combined_features = pd.concat([X_numerical_part, X_encoded_actual_data], axis=1)

# --- FINAL COLUMN REORDERING AND FILLING FOR CONSISTENCY ---
# Create the full list of expected columns by taking existing numerical + all expected OHE
# The order from reference_df_for_ohe is our blueprint for OHE columns
final_expected_ohe_cols = sorted(reference_df_for_ohe.columns.tolist()) # Sort for consistent order
all_final_columns = numerical_cols + final_expected_ohe_cols

# Reindex the combined features to match the exact set AND order of columns needed for the model.
# This is the most robust step to ensure consistency.
X_final_encoded = X_combined_features.reindex(columns=all_final_columns, fill_value=0)

# Ensure the dtypes are correct after reindex (especially for OHE columns)
for col in X_final_encoded.columns:
    if col.startswith(('team', 'toss_decision', 'venue')) and '_' in col:
        X_final_encoded[col] = X_final_encoded[col].astype(int)
    # Ensure numericals are correct if they somehow changed
    elif col in numerical_cols:
         if 'win_pct_hist' in col:
             X_final_encoded[col] = X_final_encoded[col].astype(float)
         else:
             X_final_encoded[col] = X_final_encoded[col].astype(int)

print(f"Step 8: Final features and target defined and One-Hot Encoded. Shape of X_final_encoded: {X_final_encoded.shape}")


y_final = data_for_model_final['chase_successful']


# --- Step 9: Model Training and Evaluation (Base Model) ---
print("\n--- Starting Step 9: Model Training and Evaluation (Base Model) ---")
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final_encoded, y_final, test_size=0.2, random_state=42)
model_base = RandomForestClassifier(n_estimators=100, random_state=42)
model_base.fit(X_train_final, y_train_final)
y_pred_base = model_base.predict(X_test_final)
accuracy_base = accuracy_score(y_test_final, y_pred_base)
conf_matrix_base = confusion_matrix(y_test_final, y_pred_base)
class_report_base = classification_report(y_test_final, y_pred_base)
print(f"Step 9: Base Model Accuracy with Enhanced Features: {accuracy_base:.4f}")
print("\nBase Model Confusion Matrix:\n", conf_matrix_base)
print("\nBase Model Classification Report:\n", class_report_base)
print("\n--- Step 9: Base Model Training and Evaluation Complete ---")

# --- Step 10: Hyperparameter Tuning for RandomForestClassifier ---
print("\n--- Starting Step 10: Hyperparameter Tuning ---")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)
print("Performing GridSearchCV (this may take some time)...")
grid_search.fit(X_train_final, y_train_final)
print("\nGridSearchCV complete.")
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Hyperparameters: {best_params}")
print(f"Best Cross-validation Accuracy: {best_score:.4f}")

# --- Step 11: Train Final Tuned Model and Evaluate ---
print("\n--- Starting Step 11: Train Final Tuned Model and Evaluate ---")
model_final_tuned = RandomForestClassifier(**best_params, random_state=42)
model_final_tuned.fit(X_train_final, y_train_final)
y_pred_tuned = model_final_tuned.predict(X_test_final)
accuracy_tuned = accuracy_score(y_test_final, y_pred_tuned)
conf_matrix_tuned = confusion_matrix(y_test_final, y_pred_tuned)
class_report_tuned = classification_report(y_test_final, y_pred_tuned)
print(f"Step 11: Tuned Model Accuracy on Test Set: {accuracy_tuned:.4f}")
print("\nTuned Model Confusion Matrix:\n", conf_matrix_tuned)
print("\nTuned Model Classification Report:\n", class_report_tuned)
print("\n--- Step 11: Tuned Model Training and Evaluation Complete ---")

# --- Step 12: Save Final Tuned Model and Define Robust Prediction Function ---
print("\n--- Starting Step 12: Save Final Tuned Model and Prediction Function ---")
model_filename_tuned = 'ipl_chase_prediction_model_tuned.joblib'
joblib.dump(model_final_tuned, model_filename_tuned)
print(f"Step 12: Tuned model saved successfully as '{model_filename_tuned}'")

# This is the EXPLICIT LIST of column names and their EXACT ORDER
# that the trained model expects. This is what MUST be copied to app.py.
final_model_columns_tuned = X_final_encoded.columns.tolist()
print("\n\n--- REQUIRED FOR APP.PY: FINAL MODEL COLUMNS START ---")
print(final_model_columns_tuned)
print("--- REQUIRED FOR APP.PY: FINAL MODEL COLUMNS END ---\n\n")

# Re-defining these for app.py helper functions
all_teams_raw_for_app_helper = sorted(list(set(matches_data['team1'].unique().tolist() + matches_data['team2'].unique().tolist())))
all_teams_standardized_for_app_helper = sorted(list(set([team_name_mapping.get(team, team) for team in all_teams_raw_for_app_helper])))
all_venues_for_app_helper = sorted(matches_data['venue'].unique().tolist())

# Recalculate global historical stats using the original matches_data and team_name_mapping
overall_team_win_pct_dict_global = matches_data.replace(team_name_mapping).groupby('winner').size() / len(matches_data) * 100
overall_team_played_dict_global = matches_data.replace(team_name_mapping).groupby('team1').size().add(
                                  matches_data.replace(team_name_mapping).groupby('team2').size(), fill_value=0)

def get_historical_stats_for_app_global(team_name):
    played = overall_team_played_dict_global.get(team_name, 0)
    won_pct = overall_team_win_pct_dict_global.get(team_name, 0)
    return played, won_pct

# The robust prediction function (will use the final tuned model)
# This function's input `final_cols_list_global` is paramount for ordering.
def predict_chase_outcome_robust(new_match_data_raw_dict, team_name_map_global, historical_stats_func_global, all_venues_list_global, final_cols_list_global, model_obj_global):
    input_data_processed = {
        'first_innings_score': new_match_data_raw_dict['first_innings_score'],
        'match_year': new_match_data_raw_dict['match_year'],
        'match_month': new_match_data_raw_dict['match_month'],
        'match_day': new_match_data_raw_dict['match_day'],
        'match_dayofweek': new_match_data_raw_dict['match_dayofweek'],
        'inning1_powerplay_runs': new_match_data_raw_dict['inning1_powerplay_runs'],
        'inning1_powerplay_wickets': new_match_data_raw_dict['inning1_powerplay_wickets'],
        'inning2_powerplay_runs': new_match_data_raw_dict['inning2_powerplay_runs'],
        'inning2_powerplay_wickets': new_match_data_raw_dict['inning2_powerplay_wickets'],
        'inning1_death_overs_runs': new_match_data_raw_dict['inning1_death_overs_runs'],
        'inning1_death_overs_wickets': new_match_data_raw_dict['inning1_death_overs_wickets'],
        'inning2_death_overs_runs': new_match_data_raw_dict['inning2_death_overs_runs'],
        'inning2_death_overs_wickets': new_match_data_raw_dict['inning2_death_overs_wickets'],
        'h2h_matches_played_hist': new_match_data_raw_dict['h2h_matches_played_hist'],
        'team1_h2h_wins_hist': new_match_data_raw_dict['team1_h2h_wins_hist'],
        'team2_h2h_wins_hist': new_match_data_raw_dict['team2_h2h_wins_hist'],
    }
    team1_standardized = team_name_map_global.get(new_match_data_raw_dict['team1'], new_match_data_raw_dict['team1'])
    team2_standardized = team_name_map_global.get(new_match_data_raw_dict['team2'], new_match_data_raw_dict['team2'])
    toss_winner_standardized = team_name_map_global.get(new_match_data_raw_dict['toss_winner'], new_match_data_raw_dict['toss_winner'])
    toss_decision = new_match_data_raw_dict['toss_decision']
    venue = new_match_data_raw_dict['venue']

    team1_played, team1_win_pct = historical_stats_func_global(team1_standardized)
    team2_played, team2_win_pct = historical_stats_func_global(team2_standardized)

    input_data_processed['team1_matches_played_hist'] = team1_played
    input_data_processed['team1_win_pct_hist'] = team1_win_pct
    input_data_processed['team2_matches_played_hist'] = team2_played
    input_data_processed['team2_win_pct_hist'] = team2_win_pct

    # Create an empty DataFrame with the EXACT columns and order expected by the model
    # This is critical for matching the `fit` data
    final_input_df = pd.DataFrame(0, index=[0], columns=final_cols_list_global)

    # Populate numerical features (ensuring type compatibility where floats are assigned to int columns)
    for col, value in input_data_processed.items():
        if col in final_input_df.columns:
            if final_input_df[col].dtype == 'int64' and isinstance(value, float):
                final_input_df.loc[0, col] = int(value)
            else:
                final_input_df.loc[0, col] = value

    # Apply one-hot encoding by setting the appropriate column to 1
    ohe_cols_to_set = []
    if f'team1_{team1_standardized}' in final_cols_list_global:
        ohe_cols_to_set.append(f'team1_{team1_standardized}')
    if f'team2_{team2_standardized}' in final_cols_list_global:
        ohe_cols_to_set.append(f'team2_{team2_standardized}')
    if f'toss_winner_{toss_winner_standardized}' in final_cols_list_global:
        ohe_cols_to_set.append(f'toss_winner_{toss_winner_standardized}')
    # Note: 'toss_decision_bat' is handled by 'toss_decision_field' being 0
    if toss_decision == 'field' and 'toss_decision_field' in final_cols_list_global:
        ohe_cols_to_set.append('toss_decision_field')
    if f'venue_{venue}' in final_cols_list_global:
        ohe_cols_to_set.append(f'venue_{venue}')

    for ohe_col in ohe_cols_to_set:
        final_input_df.loc[0, ohe_col] = 1 # Set the relevant OHE column to 1

    # Final check on dtypes (especially for OHE columns, which should be int)
    for col in final_input_df.columns:
        if col.startswith(('team', 'toss_decision', 'venue')) and '_' in col:
            final_input_df[col] = final_input_df[col].astype(int)
        elif 'win_pct_hist' in col: # These should be float
            final_input_df[col] = final_input_df[col].astype(float)
        else: # Other numerical columns should be int
            final_input_df[col] = final_input_df[col].astype(int)

    # Ensure the order is exactly as expected by the model by re-indexing to the final_cols_list_global
    # This is the ultimate safeguard for order mismatch.
    final_input_df = final_input_df[final_cols_list_global]

    prediction = model_obj_global.predict(final_input_df)
    prediction_proba = model_obj_global.predict_proba(final_input_df)
    return prediction, prediction_proba

print("Step 12: Tuned Prediction Function Defined.")

# --- Test the Enhanced Prediction Function (Optional, for script execution) ---
print("\n--- Testing the Enhanced Prediction Function ---")
sample_raw_input_data_dict = {
    'team1': 'Chennai Super Kings',
    'team2': 'Mumbai Indians',
    'toss_winner': 'Chennai Super Kings', # Using CSK here to ensure it's handled
    'toss_decision': 'field',
    'venue': 'Wankhede Stadium',
    'first_innings_score': 160,
    'match_year': 2025,
    'match_month': 4,
    'match_day': 10,
    'match_dayofweek': 4,
    'inning1_powerplay_runs': 55, 'inning1_powerplay_wickets': 1,
    'inning2_powerplay_runs': 40, 'inning2_powerplay_wickets': 2,
    'inning1_death_overs_runs': 50, 'inning1_death_overs_wickets': 2,
    'inning2_death_overs_runs': 20, 'inning2_death_overs_wickets': 0,
    'h2h_matches_played_hist': 10, 'team1_h2h_wins_hist': 6, 'team2_h2h_wins_hist': 4
}
try:
    sample_prediction_final, sample_proba_final = predict_chase_outcome_robust(
        sample_raw_input_data_dict,
        team_name_mapping,
        get_historical_stats_for_app_global,
        all_venues_for_app_helper,
        final_model_columns_tuned, # Pass the exactly generated columns for prediction test
        model_final_tuned
    )
    print(f"Sample raw input data prediction: {sample_prediction_final[0]}")
    print(f"Sample raw input data probabilities (Lose/Win): {sample_proba_final[0]}")
except Exception as e:
    print(f"Error during sample prediction test: {e}")

print("\n--- Full Pipeline (FE, Tuning, Training, Saving, Prediction Function) Complete ---")

# --- Step 13: Data Visualization / Graph Analysis (Saving to Files) ---
print("\n--- Starting Step 13: Data Visualization / Graph Analysis (Saving to Files) ---")
# Create a directory to save plots if it doesn't exist
plots_dir = os.path.join(script_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

# --- 13.1: Distribution of First Innings Score ---
print("\nGenerating distribution plot for First Innings Score...")
plt.figure(figsize=(10, 6))
sns.histplot(matches_with_first_innings_score['first_innings_score'], kde=True, bins=20)
plt.title('Distribution of First Innings Scores')
plt.xlabel('First Innings Score')
plt.ylabel('Number of Matches')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plots_dir, 'first_innings_score_distribution.png'))
plt.close()

# --- 13.2: Chase Success Rate vs. First Innings Score Bins ---
print("\nGenerating chase success rate vs. First Innings Score bins plot...")
if 'score_bins' not in matches_with_first_innings_score.columns:
    bins = [0, 120, 140, 160, 180, 200, 220, 250, 300]
    labels = ['<120', '120-139', '140-159', '160-179', '180-199', '200-219', '220-249', '250+']
    matches_with_first_innings_score['score_bins'] = pd.cut(
        matches_with_first_innings_score['first_innings_score'], bins=bins, labels=labels, right=False, include_lowest=True
    )
chase_success_by_score = matches_with_first_innings_score.groupby('score_bins', observed=False)['chase_successful'].mean().reset_index()
plt.figure(figsize=(12, 7))
sns.barplot(x='score_bins', y='chase_successful', data=chase_success_by_score, palette='viridis')
plt.title('Chase Success Rate by First Innings Score Bins')
plt.xlabel('First Innings Score Bins')
plt.ylabel('Chase Success Rate')
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plots_dir, 'chase_success_rate_by_score_bins.png'))
plt.close()

# --- 13.3: Feature Importance from Tuned RandomForestClassifier ---
print("\nGenerating Feature Importance plot...")
if hasattr(model_final_tuned, 'feature_importances_'):
    feature_importances = pd.Series(model_final_tuned.feature_importances_, index=X_train_final.columns)
    top_n_features = 20
    feature_importances_sorted = feature_importances.nlargest(top_n_features)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index, palette='viridis')
    plt.title(f'Top {top_n_features} Feature Importances from Tuned RandomForestClassifier')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importances.png'))
    plt.close()
else:
    print("Model does not have feature_importances_ attribute.")

# --- 13.4: Confusion Matrix Heatmap ---
print("\nGenerating Confusion Matrix Heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tuned, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Fail', 'Predicted Success'],
            yticklabels=['Actual Fail', 'Actual Success'])
plt.title('Confusion Matrix Heatmap (Tuned Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(plots_dir, 'confusion_matrix_heatmap.png'))
plt.close()
print("\n--- Step 13: Data Visualization / Graph Analysis (Saved to Files) Complete ---")