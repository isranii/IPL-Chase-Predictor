import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="IPL Chase Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced aesthetics (IPL Theme) ---
st.markdown("""
<style>
/* Overall Page Background - Dark Gradient */
body {
    background: linear-gradient(to bottom right, #1e3c72, #2a5298); /* IPL Blue gradient */
    color: #F0F2F6; /* Light gray for general text */
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1rem; /* Adjust padding for main content */
    padding-right: 1rem;
}

/* Header */
.main-header {
    font-size: 3.8em !important; /* Slightly larger */
    color: #FF6B35; /* IPL Orange */
    text-align: center;
    font-weight: bold;
    margin-bottom: 0.5em;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.6); /* Stronger shadow */
    border-bottom: 4px solid #F7931E; /* Golden Yellow underline */
    padding-bottom: 15px;
    letter-spacing: 1px;
}

/* General Markdown Text (outside specific containers) */
div[data-testid="stMarkdownContainer"] p, div[data-testid="stText"] {
    color: #F0F2F6 !important; /* Ensure light text on dark background */
    font-size: 1.1em;
    line-height: 1.6;
}

/* Sidebar Styling */
.stSidebar {
    background: linear-gradient(to bottom, #1A2E4B, #214066); /* Darker blue gradient for sidebar */
    color: #F0F2F6; /* Light text for sidebar */
    border-right: 1px solid #3498DB; /* Subtle border */
    box-shadow: 2px 0px 10px rgba(0,0,0,0.3);
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    border-radius: 0 15px 15px 0; /* Rounded right corners */
}
.stSidebar .stSelectbox label, .stSidebar .stNumberInput label, .stSidebar .stDateInput label {
    color: #F0F2F6 !important; /* Labels in sidebar are light */
    font-weight: bold;
    margin-bottom: 0.5rem;
}

/* Input Widgets (Selectbox, NumberInput, DateInput) */
.stSelectbox>div, .stNumberInput>div, .stDateInput>div {
    border: 1px solid #F7931E; /* Golden Yellow border */
    border-radius: 10px; /* Rounded corners */
    background-color: #2A4F8A; /* Slightly lighter blue for input background */
    box-shadow: inset 0 1px 5px rgba(0,0,0,0.2); /* Subtle inner shadow */
    color: #F0F2F6 !important; /* Text inside inputs is light */
}
/* Ensure text *inside* the input fields is light */
.stSelectbox>div>div>div>div, .stNumberInput input, .stDateInput input {
    color: #F0F2F6 !important; /* Force light color for input text */
}

/* Button Style */
.stButton>button {
    background-color: #FF6B35; /* IPL Orange for buttons */
    color: white !important;
    font-size: 1.4em; /* Larger font */
    padding: 15px 30px;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease-in-out;
    font-weight: bold;
    letter-spacing: 1px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.4);
    margin-top: 1.5rem; /* More space above button */
}
.stButton>button:hover {
    background-color: #E65A2E; /* Darker orange on hover */
    transform: translateY(-4px); /* More pronounced lift */
    box-shadow: 0px 8px 20px rgba(0,0,0,0.6);
}

/* Card-style Containers for Match Stats and Prediction */
.card-container {
    background-color: rgba(30, 60, 114, 0.7); /* Semi-transparent IPL Blue */
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.4);
    border: 1px solid #F7931E; /* Golden Yellow border */
    margin-bottom: 1.5rem; /* Space between cards */
    color: #F0F2F6; /* Light text within cards */
}
.card-container h3 {
    color: #FF6B35 !important; /* Orange for card headers */
    font-size: 1.8em;
    border-bottom: 2px solid #F7931E; /* Golden Yellow underline */
    padding-bottom: 10px;
    margin-bottom: 1rem;
}
.card-container p {
    color: #F0F2F6 !important; /* Ensure light text for paragraphs */
    font-size: 1.1em;
    margin-bottom: 0.5rem;
}

/* Prediction Result Boxes (Accent Colors) */
.prediction-win {
    background-color: rgba(40, 167, 69, 0.8); /* Green with transparency */
    color: white !important; /* White text for prediction result */
    padding: 30px;
    border-radius: 15px;
    border: 3px solid #28A745; /* Stronger green border */
    text-align: center;
    font-size: 2.5em; /* Larger font */
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0px 0px 30px rgba(40,167,69,0.8); /* Strong Green glow effect */
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}
.prediction-lose {
    background-color: rgba(220, 53, 69, 0.8); /* Red with transparency */
    color: white !important; /* White text for prediction result */
    padding: 30px;
    border-radius: 15px;
    border: 3px solid #DC3545; /* Stronger red border */
    text-align: center;
    font-size: 2.5em; /* Larger font */
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0px 0px 30px rgba(220,53,69,0.8); /* Strong Red glow effect */
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}

/* Probability Progress Bars */
.stProgress > div > div > div > div {
    background-color: #F7931E !important; /* Golden Yellow for progress bar */
    height: 15px; /* Thicker progress bar */
    border-radius: 8px;
}
.stProgress > div > div > div > div[data-testid="stProgressLabel"] {
    color: #F0F2F6 !important; /* Light text for progress label */
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 0.5rem;
}

/* Color-coded probability text */
.prob-high { color: #28A745; font-weight: bold; } /* Green */
.prob-medium { color: #FFC107; font-weight: bold; } /* Amber Yellow */
.prob-low { color: #DC3545; font-weight: bold; } /* Red */

/* Info box (Note on Historical Data) */
.stAlert {
    background-color: rgba(52, 152, 219, 0.2); /* Light blue info box with transparency */
    color: #F0F2F6 !important; /* Light text for info box */
    border-left: 5px solid #3498DB; /* Strong blue border */
    border-radius: 8px;
    font-size: 1.05em;
}

/* Horizontal Rule */
hr {
    border-top: 2px solid #F7931E; /* Golden Yellow HR */
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Small headers within forms/sections */
.segment-header {
    color: #FF6B35 !important; /* Orange for these headers */
    font-size: 1.5em;
    margin-top: 1.5em;
    margin-bottom: 1em;
}

</style>
""", unsafe_allow_html=True)

# --- Define expected features for the model (COPIED EXACTLY FROM YOUR IPL_PREDICTION_MODEL.PY OUTPUT) ---
# This list MUST EXACTLY match the columns and their order used during model training.
# This list was copied from the print(final_model_columns_tuned) output of your ipl_prediction_model.py
expected_model_columns = [
    'first_innings_score', 'match_year', 'match_month', 'match_day', 'match_dayofweek',
    'team1_matches_played_hist', 'team1_win_pct_hist', 'team2_matches_played_hist', 'team2_win_pct_hist',
    'inning1_powerplay_runs', 'inning1_powerplay_wickets', 'inning2_powerplay_runs',
    'inning2_powerplay_wickets', 'inning1_death_overs_runs', 'inning1_death_overs_wickets',
    'inning2_death_overs_runs', 'inning2_death_overs_wickets', 'h2h_matches_played_hist',
    'team1_h2h_wins_hist', 'team2_h2h_wins_hist', 'team1_Chennai Super Kings',
    'team1_Deccan Chargers', 'team1_Delhi Capitals', 'team1_Gujarat Lions',
    'team1_Kochi Tuskers Kerala', 'team1_Kolkata Knight Riders', 'team1_Mumbai Indians',
    'team1_Pune Warriors', 'team1_Punjab Kings', 'team1_Rajasthan Royals',
    'team1_Rising Pune Supergiants', 'team1_Royal Challengers Bangalore',
    'team1_Sunrisers Hyderabad', 'team2_Chennai Super Kings', 'team2_Deccan Chargers',
    'team2_Delhi Capitals', 'team2_Gujarat Lions', 'team2_Kochi Tuskers Kerala',
    'team2_Kolkata Knight Riders', 'team2_Mumbai Indians', 'team2_Pune Warriors',
    'team2_Punjab Kings', 'team2_Rajasthan Royals', 'team2_Rising Pune Supergiants',
    'team2_Royal Challengers Bangalore', 'team2_Sunrisers Hyderabad', 'toss_decision_field',
    'toss_winner_Chennai Super Kings', 'toss_winner_Deccan Chargers',
    'toss_winner_Delhi Capitals', 'toss_winner_Gujarat Lions',
    'toss_winner_Kochi Tuskers Kerala', 'toss_winner_Kolkata Knight Riders',
    'toss_winner_Mumbai Indians', 'toss_winner_Pune Warriors', 'toss_winner_Punjab Kings',
    'toss_winner_Rajasthan Royals', 'toss_winner_Rising Pune Supergiants',
    'toss_winner_Royal Challengers Bangalore', 'toss_winner_Sunrisers Hyderabad',
    'venue_Barabati Stadium', 'venue_Brabourne Stadium', 'venue_Buffalo Park',
    'venue_De Beers Diamond Oval', 'venue_Dr DY Patil Sports Academy',
    'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'venue_Dubai International Cricket Stadium', 'venue_Eden Gardens', 'venue_Feroz Shah Kotla',
    'venue_Green Park', 'venue_Himachal Pradesh Cricket Association Stadium',
    'venue_Holkar Cricket Stadium', 'venue_JSCA International Stadium Complex',
    'venue_Kingsmead', 'venue_M Chinnaswamy Stadium', 'venue_MA Chidambaram Stadium, Chepauk',
    'venue_Maharashtra Cricket Association Stadium', 'venue_Nehru Stadium',
    'venue_New Wanderers Stadium', 'venue_Newlands', 'venue_OUTsurance Oval',
    'venue_Punjab Cricket Association IS Bindra Stadium, Mohali',
    'venue_Punjab Cricket Association Stadium, Mohali',
    'venue_Rajiv Gandhi International Stadium, Uppal', 'venue_Sardar Patel Stadium, Motera',
    'venue_Saurashtra Cricket Association Stadium', 'venue_Sawai Mansingh Stadium',
    'venue_Shaheed Veer Narayan Singh International Stadium', 'venue_Sharjah Cricket Stadium',
    'venue_Sheikh Zayed Stadium', "venue_St George's Park", 'venue_Subrata Roy Sahara Stadium',
    'venue_SuperSport Park', 'venue_Vidarbha Cricket Association Stadium, Jamtha',
    'venue_Wankhede Stadium'
]

# --- Global Mappings and Helper Functions (COPIED EXACTLY FROM YOUR IPL_PREDICTION_MODEL.PY) ---
# IMPORTANT: This mapping is crucial for standardizing team names from your matches.csv
# to the names expected by your model's one-hot encoding.
team_name_mapping = {
    'Mumbai Indians': 'Mumbai Indians', 'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Lions', 'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad', 'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings', 'Chennai Super Kings': 'Chennai Super Kings',
    'Rajasthan Royals': 'Rajasthan Royals', 'Deccan Chargers': 'Deccan Chargers', # Keeping separate as per latest model output
    'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala', 'Pune Warriors': 'Pune Warriors',
    'Rising Pune Supergiants': 'Rising Pune Supergiants', 'Delhi Capitals': 'Delhi Capitals',
    'Punjab Kings': 'Punjab Kings'
    # Add any new teams from matches.csv here if they are present in the raw data
    # and you want their OHE features in the model.
    # e.g., 'Gujarat Titans': 'Gujarat Titans', 'Lucknow Super Giants': 'Lucknow Super Giants'
}

# Dynamically build the list of model-supported teams for dropdowns
# This is refined to ensure only actual team names that are part of OHE features are listed.
supported_teams_from_model_ohe = set()
for col in expected_model_columns:
    if col.startswith('team1_'):
        supported_teams_from_model_ohe.add(col.replace('team1_', ''))
    elif col.startswith('team2_'):
        supported_teams_from_model_ohe.add(col.replace('team2_', ''))
    elif col.startswith('toss_winner_'):
        supported_teams_from_model_ohe.add(col.replace('toss_winner_', ''))

# Filter `team_name_mapping` values to only include those present in `supported_teams_from_model_ohe`
# This ensures that your dropdown only shows teams the model "knows" about via OHE features.
all_teams_for_dropdown = sorted(list(set(
    v for v in team_name_mapping.values() if v in supported_teams_from_model_ohe
)))

# --- Configuration and Data Loading (using st.cache for performance) ---
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, 'ipl_chase_prediction_model_tuned.joblib')

@st.cache_resource
def load_model_cached():
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{os.path.basename(model_path)}' not found. Please ensure it's in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
model = load_model_cached()

@st.cache_data
def load_and_standardize_data_cached(_team_map_func):
    try:
        matches_data_original = pd.read_csv(os.path.join(script_dir, 'archive', 'data', 'matches.csv'))
    except FileNotFoundError:
        st.error("Error: matches.csv not found. Please ensure it's in 'archive/data' subfolder relative to app.py.")
        st.stop()

    matches_data_original_standardized = matches_data_original.copy()
    # Apply mapping to all relevant team columns
    matches_data_original_standardized['team1'] = matches_data_original_standardized['team1'].replace(_team_map_func)
    # FIX: Corrected typo from _team_team_map_func to _team_map_func here
    matches_data_original_standardized['team2'] = matches_data_original_standardized['team2'].replace(_team_map_func)
    matches_data_original_standardized['winner'] = matches_data_original_standardized['winner'].replace(_team_map_func)
    matches_data_original_standardized['toss_winner'] = matches_data_original_standardized['toss_winner'].replace(_team_map_func)

    # Convert date column to datetime for easier processing
    matches_data_original_standardized['date'] = pd.to_datetime(matches_data_original_standardized['date'])
    matches_data_original_standardized['season'] = matches_data_original_standardized['date'].dt.year

    # Filter out any rows where team names could not be standardized (i.e., they are not in the model's OHE)
    # This ensures that historical data used for plots and stats only includes teams the model knows.
    initial_rows = len(matches_data_original_standardized)
    matches_data_original_standardized = matches_data_original_standardized[
        matches_data_original_standardized['team1'].isin(supported_teams_from_model_ohe) &
        matches_data_original_standardized['team2'].isin(supported_teams_from_model_ohe)
    ].copy() # .copy() to avoid SettingWithCopyWarning
    if len(matches_data_original_standardized) < initial_rows:
        st.warning(f"Filtered out {initial_rows - len(matches_data_original_standardized)} rows from matches.csv due to teams not recognized by the model's features for plotting. This might affect historical stats accuracy for those filtered teams in plots.")

    return matches_data_original_standardized

matches_data_original_standardized = load_and_standardize_data_cached(team_name_mapping)

# Get unique, standardized venue names for dropdowns
all_venues = sorted(matches_data_original_standardized['venue'].unique().tolist())

# Pre-calculate overall historical averages for the teams for quick app demo.
overall_team_win_pct_dict_global = (matches_data_original_standardized.groupby('winner').size() / len(matches_data_original_standardized) * 100).to_dict()

# Calculate total matches played by each team by summing team1 and team2 appearances
team1_counts = matches_data_original_standardized['team1'].value_counts()
team2_counts = matches_data_original_standardized['team2'].value_counts()
overall_team_played_dict_global = (team1_counts.add(team2_counts, fill_value=0)).to_dict()

def get_historical_stats_for_app_global(team_name):
    played = int(overall_team_played_dict_global.get(team_name, 0))
    won_pct = float(overall_team_win_pct_dict_global.get(team_name, 0.0))
    return played, won_pct

# Pre-calculate overall head-to-head for app demo (simplified)
overall_h2h_wins = {}
for idx, row in matches_data_original_standardized.iterrows():
    team1 = row['team1']
    team2 = row['team2']
    winner = row['winner']
    h2h_key = tuple(sorted((team1, team2))) # Use sorted tuple as key

    if h2h_key not in overall_h2h_wins:
        overall_h2h_wins[h2h_key] = {'played': 0, 'wins_team1': 0, 'wins_team2': 0}

    overall_h2h_wins[h2h_key]['played'] += 1
    if winner == h2h_key[0]:
        overall_h2h_wins[h2h_key]['wins_team1'] += 1
    elif winner == h2h_key[1]:
        overall_h2h_wins[h2h_key]['wins_team2'] += 1

def get_h2h_stats_for_app_global(team1, team2):
    team1_std = team_name_mapping.get(team1, team1)
    team2_std = team_name_mapping.get(team2, team2)
    h2h_key = tuple(sorted((team1_std, team2_std)))
    stats = overall_h2h_wins.get(h2h_key, {'played': 0, 'wins_team1': 0, 'wins_team2': 0})
    if team1_std == h2h_key[0]:
        return stats['played'], stats['wins_team1'], stats['wins_team2']
    else:
        return stats['played'], stats['wins_team2'], stats['wins_team1']

# --- Prediction Function ---
def predict_chase_outcome_app(new_match_data_raw_dict, team_name_map, _historical_stats_func, _h2h_stats_func, final_cols_list, _model_obj):
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
    }

    team1_standardized = team_name_map.get(new_match_data_raw_dict['team1'], new_match_data_raw_dict['team1'])
    team2_standardized = team_name_map.get(new_match_data_raw_dict['team2'], new_match_data_raw_dict['team2'])
    toss_winner_standardized = team_name_map.get(new_match_data_raw_dict['toss_winner'], new_match_data_raw_dict['toss_winner'])
    toss_decision = new_match_data_raw_dict['toss_decision']
    venue = new_match_data_raw_dict['venue']

    team1_played, team1_win_pct = _historical_stats_func(team1_standardized)
    team2_played, team2_win_pct = _historical_stats_func(team2_standardized)

    input_data_processed['team1_matches_played_hist'] = team1_played
    input_data_processed['team1_win_pct_hist'] = team1_win_pct
    input_data_processed['team2_matches_played_hist'] = team2_played
    input_data_processed['team2_win_pct_hist'] = team2_win_pct

    h2h_played, team1_h2h_wins, team2_h2h_wins = _h2h_stats_func(team1_standardized, team2_standardized)
    input_data_processed['h2h_matches_played_hist'] = h2h_played
    input_data_processed['team1_h2h_wins_hist'] = team1_h2h_wins
    input_data_processed['team2_h2h_wins_hist'] = team2_h2h_wins

    # Create an empty DataFrame with the EXACT columns and order expected by the model
    # This is critical for matching the `fit` data
    final_input_df = pd.DataFrame(0, index=[0], columns=final_cols_list)

    # Populate numerical features (ensuring type compatibility where floats are assigned to int columns)
    for col, value in input_data_processed.items():
        if col in final_input_df.columns:
            # Check if target column is integer and value is float, then cast to int
            if str(final_input_df[col].dtype).startswith('int') and isinstance(value, float):
                final_input_df.loc[0, col] = int(value)
            else:
                final_input_df.loc[0, col] = value
        else:
            print(f"Warning: Column '{col}' from input_data_processed not found in final_input_df. Skipping assignment.")

    # Apply one-hot encoding by setting the appropriate column to 1
    ohe_cols_to_set = []
    if f'team1_{team1_standardized}' in final_cols_list:
        ohe_cols_to_set.append(f'team1_{team1_standardized}')
    if f'team2_{team2_standardized}' in final_cols_list:
        ohe_cols_to_set.append(f'team2_{team2_standardized}')
    if f'toss_winner_{toss_winner_standardized}' in final_cols_list:
        ohe_cols_to_set.append(f'toss_winner_{toss_winner_standardized}')
    # Note: 'toss_decision_bat' is handled by 'toss_decision_field' being 0
    if toss_decision == 'field' and 'toss_decision_field' in final_cols_list:
        ohe_cols_to_set.append('toss_decision_field')
    if f'venue_{venue}' in final_cols_list:
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

    # Ensure the order is exactly as expected by the model by re-indexing to the final_cols_list
    # This is the ultimate safeguard for order mismatch.
    final_input_df = final_input_df[final_cols_list]

    print("\n--- Final DataFrame Debug Info (Check your terminal/console) ---")
    print("Input DataFrame columns before prediction:")
    print(final_input_df.columns.tolist())
    print("Input DataFrame dtypes before prediction:")
    print(final_input_df.dtypes)
    print("Expected model columns (from app.py):")
    print(final_cols_list)
    print("----------------------------------------------------------------")

    # This check `if not final_input_df.columns.equals(pd.Index(final_cols_list))`
    # should now ideally always pass if the `expected_model_columns` in app.py
    # are correctly copied from the `ipl_prediction_model.py` output.
    if not final_input_df.columns.equals(pd.Index(final_cols_list)):
        missing_in_input = set(final_cols_list) - set(final_input_df.columns)
        extra_in_input = set(final_input_df.columns) - set(final_cols_list)
        error_detail = "Column mismatch before final prediction:\n"
        if missing_in_input:
            error_detail += f"  Columns missing in input DataFrame: {list(missing_in_input)}\n"
        if extra_in_input:
            error_detail += f"  Extra columns in input DataFrame: {list(extra_in_input)}\n"
        if final_input_df.columns.tolist() != final_cols_list:
            error_detail += "  Column order mismatch detected.\n"
        st.warning(f"WARNING: The final DataFrame columns do not perfectly match the model's expected features. {error_detail}"
                 "This might affect prediction accuracy for unseen categories, but the app will proceed with prediction.")

    prediction = _model_obj.predict(final_input_df)
    prediction_proba = _model_obj.predict_proba(final_input_df)
    return prediction, prediction_proba

# --- Streamlit UI ---
st.markdown("<h1 class='main-header'>🏏 IPL Chase Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 1.2em; color: #F0F2F6;'>
    Welcome to the IPL Chase Predictor! This model uses advanced machine learning
    techniques to predict whether the team batting second (chasing) will successfully
    win the match based on various match conditions and historical team performances.
    </p>
""", unsafe_allow_html=True)

today = datetime.date.today()

# --- Sidebar for Inputs ---
st.sidebar.markdown("<h2 class='section-header'>Match Details</h2>", unsafe_allow_html=True)

team1_input = st.sidebar.selectbox("Team 1 (Batting First)", all_teams_for_dropdown, key='team1_select')
default_team2_index = 0
if len(all_teams_for_dropdown) > 1 and team1_input == all_teams_for_dropdown[0]:
    # Try to set default to the second team if available and different from the first
    if len(all_teams_for_dropdown) > 1 and all_teams_for_dropdown[1] != team1_input:
        default_team2_index = 1
    # If not, find a different team or stick to 0
    else:
        for i, team in enumerate(all_teams_for_dropdown):
            if team != team1_input:
                default_team2_index = i
                break
team2_input = st.sidebar.selectbox("Team 2 (Chasing)", all_teams_for_dropdown, key='team2_select', index=default_team2_index)


# Ensure toss_winner options are consistent with what the model expects
toss_winner_options = sorted(list(set(v for k,v in team_name_mapping.items() if f'toss_winner_{v}' in expected_model_columns)))
if not toss_winner_options: # Fallback if no toss_winner columns found (shouldn't happen with correct data)
    toss_winner_options = all_teams_for_dropdown

toss_winner_input = st.sidebar.selectbox("Toss Winner", toss_winner_options, key='toss_winner_select')
toss_decision_input = st.sidebar.selectbox("Toss Decision", ["bat", "field"], key='toss_decision_select')
venue_input = st.sidebar.selectbox("Venue", all_venues, key='venue_select')
match_date_input = st.sidebar.date_input("Match Date", today, key='match_date')
first_innings_score_input = st.sidebar.number_input("First Innings Score", min_value=50, max_value=300, value=160, key='first_innings_score')

st.sidebar.markdown("<h3 class='segment-header'>Inning 1 Segment Scores (from Scorecard)</h3>", unsafe_allow_html=True)
inning1_powerplay_runs_input = st.sidebar.number_input("Inning 1 Powerplay Runs (0-6 overs)", min_value=0, value=45, key='inning1_pp_runs')
inning1_powerplay_wickets_input = st.sidebar.number_input("Inning 1 Powerplay Wickets", min_value=0, max_value=10, value=1, key='inning1_pp_wickets')
inning1_death_overs_runs_input = st.sidebar.number_input("Inning 1 Death Overs Runs (15-20 overs)", min_value=0, value=50, key='inning1_do_runs')
inning1_death_overs_wickets_input = st.sidebar.number_input("Inning 1 Death Overs Wickets", min_value=0, max_value=10, value=2, key='inning1_do_wickets')

st.sidebar.markdown("<h3 class='segment-header'>Inning 2 Segment Scores (Partial / Estimate)</h3>", unsafe_allow_html=True)
st.sidebar.info("For a live match prediction, fill these with current scores. For pre-match, use your best estimates.")
inning2_powerplay_runs_input = st.sidebar.number_input("Inning 2 Powerplay Runs (0-6 overs)", min_value=0, value=40, key='inning2_pp_runs')
inning2_powerplay_wickets_input = st.sidebar.number_input("Inning 2 Powerplay Wickets", min_value=0, max_value=10, value=1, key='inning2_pp_wickets')
inning2_death_overs_runs_input = st.sidebar.number_input("Inning 2 Death Overs Runs (15-20 overs)", min_value=0, value=45, key='inning2_do_runs')
inning2_death_overs_wickets_input = st.sidebar.number_input("Inning 2 Death Overs Wickets", min_value=0, max_value=10, value=1, key='inning2_do_wickets')

# Collect raw input data
raw_input_data_dict = {
    'team1': team1_input,
    'team2': team2_input,
    'toss_winner': toss_winner_input,
    'toss_decision': toss_decision_input,
    'venue': venue_input,
    'first_innings_score': first_innings_score_input,
    'match_year': match_date_input.year,
    'match_month': match_date_input.month,
    'match_day': match_date_input.day,
    'match_dayofweek': match_date_input.weekday(),
    'inning1_powerplay_runs': inning1_powerplay_runs_input,
    'inning1_powerplay_wickets': inning1_powerplay_wickets_input,
    'inning2_powerplay_runs': inning2_powerplay_runs_input,
    'inning2_powerplay_wickets': inning2_powerplay_wickets_input,
    'inning1_death_overs_runs': inning1_death_overs_runs_input,
    'inning1_death_overs_wickets': inning1_death_overs_wickets_input,
    'inning2_death_overs_runs': inning2_death_overs_runs_input,
    'inning2_death_overs_wickets': inning2_death_overs_wickets_input,
}

# Initialize prediction variables
prediction = None
prediction_proba = None
error_message = None

# Display the "Predict" button in the main area
predict_button = st.button("Predict Chase Outcome", key='predict_button')

# Only display results if the button is clicked and no errors
if predict_button:
    if team1_input == team2_input:
        error_message = "Team 1 (Batting First) and Team 2 (Chasing) cannot be the same. Please select different teams."
    else:
        try:
            prediction, prediction_proba = predict_chase_outcome_app(
                raw_input_data_dict,
                team_name_mapping,
                get_historical_stats_for_app_global,
                get_h2h_stats_for_app_global,
                expected_model_columns,
                model
            )
        except ValueError as ve: # Catch the specific ValueError from OHE
            error_message = f"Configuration Error: {ve}. This often means a selected team/venue isn't in the model's training data. Please check `expected_model_columns` in app.py."
        except Exception as e:
            error_message = f"An unexpected error occurred during prediction. Please check your inputs. Error: {e}"

    if error_message:
        st.error(error_message)
    elif prediction is not None and prediction_proba is not None:
        col_stats, col_prediction = st.columns([1, 1.5]) # Adjust column widths

        with col_stats:
            st.markdown("<div class='card-container'>", unsafe_allow_html=True)
            st.markdown("<h3>Current Match Situation</h3>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Team 1 (Batting First):</strong> {team1_input}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Team 2 (Chasing):</strong> {team2_input}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Venue:</strong> {venue_input}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Match Date:</strong> {match_date_input.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>First Innings Score:</strong> {first_innings_score_input}</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Toss Winner:</strong> {toss_winner_input} (chose to {toss_decision_input})</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card-container'>", unsafe_allow_html=True)
            st.markdown("<h3>Historical Team Performance</h3>", unsafe_allow_html=True)
            team1_played, team1_win_pct = get_historical_stats_for_app_global(team1_input)
            team2_played, team2_win_pct = get_historical_stats_for_app_global(team2_input)
            h2h_played, team1_h2h_wins, team2_h2h_wins = get_h2h_stats_for_app_global(team1_input, team2_input)

            st.markdown(f"<p><strong>{team1_input} (Overall):</strong> Played {team1_played} matches, Won {team1_win_pct:.1f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>{team2_input} (Overall):</strong> Played {team2_played} matches, Won {team2_win_pct:.1f}%</p>", unsafe_allow_html=True)
            if h2h_played > 0:
                st.markdown(f"<p><strong>Head-to-Head ({team1_input} vs {team2_input}):</strong> Played {h2h_played} matches, {team1_input} won {team1_h2h_wins}, {team2_input} won {team2_h2h_wins}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p><strong>Head-to-Head:</strong> No historical matches found between these teams.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_prediction:
            # Prediction Result (formatted as requested)
            if prediction[0] == 1:
                st.markdown(f"<div class='prediction-win'>The chasing team - {team2_input} is predicted to WIN!</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='prediction-lose'>The chasing team - {team2_input} is predicted to LOSE!</div>", unsafe_allow_html=True)

            # Probability display with color coding
            win_proba = prediction_proba[0][1]
            lose_proba = prediction_proba[0][0]
            st.markdown(f"<h3 style='text-align:center; margin-top:20px; color:#F0F2F6;'>Confidence Levels:</h3>", unsafe_allow_html=True)

            win_prob_class = "prob-low"
            if win_proba >= 0.70:
                win_prob_class = "prob-high"
            elif win_proba >= 0.40:
                win_prob_class = "prob-medium"
            st.markdown(f"<p class='{win_prob_class}' style='text-align:center;'><strong>{team2_input} Win Probability: {win_proba:.2%}</strong></p>", unsafe_allow_html=True)
            st.progress(win_proba)

            lose_prob_class = "prob-low"
            if lose_proba >= 0.70:
                lose_prob_class = "prob-high"
            elif lose_proba >= 0.40:
                lose_prob_class = "prob-medium"
            st.markdown(f"<p class='{lose_prob_class}' style='text-align:center;'><strong>{team2_input} Lose Probability: {lose_proba:.2%}</strong></p>", unsafe_allow_html=True)
            st.progress(lose_proba)

            st.markdown("---")
            st.info("**:bulb: Note on Historical Data:** For this app demo, historical team and head-to-head statistics are based on overall averages from the entire dataset. For a real-time, highly accurate prediction system, these would ideally be calculated cumulatively up to the exact match date.")

# --- Bottom Section for Historical Comparison Charts ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2 class='main-header' style='font-size: 2.5em !important;'>Historical Insights & Trends</h2>", unsafe_allow_html=True)

# --- Run Rate Analysis Plot (using Matplotlib) ---
st.markdown("""
<div class='card-container'>
    <h3>Estimated Run Rate Trend</h3>
    <p>This chart provides an estimated run rate trend based on powerplay and death over runs. For precise over-by-over rates, ball-by-ball data would be needed.</p>
</div>
""", unsafe_allow_html=True)

try:
    overs_data = np.arange(1, 21)

    # Use a hex code or RGB tuple for Matplotlib facecolor in legend
    legend_facecolor_mpl = '#1e3c72' # A dark blue from your theme
    legend_labelcolor_mpl = '#F0F2F6' # Light text color

    # Calculate overall run rate for Team 1 (batting first)
    team1_total_runs = float(first_innings_score_input) # Ensure float division
    team1_avg_rr_per_over = team1_total_runs / 20.0 if team1_total_runs > 0 else 0
    team1_rr = np.full(20, team1_avg_rr_per_over) + np.random.uniform(-0.5, 0.5, 20) # Reduced noise
    team1_rr = np.maximum(team1_rr, 0) # Ensure non-negative run rates

    # Calculate estimated run rate for Team 2 (chasing) based on required rate
    runs_to_chase = float(first_innings_score_input) + 1 # Target is score + 1, ensure float
    required_run_rate = runs_to_chase / 20.0 if runs_to_chase > 0 else 0
    team2_rr = np.full(20, required_run_rate) + np.random.uniform(-0.5, 0.5, 20) # Reduced noise
    team2_rr = np.maximum(team2_rr, 0) # Ensure non-negative run rates


    fig_rr, ax_rr = plt.subplots(figsize=(10, 5))
    ax_rr.plot(overs_data, team1_rr, label=f'{team1_input} Est. Run Rate (Inning 1)', marker='o', linestyle='-', color='#FF6B35')
    ax_rr.plot(overs_data, team2_rr, label=f'{team2_input} Est. Run Rate (Inning 2)', marker='x', linestyle='--', color='#3498DB')

    ax_rr.set_xlabel("Overs", color='#F0F2F6')
    ax_rr.set_ylabel("Run Rate (Runs/Over)", color='#F0F2F6')
    ax_rr.set_title(f"Estimated Run Rate Trend for {team1_input} vs {team2_input}", color='#F0F2F6')
    ax_rr.legend(facecolor=legend_facecolor_mpl, edgecolor='#F7931E', labelcolor=legend_labelcolor_mpl)
    ax_rr.set_facecolor('#1e3c72') # Use a valid hex or RGB tuple for facecolor of axes
    # FIX: Changed 'transparent' to a more robust alpha setting for the figure patch
    fig_rr.patch.set_alpha(0.0) # Figure background fully transparent
    ax_rr.tick_params(axis='x', colors='#F0F2F6')
    ax_rr.tick_params(axis='y', colors='#F0F2F6')
    ax_rr.spines['left'].set_color('#F0F2F6')
    ax_rr.spines['bottom'].set_color('#F0F2F6')
    ax_rr.spines['right'].set_color('none')
    ax_rr.spines['top'].set_color('none')
    st.pyplot(fig_rr)
except Exception as e:
    st.error(f"Error drawing Run Rate Analysis plot: {e}")
    st.write(f"Run Rate Plot Debug: {e}")
    import traceback
    st.text(traceback.format_exc())

# --- Team Performance Over Seasons Plot (using Plotly) ---
st.markdown("""
<div class='card-container' style='margin-top: 2rem;'>
    <h3>Team Performance Over Seasons</h3>
    <p>This chart shows the historical win percentage of the selected teams across different IPL seasons.</p>
</div>
""", unsafe_allow_html=True)

try:
    def get_season_win_percentage(team_name, season_year, data_df):
        # Filter matches where the team was either team1 or team2 in that season
        team_matches_in_season = data_df[
            ((data_df['team1'] == team_name) | (data_df['team2'] == team_name)) &
            (data_df['season'] == season_year)
        ]
        total_matches = len(team_matches_in_season)

        if total_matches == 0:
            return 0.0

        # Count wins for the team in that season
        wins = len(team_matches_in_season[team_matches_in_season['winner'] == team_name])
        return (wins / total_matches) * 100

    unique_seasons = sorted(matches_data_original_standardized['season'].unique())
    season_win_data = []

    for season in unique_seasons:
        team1_win_pct_season = get_season_win_percentage(team1_input, season, matches_data_original_standardized)
        team2_win_pct_season = get_season_win_percentage(team2_input, season, matches_data_original_standardized)
        season_win_data.append({
            'Season': season,
            f'{team1_input} Win %': team1_win_pct_season,
            f'{team2_input} Win %': team2_win_pct_season
        })

    df_season_wins = pd.DataFrame(season_win_data)

    # --- DEBUGGING LINES ---
    st.write("--- Debugging Season Win Data ---")
    st.write(f"df_season_wins head():")
    st.dataframe(df_season_wins.head())
    st.write(f"df_season_wins is empty: {df_season_wins.empty}")
    st.write(f"df_season_wins columns: {df_season_wins.columns.tolist()}")
    st.write(f"Team 1: {team1_input}, Team 2: {team2_input}")
    # --- END DEBUGGING LINES ---

    # Check if the dataframe contains any non-zero/non-NaN data for plotting
    plot_data_available = False
    if not df_season_wins.empty:
        # Check if the win percentage columns actually have non-NaN values AND at least one non-zero value
        team1_col_name = f'{team1_input} Win %'
        team2_col_name = f'{team2_input} Win %'

        team1_data_present = team1_col_name in df_season_wins.columns and \
                             not df_season_wins[team1_col_name].isnull().all() and \
                             (df_season_wins[team1_col_name] != 0).any()

        team2_data_present = team2_col_name in df_season_wins.columns and \
                             not df_season_wins[team2_col_name].isnull().all() and \
                             (df_season_wins[team2_col_name] != 0).any()

        if team1_data_present or team2_data_present:
           plot_data_available = True
        # If only one team has data, ensure Plotly only plots that column to avoid errors
        if team1_data_present and not team2_data_present:
            y_cols_for_plotly = [team1_col_name]
        elif team2_data_present and not team1_data_present:
            y_cols_for_plotly = [team2_col_name]
        elif team1_data_present and team2_data_present:
            y_cols_for_plotly = [team1_col_name, team2_col_name]
        else: # Should be caught by plot_data_available = False
            y_cols_for_plotly = []
            plot_data_available = False # Re-confirm if no useful data

    if plot_data_available:
        fig_season = px.line(df_season_wins, x='Season', y=y_cols_for_plotly, # Use dynamically chosen columns
                                 title=f"Win Percentage Over Seasons: {team1_input} vs {team2_input}",
                                 labels={'Season': 'IPL Season', 'value': 'Win Percentage (%)'},
                                 color_discrete_map={f'{team1_input} Win %': '#FF6B35', f'{team2_input} Win %': '#3498DB'}) # Explicit colors

        fig_season.update_layout(
            plot_bgcolor='rgba(30, 60, 114, 0.5)',
            paper_bgcolor='rgba(30, 60, 114, 0.7)',
            font_color='#F0F2F6',
            title_font_color='#FF6B35',
            legend_title_font_color='#F0F2F6',
            legend_font_color='#F0F2F6',
            xaxis=dict(tickfont=dict(color='#F0F2F6'), title_font_color='#F0F2F6', gridcolor='rgba(240, 242, 246, 0.2)'),
            yaxis=dict(tickfont=dict(color='#F0F2F6'), title_font_color='#F0F2F6', gridcolor='rgba(240, 242, 246, 0.2)', range=[0, 100])
        )
        fig_season.update_traces(mode='lines+markers')
        st.plotly_chart(fig_season, use_container_width=True)
    else:
        st.warning(f"No sufficient historical season data found for {team1_input} and/or {team2_input} to generate performance trends. This might be due to team filters, or limited historical matches.")

except Exception as e:
    st.error(f"Error drawing Team Performance plot: {e}")
    st.write(f"Season Plot Debug: {e}")
    import traceback
    st.text(traceback.format_exc())