# constants.py

TEAM_NAME_MAPPING = {
    'Mumbai Indians': 'Mumbai Indians',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Lions',
    'Kolkata Knight Riders': 'Kolkata Knight Riders',
    'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
    'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
    'Delhi Daredevils': 'Delhi Capitals',
    'Kings XI Punjab': 'Punjab Kings',
    'Chennai Super Kings': 'Chennai Super Kings',
    'Rajasthan Royals': 'Rajasthan Royals',
    'Deccan Chargers': 'Deccan Chargers',
    'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala',
    'Pune Warriors': 'Pune Warriors',
    'Rising Pune Supergiants': 'Rising Pune Supergiants',
    'Delhi Capitals': 'Delhi Capitals',
    'Punjab Kings': 'Punjab Kings',
    # Add any new teams from matches.csv here if they are present in the raw data
    # and you want their OHE features in the model.
    # e.g., 'Gujarat Titans': 'Gujarat Titans', 'Lucknow Super Giants': 'Lucknow Super Giants'
}

# You can add other global constants here if needed, e.g., DATA_DIR, MODEL_FILENAME etc.