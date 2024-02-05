import pandas as pd
import os
import streamlit as st
import numpy as np
from joblib import load

# Load the dataframe containing the scores
openings_df = pd.read_csv("openings_analysis.csv")

# Load the trained model
model = load('best_logistic_regression_model.joblib')

# Path to the directory where the images are stored
heatmap_dir = 'heatmaps'
piece_mov_dir = 'piece_movement'
game_phase_dir = 'game_phase_play_rate'
opening_heatmaps_dir = 'opening_heatmaps'

# List of unique pieces from the heatmap filenames (e.g., ['Bishop', 'Knight'])
# Assumes heatmap filenames are in the format 'ColorPiece_Perspective_heatmap.png'
unique_pieces = set(filename.split('_')[0].replace('White', '').replace('Black', '') for filename in os.listdir(heatmap_dir))

# Set Streamlit page configuration to use wide mode
st.set_page_config(page_title="Streamlit app", layout="wide")

# Main title for the web page
st.title('Visualization of the dynamics of chess games through data analysis')

# Subtitle
st.header('How should I move my pieces')

# Dropdown to select the piece
selected_piece = st.selectbox('Select a piece', sorted(list(unique_pieces)))

# Sidebar for perspective selection
selected_perspective = st.sidebar.radio('Select a perspective', ('White', 'Black'))

col1, col2 = st.columns(2)

with col1:
    # Display the image for the movement of the selected piece
    piece_mov_img_path = os.path.join(piece_mov_dir, f"{selected_perspective}{selected_piece}.png")
    if os.path.exists(piece_mov_img_path):
        st.image(piece_mov_img_path, caption=f"Movement of {selected_piece}")

with col2:
    # Display the image for the movement of the selected piece
    piece_play_img_path = os.path.join(game_phase_dir, f"{selected_piece}.png")
    if os.path.exists(piece_play_img_path):
        st.image(piece_play_img_path, caption=f"Play rate of {selected_piece} by game phase")


# Creating two columns for White and Black side heatmaps
col3, col4 = st.columns(2)

with col3:
    # Display heatmap for the selected piece from the White side
    white_file_name = f"White{selected_piece}_{selected_perspective}_heatmap.png"
    white_file_path = os.path.join(heatmap_dir, white_file_name)
    if os.path.exists(white_file_path):
        st.image(white_file_path, caption=f"White {selected_piece} ({selected_perspective} Perspective)")

with col4:
    # Display heatmap for the selected piece from the Black side
    black_file_name = f"Black{selected_piece}_{selected_perspective}_heatmap.png"
    black_file_path = os.path.join(heatmap_dir, black_file_name)
    if os.path.exists(black_file_path):
        st.image(black_file_path, caption=f"Black {selected_piece} ({selected_perspective} Perspective)")


# New Section for Openings
st.header('What opening should I play and how should I play it')

# Sidebar for criteria weights
st.sidebar.header('Opening Selector Criteria Weights')
effectiveness_weight = st.sidebar.slider('Weight for Effectiveness', -5.0, 5.0, 0.0, 0.5)
aggressivity_weight = st.sidebar.slider('Weight for Aggressivity', -5.0, 5.0, 0.0, 0.5)
volatility_weight = st.sidebar.slider('Weight for Volatility', -5.0, 5.0, 0.0, 0.5)
popularity_weight = st.sidebar.slider('Weight for Popularity', -5.0, 5.0, 0.0, 0.5)

# Function to calculate score based on criteria
def calculate_score(row):
    effectiveness = row['EffectivenessWhite'] if selected_perspective == 'White' else row['EffectivenessBlack']
    return (effectiveness * effectiveness_weight +
            row['Aggressivity'] * aggressivity_weight +
            row['Volatility'] * volatility_weight +
            row['Popularity'] * popularity_weight)

# Apply the scoring function to the DataFrame
openings_df['OpeningScore'] = openings_df.apply(calculate_score, axis=1)

# Sort the DataFrame based on the calculated score or alphabetically if all weights are 0
if all(w == 0 for w in [effectiveness_weight, aggressivity_weight, volatility_weight, popularity_weight]):
    sorted_openings = openings_df.sort_values('Opening')
else:
    sorted_openings = openings_df.sort_values('OpeningScore', ascending=False)

# Use the sorted DataFrame to populate the selectbox
selected_opening = st.selectbox('Select an Opening', sorted_openings['Opening'])

# Link to Lichess to view the opening moves
st.markdown("Check out your opening : https://lichess.org/opening/"+selected_opening.replace(' ', '_'))

# Display the sorted DataFrame
st.subheader("Sorted Openings")
st.dataframe(sorted_openings[['Opening', 'OpeningScore']])

# Check if selected_opening is valid before proceeding
if selected_opening and selected_opening != 'No openings available':
    opening_dir = os.path.join(opening_heatmaps_dir, selected_opening.replace(':', '_'))


    # Filter and display heatmaps
    if os.path.exists(opening_dir):
        # Get all relevant heatmap files
        heatmap_files = [f for f in os.listdir(opening_dir) if f.endswith(f"_{selected_perspective}_heatmap.png")]
        heatmap_files.sort()  # Sort files

        # Display heatmaps in rows of three
        for i in range(0, len(heatmap_files), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(heatmap_files):
                    file_path = os.path.join(opening_dir, heatmap_files[i + j])
                    with cols[j]:
                        st.image(file_path, caption=heatmap_files[i + j].replace('_', ' ').replace('.png', ''))

# New Section for outcome prediction
st.header('What is the likely outcome of my chess game?')

# User inputs
rating_diff_input = st.number_input("Enter the rating difference (negative if you are rated lower):", min_value=-800, max_value=800, step=10, value=0, format="%d")

# Adjust rating difference based on user color
if selected_perspective == 'Black':
    rating_diff = -rating_diff_input
else:
    rating_diff = rating_diff_input

# Predict button
if st.button('Predict Outcome'):
    # Making prediction
    prediction_proba = model.predict_proba(np.array([[rating_diff]]))

    # Adjusting prediction labels based on user color
    if selected_perspective == 'White':
        win_proba = prediction_proba[0][2]  # Index of WhiteWin
        lose_proba = prediction_proba[0][0]  # Index of BlackWin
    else:
        win_proba = prediction_proba[0][0]  # Index of BlackWin
        lose_proba = prediction_proba[0][2]  # Index of WhiteWin
    
    tie_proba = prediction_proba[0][1]  # Index of Tie

    # Displaying predictions
    st.markdown(f"<h1 style='color: green;'>Win Probability: {win_proba*100:.2f}%</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: red;'>Lose Probability: {lose_proba*100:.2f}%</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color: yellow;'>Tie Probability: {tie_proba*100:.2f}%</h1>", unsafe_allow_html=True)
