import streamlit as st
import pandas as pd
import xgboost as xgb
import data_fetcher as df_tools

# Page Config
st.set_page_config(page_title="CBB DFS Predictor", layout="wide")

st.title("🏀 College Basketball DFS Predictor")
st.markdown("Select today's matchups to generate fantasy projections using XGBoost.")


# 1. Load Model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("model.json")
    return model


try:
    bst = load_model()
    st.success("Model loaded successfully.")
except:
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

# 2. Get Today's Schedule
st.header("1. Select Games")
schedule = df_tools.get_todays_schedule()

# Create a multiselect for the user to pick games
# We create a display string "Home vs Away"
schedule['Matchup'] = schedule['Home'] + " vs " + schedule['Away']
selected_matchups = st.multiselect("Choose games to analyze:", schedule['Matchup'].tolist())

if st.button("Generate Predictions"):
    if not selected_matchups:
        st.warning("Please select at least one game.")
    else:
        all_projections = []

        st.write("Fetching stats and running model...")

        # Loop through selected games
        progress_bar = st.progress(0)
        for i, match_str in enumerate(selected_matchups):
            # Parse teams back out
            row = schedule[schedule['Matchup'] == match_str].iloc[0]
            home_team = row['Home']
            away_team = row['Away']

            # Fetch players (using our helper)
            # In a real app, we would pass the opponent rank here
            home_players = df_tools.get_player_stats(home_team)
            away_players = df_tools.get_player_stats(away_team)

            # Check if data exists
            if home_players.empty or away_players.empty:
                st.warning(f"Stats missing for {home_team} or {away_team}. Skipping...")
                continue

            # Add context features
            home_players['Is_Home'] = 1
            home_players['Minutes_Avg'] = 30  # Mock value

            away_players['Is_Home'] = 0
            away_players['Minutes_Avg'] = 30  # Mock value

            # Combine
            game_df = pd.concat([home_players, away_players])

            # Define Features
            features = game_df[['Last_5_Avg_FP', 'Opp_Def_Rank', 'Is_Home', 'Minutes_Avg']]

            # Predict
            preds = bst.predict(features)
            game_df['Predicted_FP'] = preds

            all_projections.append(game_df)
            progress_bar.progress((i + 1) / len(selected_matchups))

        # Combine all games into one table
        final_df = pd.concat(all_projections)

        # Display Results
        st.header("2. Projections")

        # Filter options
        min_fp = st.slider("Filter: Minimum Projected Points", 0, 60, 20)
        filtered_df = final_df[final_df['Predicted_FP'] > min_fp].sort_values(by='Predicted_FP', ascending=False)

        # Clean up columns for display
        display_cols = ['Player', 'Team', 'Pos', 'Opp_Def_Rank', 'Last_5_Avg_FP', 'Predicted_FP']
        st.dataframe(filtered_df[display_cols].style.format({'Predicted_FP': '{:.2f}', 'Last_5_Avg_FP': '{:.1f}'}))

        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "cbb_projections.csv", "text/csv")
