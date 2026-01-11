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

        # ... inside app.py ...

        # Loop through selected games
        progress_bar = st.progress(0)
        for i, match_str in enumerate(selected_matchups):
            # ... (parsing logic remains the same) ...

            # Fetch players
            home_players = df_tools.get_player_stats(home_team)
            away_players = df_tools.get_player_stats(away_team)

            # CRITICAL FIX: Skip if empty
            if home_players.empty and away_players.empty:
                st.warning(f"Could not find stats for {home_team} or {away_team}. Check team names.")
                continue

            # Add context features (Fix: Ensure we don't overwrite if columns missing)
            if not home_players.empty:
                home_players['Is_Home'] = 1
                # Ensure Minutes_Avg exists
                if 'Minutes_Avg' not in home_players.columns: home_players['Minutes_Avg'] = 25

            if not away_players.empty:
                away_players['Is_Home'] = 0
                if 'Minutes_Avg' not in away_players.columns: away_players['Minutes_Avg'] = 25

            # Combine
            game_df = pd.concat([home_players, away_players])

            # Predict
            if not game_df.empty:
                # Ensure all feature columns exist
                required_cols = ['Last_5_Avg_FP', 'Opp_Def_Rank', 'Is_Home', 'Minutes_Avg']
                for col in required_cols:
                    if col not in game_df.columns:
                        game_df[col] = 0  # Default fill to prevent crash

                features = game_df[required_cols]
                preds = bst.predict(features)
                game_df['Predicted_FP'] = preds
                all_projections.append(game_df)

            progress_bar.progress((i + 1) / len(selected_matchups))

        # CRITICAL FIX: Check if we have any projections before concatenating
        if all_projections:
            final_df = pd.concat(all_projections)

            # ... (rest of your display logic) ...
            st.header("2. Projections")
            # ...
        else:
            st.error(
                "No projections generated. This usually means the Team Names in the schedule didn't match the Team Names in your stats file.")

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
