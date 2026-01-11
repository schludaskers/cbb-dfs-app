import streamlit as st
import pandas as pd
import xgboost as xgb
import data_fetcher as df_tools
import os

# Page Config
st.set_page_config(page_title="CBB DFS Predictor", layout="wide")

st.title("🏀 College Basketball DFS Predictor")
st.markdown("Select today's matchups to generate fantasy projections using XGBoost.")

# 1. Load Model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    # Check if model exists, otherwise warn user
    if os.path.exists("model.json"):
        model.load_model("model.json")
    else:
        st.error("⚠️ 'model.json' not found. Please run 'train_model.py' locally and upload the file.")
        return None
    return model

bst = load_model()
if bst is None:
    st.stop() # Stop app if no model

# 2. Get Today's Schedule
st.header("1. Select Games")
schedule = df_tools.get_todays_schedule()

if schedule.empty:
    st.warning("No games found in 'todays_schedule.csv'. Please run the scraper/generator.")
    st.stop()

# Create a multiselect for the user to pick games
# We create a display string "Home vs Away"
if 'Matchup' not in schedule.columns:
    # Create Matchup column if missing
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
            try:
                row = schedule[schedule['Matchup'] == match_str].iloc[0]
                home_team = row['Home']
                away_team = row['Away']
            except IndexError:
                continue

            # Fetch players (using our helper)
            home_players = df_tools.get_player_stats(home_team)
            away_players = df_tools.get_player_stats(away_team)
            
            # Skip if stats are missing for BOTH teams
            if home_players.empty and away_players.empty:
                st.warning(f"⚠️ Missing stats for {home_team} vs {away_team}. Skipping...")
                continue
            
            # Prepare data for Home Team
            if not home_players.empty:
                home_players['Is_Home'] = 1
                if 'Minutes_Avg' not in home_players.columns:
                    home_players['Minutes_Avg'] = 25 # Default
            
            # Prepare data for Away Team
            if not away_players.empty:
                away_players['Is_Home'] = 0
                if 'Minutes_Avg' not in away_players.columns:
                    away_players['Minutes_Avg'] = 25 # Default
            
            # Combine into one game dataframe
            game_df = pd.concat([home_players, away_players])
            
            if not game_df.empty:
                # Ensure all feature columns exist for the model
                # Model expects: ['Last_5_Avg_FP', 'Opp_Def_Rank', 'Is_Home', 'Minutes_Avg']
                required_cols = ['Last_5_Avg_FP', 'Opp_Def_Rank', 'Is_Home', 'Minutes_Avg']
                
                # Fill missing columns with 0 to prevent crash
                for col in required_cols:
                    if col not in game_df.columns:
                        game_df[col] = 0 
                
                features = game_df[required_cols]
                
                # Predict
                preds = bst.predict(features)
                game_df['Predicted_FP'] = preds
                
                all_projections.append(game_df)
            
            # Update progress
            progress_bar.progress((i + 1) / len(selected_matchups))
            
        # Combine all games into one table
        if all_projections:
            final_df = pd.concat(all_projections)
            
            # Display Results
            st.header("2. Projections")
            
            # Filter options
            min_fp = st.slider("Filter: Minimum Projected Points", 0, 60, 15)
            filtered_df = final_df[final_df['Predicted_FP'] > min_fp].sort_values(by='Predicted_FP', ascending=False)
            
            # Clean up columns for display
            display_cols = ['Player', 'Team', 'Pos', 'Opp_Def_Rank', 'Last_5_Avg_FP', 'Predicted_FP']
            
            # Only show columns that actually exist
            existing_cols = [c for c in display_cols if c in filtered_df.columns]
            
            st.dataframe(filtered_df[existing_cols].style.format({'Predicted_FP': '{:.2f}', 'Last_5_Avg_FP': '{:.1f}'}))
            
            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "cbb_projections.csv", "text/csv")
        else:
            st.error("No projections generated. Please check if your Team Names in 'todays_schedule.csv' match the names in 'daily_stats_cleaned.csv'.")
