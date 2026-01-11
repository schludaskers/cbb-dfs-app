# data_fetcher.py
import pandas as pd
import os


def get_todays_schedule():
    if os.path.exists("todays_schedule.csv"):
        return pd.read_csv("todays_schedule.csv")
    else:
        # Fallback logic
        return pd.DataFrame()


def get_player_stats(team_name):
    # CHANGED: Read the cleaned file
    filename = "daily_stats_cleaned.csv"

    if not os.path.exists(filename):
        # Fallback to check if user hasn't run the fixer yet
        if os.path.exists("daily_stats.csv"):
            print("Warning: Using raw daily_stats.csv. Run fix_data.py for better results.")
            filename = "daily_stats.csv"
        else:
            return pd.DataFrame()

    df = pd.read_csv(filename)

    # Filter for the specific team
    # We use string contains/lower to be more forgiving on matches
    # e.g. "Ohio State" matches "Ohio State"
    team_stats = df[df['Team'].astype(str).str.lower() == team_name.lower()].copy()

    if team_stats.empty:
        # Debug print to help you see mismatches in logs
        # print(f"No stats found for {team_name}")
        return pd.DataFrame()

    return team_stats
