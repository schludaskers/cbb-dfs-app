import pandas as pd
import numpy as np
from datetime import datetime


# Define Fantasy Scoring (DraftKings/FanDuel hybrid)
def calculate_fp(row):
    # PTS(1) + REB(1.25) + AST(1.5) + STL(2) + BLK(2) - TO(-0.5)
    # Using generic averages for simplicity in this demo
    pts = row.get('PTS', 0)
    reb = row.get('TRB', 0)
    ast = row.get('AST', 0)
    stl = row.get('STL', 0)
    blk = row.get('BLK', 0)
    tov = row.get('TOV', 0)

    return pts + (reb * 1.25) + (ast * 1.5) + (stl * 2) + (blk * 2) - (tov * 0.5)


def get_todays_schedule():
    """
    Scrapes Sports-Reference for today's games.
    Falls back to mock data if no games are scheduled or scrape fails.
    """
    try:
        # Get today's date
        today = datetime.now()
        url = f"https://www.sports-reference.com/cbb/boxscores/index.cgi?month={today.month}&day={today.day}&year={today.year}"

        # Read tables from the page
        dfs = pd.read_html(url)

        games = []
        # The structure of these tables is specific; usually pairs of teams
        # We will extract just the team names for the dropdown
        for df in dfs:
            # SportsRef boxscore tables usually list teams in the first column
            if len(df) >= 2:
                teams = df.iloc[:, 0].tolist()
                # Basic cleanup to ensure we got team names
                if len(teams) >= 2:
                    games.append({"Home": teams[0], "Away": teams[1]})

        if not games:
            raise ValueError("No games found table structure mismatch.")

        return pd.DataFrame(games)

    except Exception as e:
        print(f"Scraping failed or no games ({e}). Using Mock Schedule.")
        return pd.DataFrame([
            {"Home": "Duke", "Away": "North Carolina"},
            {"Home": "Kentucky", "Away": "Kansas"},
            {"Home": "Gonzaga", "Away": "Baylor"},
            {"Home": "Purdue", "Away": "Michigan State"}
        ])


def get_player_stats(team_name):
    """
    In a real production app, this would scrape 'https://www.sports-reference.com/cbb/schools/{team_slug}/2024.html'
    For this demo, we generate realistic random player stats to ensure the model runs.
    """
    # Mocking a roster of 8 players per team
    players = []
    positions = ['G', 'G', 'G', 'F', 'F', 'C', 'G', 'F']

    for i in range(8):
        # Generate random stats based on position logic
        is_guard = positions[i] == 'G'

        pts = np.random.randint(5, 25)
        ast = np.random.randint(2, 8) if is_guard else np.random.randint(0, 3)
        reb = np.random.randint(1, 5) if is_guard else np.random.randint(4, 12)

        players.append({
            "Player": f"{team_name} Player {i + 1}",
            "Team": team_name,
            "Pos": positions[i],
            "Opponent": "TBD",  # Placeholder
            "PTS": pts,
            "TRB": reb,
            "AST": ast,
            "STL": np.random.randint(0, 3),
            "BLK": np.random.randint(0, 3),
            "TOV": np.random.randint(0, 4),
            "Last_5_Avg_FP": 0  # To be filled
        })

    df = pd.DataFrame(players)
    df['FantasyPoints_Proj'] = df.apply(calculate_fp, axis=1)

    # Feature Engineering: Add random "Last 5 Games" average for the model
    df['Last_5_Avg_FP'] = df['FantasyPoints_Proj'] * np.random.uniform(0.8, 1.2)
    df['Opp_Def_Rank'] = np.random.randint(1, 360)  # Random opponent rank

    return df