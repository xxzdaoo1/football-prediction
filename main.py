from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load match stats data to extract features
df = pd.read_csv("match_stats_cleaned.csv")

class MatchTeams(BaseModel):
    home_team: str
    away_team: str

@app.post("/predict/")
def predict_score(teams: MatchTeams):
    home_team = teams.home_team
    away_team = teams.away_team

    try:
        # Find latest match data for each team
        home_stats = df[df["team_h"] == home_team].sort_values("date").iloc[-1]
        away_stats = df[df["team_a"] == away_team].sort_values("date").iloc[-1]

        # Build the feature vector
        feature_row = pd.DataFrame([{
            "h_xg": home_stats["h_xg"],
            "a_xg": away_stats["a_xg"],
            "h_shot": home_stats["h_shot"],
            "a_shot": away_stats["a_shot"],
            "h_shotOnTarget": home_stats["h_shotOnTarget"],
            "a_shotOnTarget": away_stats["a_shotOnTarget"],
            "h_deep": home_stats["h_deep"],
            "a_deep": away_stats["a_deep"],
            "h_ppda": home_stats["h_ppda"],
            "a_ppda": away_stats["a_ppda"],
            "team_h_encoded": home_stats["team_h_encoded"],
            "team_a_encoded": away_stats["team_a_encoded"],
        }])

        # Scale and predict
        X_scaled = scaler.transform(feature_row)
        prediction = model.predict(X_scaled)[0]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_pred": round(float(prediction[0])),
            "away_pred": round(float(prediction[1]))
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
