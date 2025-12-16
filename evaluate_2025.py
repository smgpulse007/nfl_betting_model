"""
Evaluate 2025 Season Performance (Weeks 1-15)

This script analyzes how well our model predicted 
the completed 2025 games.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss, accuracy_score, log_loss

import sys
sys.path.append(str(Path(__file__).parent))

from config import PROCESSED_DATA_DIR
from src.models import NFLBettingModels
from src.backtesting import implied_prob, Backtester

DATA_2025 = PROCESSED_DATA_DIR.parent / "2025"


def main():
    print("=" * 70)
    print("2025 SEASON MODEL PERFORMANCE (Weeks 1-15)")
    print("=" * 70)

    # 1. Load historical data (1999-2024) for training
    print("\n[1/5] Loading training data (1999-2024)...")
    historical = pd.read_parquet(PROCESSED_DATA_DIR / "games_with_features.parquet")
    print(f"  Training games: {len(historical)}")

    # 2. Load 2025 completed games
    print("\n[2/5] Loading 2025 completed games...")
    completed_2025 = pd.read_parquet(DATA_2025 / "completed_2025.parquet")
    print(f"  2025 completed games: {len(completed_2025)}")

    # 3. Train model on historical data only (no 2025 data leakage)
    print("\n[3/5] Training models on 1999-2024 data...")
    models = NFLBettingModels()
    models.fit(historical)

    # 4. Generate predictions for 2025
    print("\n[4/5] Generating predictions for 2025...")
    preds_2025 = models.predict(completed_2025)

    # Merge predictions with actual results
    results = completed_2025.merge(
        preds_2025, on=["game_id", "season", "week", "home_team", "away_team"]
    )

    # 5. Evaluate performance
    print("\n[5/5] Evaluating performance...")

    y_true = results["home_win"].values

    print("\n" + "=" * 70)
    print("OVERALL MODEL PERFORMANCE - 2025 SEASON")
    print("=" * 70)

    model_results = {}
    for prob_col in ["elo_prob_x", "xgb_win_prob", "lr_win_prob", "ensemble_prob"]:
        probs = results[prob_col].values
        preds_binary = (probs > 0.5).astype(int)

        acc = accuracy_score(y_true, preds_binary)
        brier = brier_score_loss(y_true, probs)
        ll = log_loss(y_true, probs)

        name = prob_col.replace("_x", "").replace("_prob", "").replace("_win", "")
        model_results[name] = {"accuracy": acc, "brier": brier, "log_loss": ll}

        print(f"\n{name.upper()}:")
        print(f"  Accuracy:    {acc:.1%} ({int(acc * len(y_true))}/{len(y_true)} correct)")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Log Loss:    {ll:.4f}")

    # Weekly breakdown
    print("\n" + "=" * 70)
    print("WEEKLY ACCURACY BREAKDOWN (Ensemble Model)")
    print("=" * 70)

    weekly_stats = []
    for week in sorted(results["week"].unique()):
        week_data = results[results["week"] == week]
        week_true = week_data["home_win"].values
        week_preds = (week_data["ensemble_prob"].values > 0.5).astype(int)

        correct = (week_true == week_preds).sum()
        total = len(week_true)
        acc = correct / total

        weekly_stats.append({
            "week": week,
            "games": total,
            "correct": correct,
            "accuracy": f"{acc:.0%}"
        })

    weekly_df = pd.DataFrame(weekly_stats)
    print(weekly_df.to_string(index=False))

    total_correct = sum(w["correct"] for w in weekly_stats)
    total_games = sum(w["games"] for w in weekly_stats)
    print(f"\nSEASON TOTAL: {total_correct}/{total_games} = {total_correct/total_games:.1%}")

    # Save results
    results.to_parquet(DATA_2025 / "results_2025_evaluated.parquet", index=False)
    
    return results, weekly_df


def week15_deep_dive(results: pd.DataFrame):
    """Deep dive into Week 15 predictions."""
    print("\n" + "=" * 70)
    print("WEEK 15 DEEP DIVE")
    print("=" * 70)

    week15 = results[results["week"] == 15].copy()
    print(f"\nTotal Week 15 Games: {len(week15)}")

    # Add prediction analysis
    week15["predicted_home_win"] = week15["ensemble_prob"] > 0.5
    week15["actual_home_win"] = week15["home_win"] == 1
    week15["correct"] = week15["predicted_home_win"] == week15["actual_home_win"]

    print("\n" + "-" * 70)
    print("GAME-BY-GAME BREAKDOWN")
    print("-" * 70)

    for _, game in week15.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        home_score = int(game["home_score"])
        away_score = int(game["away_score"])
        prob = game["ensemble_prob"]
        correct = "✅" if game["correct"] else "❌"
        
        predicted_winner = home if prob > 0.5 else away
        actual_winner = home if home_score > away_score else away
        
        print(f"\n{correct} {away} @ {home}")
        print(f"   Score: {away} {away_score} - {home} {home_score}")
        print(f"   Model: {home} {prob:.1%} | {away} {1-prob:.1%}")
        print(f"   Spread: {game['spread_line']}")
        print(f"   Predicted: {predicted_winner} | Actual: {actual_winner}")

    # Summary
    correct_count = week15["correct"].sum()
    total = len(week15)
    print(f"\n" + "-" * 70)
    print(f"WEEK 15 SUMMARY: {correct_count}/{total} correct ({correct_count/total:.0%})")
    print("-" * 70)

    return week15


if __name__ == "__main__":
    results, weekly = main()
    week15 = week15_deep_dive(results)

