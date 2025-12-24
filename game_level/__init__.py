"""
Game-Level Feature Derivation Module

This module contains functions for deriving ESPN features at the game level
(one row per team-game) rather than season level (one row per team-season).

Key differences from season-level derivation:
- Filters play-by-play data to a single game_id
- Derives features for a team's performance in that specific game
- Enables game-by-game predictions for moneyline betting
"""

__version__ = "1.0.0"
__author__ = "NFL Betting Model Team"

