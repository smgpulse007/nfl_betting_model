"""Team abbreviation mapping between ESPN and nfl-data-py"""

# ESPN uses different abbreviations than nfl-data-py for some teams
ESPN_TO_NFL_DATA_PY = {
    'LAR': 'LA',   # Los Angeles Rams
    'WSH': 'WAS',  # Washington Commanders
}

NFL_DATA_PY_TO_ESPN = {
    'LA': 'LAR',
    'WAS': 'WSH',
}

def espn_to_nfl_data_py(team: str) -> str:
    """Convert ESPN team abbreviation to nfl-data-py abbreviation"""
    return ESPN_TO_NFL_DATA_PY.get(team, team)

def nfl_data_py_to_espn(team: str) -> str:
    """Convert nfl-data-py team abbreviation to ESPN abbreviation"""
    return NFL_DATA_PY_TO_ESPN.get(team, team)

