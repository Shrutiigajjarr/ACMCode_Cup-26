from PIL.features import features
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="TactixAI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F7F6F2;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #111111;
    }
    h1, h2, h3, h4 {
        color: #111111 !important;
        font-weight: 800 !important;
    }
    .tx-card {
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .tx-metric-box {
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .tx-metric-label {
        font-size: 14px;
        font-weight: 700;
        color: #111111;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tx-metric-value {
        font-size: 30px;
        font-weight: 900;
        color: #111111;
        margin: 4px 0 8px 0;
    }
    .tx-progress-outer {
        background-color: #E6E4DD;
        border: 1.5px solid #111111;
        border-radius: 6px;
        height: 14px;
        width: 100%;
        overflow: hidden;
    }
    .tx-progress-inner-win { background-color: #1D9A50; height: 100%; }
    .tx-progress-inner-draw { background-color: #C9A227; height: 100%; }
    .tx-progress-inner-loss { background-color: #C0392B; height: 100%; }
    .tx-pitch {
        background: repeating-linear-gradient(
            to bottom, #2E8B4F, #2E8B4F 40px, #34995A 40px, #34995A 80px
        );
        border: 3px solid #111111;
        border-radius: 12px;
        position: relative;
        height: 660px;
        width: 100%;
        margin-top: 8px;
    }
    .tx-pitch-line {
        position: absolute;
        border: 2px solid rgba(255,255,255,0.85);
    }
    .tx-player-box {
        position: absolute;
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 8px;
        width: 118px;
        padding: 5px 6px;
        text-align: center;
        transform: translate(-50%, -50%);
        box-shadow: 2px 2px 0px rgba(0,0,0,0.35);
    }
    .tx-player-name { font-size: 12px; font-weight: 800; color: #111111; line-height: 1.1; }
    .tx-player-pos { font-size: 10px; font-weight: 700; color: #555555; }
    .tx-player-stat { font-size: 11px; font-weight: 700; color: #1D6FA5; margin-top: 2px; }
    .tx-log-item {
        border-left: 4px solid #111111;
        padding: 8px 12px;
        margin-bottom: 10px;
        background-color: #F1F0EA;
        border-radius: 4px;
        font-size: 14px;
        color: #111111;
    }
    .tx-bracket-node {
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin: 6px;
        min-width: 150px;
    }
    .tx-bracket-node-highlight {
        border: 3px solid #1D6FA5;
        box-shadow: 0 0 0 3px rgba(29,111,165,0.15);
    }
    .tx-bracket-node-critical {
        border: 3px solid #C0392B;
        box-shadow: 0 0 0 3px rgba(192,57,43,0.18);
        background-color: #FDF1F0;
    }
    .tx-bracket-round-label {
        font-weight: 800;
        font-size: 13px;
        text-align: center;
        color: #111111;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .tx-warning-icon { color: #C0392B; font-weight: 900; }
    .tx-narrative-box {
        background-color: #FFFFFF;
        border: 2px solid #111111;
        border-radius: 10px;
        padding: 18px;
        font-size: 15px;
        line-height: 1.6;
        color: #111111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA DIRECTORIES & DATA LOADERS (DYNAMIC)
# ============================================================================
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PLAYERS_FILE = "players.csv"
RESULTS_FILE = "results.csv"

def resolve_data_file(filename):
    return os.path.join(APP_ROOT, filename)

@st.cache_data
def load_player_database():
    """Reads players CSV file and groups them dynamically by country"""
    players_path = resolve_data_file(PLAYERS_FILE)
    if os.path.exists(players_path):
        df = pd.read_csv(players_path)
        db = {}
        for country, group in df.groupby("country"):
            db[country] = group.drop(columns=["country"]).to_dict(orient="records")
        return db
    else:
        st.error(f"⚠️ Crucial Data Missing: Please check '{PLAYERS_FILE}'")
        return {}

@st.cache_data
def load_prepared_kaggle_history():
    """Loads and formats the historical results file from Kaggle"""
    results_path = resolve_data_file(RESULTS_FILE)
    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            if df.empty:
                return pd.DataFrame()
            df['date'] = pd.to_datetime(df['date'])
            df['outcome'] = np.where(df['home_score'] > df['away_score'], 'Win',
                                     np.where(df['home_score'] == df['away_score'], 'Draw', 'Loss'))
            return df.sort_values('date')
        except Exception:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# Initialize dynamic operational matrices
PLAYER_DB = load_player_database()
TEAMS = list(PLAYER_DB.keys()) if PLAYER_DB else []
KAG_DF = load_prepared_kaggle_history()

# ============================================================================
# PREDICTOR ENGINE (FEATURE ASSEMBLY UNIT)
# ============================================================================
def team_overall(team_name):
    if team_name not in PLAYER_DB:
        return 75.0
    roster = PLAYER_DB[team_name]
    arr = np.array([[p["pace"], p["def"], p["ref"], p["pas"]] for p in roster])
    return float(arr.mean())

def compute_historical_team_metrics(df, team, current_date=None, window=10):
    if df.empty:
        return [1.2, 1.1, 0.45, 14]
        
    if current_date is not None and len(df) >= 10:
        past_games = df[((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < current_date)]
    else:
        past_games = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
    recent_games = past_games.tail(window)
    if len(recent_games) == 0:
        return [1.2, 1.1, 0.45, 14]
        
    goals_scored, goals_conceded, points = [], [], []
    for _, row in recent_games.iterrows():
        is_home = row['home_team'] == team
        scored = row['home_score'] if is_home else row['away_score']
        conceded = row['away_score'] if is_home else row['home_score']
        goals_scored.append(scored)
        goals_conceded.append(conceded)
        
        if row['outcome'] == 'Draw':
            points.append(1)
        elif (row['outcome'] == 'Win' and is_home) or (row['outcome'] == 'Loss' and not is_home):
            points.append(3)
        else:
            points.append(0)
            
    return [float(np.mean(goals_scored)), float(np.mean(goals_conceded)), float(points.count(3)/len(points)), int(np.sum(points))]

@st.cache_data
def build_model_training_set():
    if KAG_DF.empty:
        X_fake = np.random.uniform(55, 90, (100, 9))
        y_fake = np.random.choice(["Win", "Draw", "Loss"], 100)
        return X_fake, y_fake

    features, targets = [], []
    # Train using up to the latest 10,000 historical matches
    training_sample = KAG_DF.tail(min(10000, len(KAG_DF)))
    
    for _, row in training_sample.iterrows():
        h_team, a_team, m_date = row['home_team'], row['away_team'], row['date']
        h_metrics = compute_historical_team_metrics(KAG_DF, h_team, m_date)
        a_metrics = compute_historical_team_metrics(KAG_DF, a_team, m_date)
        neutral_flag = [1 if row['neutral'] else 0]
        
        home_overall = team_overall(h_team)
        away_overall = team_overall(a_team)

        feature_vector = (h_metrics + a_metrics + [home_overall, away_overall] + neutral_flag
    )

        features.append(feature_vector)
        targets.append(row['outcome'])
        
    return np.array(features), np.array(targets)

@st.cache_resource
def get_trained_model():
    X, y = build_model_training_set()
    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

PREDICTOR_MODEL = get_trained_model()

def predict_match_outcome(home_team, away_team):
    # Calculate overalls to act as a dynamic fallback
    h_overall = team_overall(home_team)
    a_overall = team_overall(away_team)
    
    if not KAG_DF.empty and len(KAG_DF) >= 2:
        try:
            home_metrics = compute_historical_team_metrics(KAG_DF, home_team)
            away_metrics = compute_historical_team_metrics(KAG_DF, away_team)
            neutral_flag = [0]
            home_overall = team_overall(home_team)
            away_overall = team_overall(away_team)

            live_match_vector = np.array([home_metrics + away_metrics + [home_overall, away_overall] + neutral_flag]
            )
            
            model_probs = PREDICTOR_MODEL.predict_proba(live_match_vector)[0]
            classes = list(PREDICTOR_MODEL.classes_)
            
            win_pct = round(float(model_probs[classes.index("Win")]) * 100)
            draw_pct = round(float(model_probs[classes.index("Draw")]) * 100)
            loss_pct = 100 - win_pct - draw_pct
        except Exception:
            # Dynamic fallback based on player statistics if model breaks
            total = h_overall + a_overall
            win_pct = round((h_overall / total) * 100)
            draw_pct = 20
            loss_pct = 100 - win_pct - draw_pct
    else:
        # Dynamic fallback based on player statistics
        total = h_overall + a_overall
        win_pct = round((h_overall / total) * 100)
        draw_pct = 20
        loss_pct = 100 - win_pct - draw_pct

    return {
        "win": win_pct, "draw": draw_pct, "loss": loss_pct,
        "home_overall": round(h_overall, 1),
        "away_overall": round(a_overall, 1),
    }

# ============================================================================
# STYLE OPTIMIZER ENGINE (11-PLAYER GENERATOR)
# ============================================================================
def opponent_winger_pace(opponent_team):
    if opponent_team not in PLAYER_DB: return 70.0
    roster = PLAYER_DB[opponent_team]
    wingers = [p for p in roster if p["pos"] in ("LW", "RW")]
    if not wingers: return 70.0
    return float(np.mean([p["pace"] for p in wingers]))

def fullback_score(player):
    return player["pace"] * 0.6 + player["def"] * 0.4

POSITION_GROUPS = {
    "GK": ("GK",),
    "LB": ("LB", "RB", "CB"),
    "RB": ("RB", "LB", "CB"),
    "CB": ("CB", "LB", "RB"),
    "CDM": ("CDM", "CM", "CAM"),
    "CM": ("CM", "CDM", "CAM"),
    "CAM": ("CAM", "CM", "CDM"),
    "LW": ("LW", "RW", "ST"),
    "RW": ("RW", "LW", "ST"),
    "ST": ("ST", "LW", "RW"),
}

def pick_best_by(roster, pos, key, exclude=None):
    exclude = exclude or []
    candidates = [p for p in roster if p["pos"] == pos and p["name"] not in exclude]
    if not candidates:
        group_positions = POSITION_GROUPS.get(pos, (pos,))
        candidates = [p for p in roster if p["pos"] in group_positions and p["name"] not in exclude]
    if not candidates:
        candidates = [p for p in roster if p["name"] not in exclude]
    if not candidates: return {"name": f"Unknown {pos}", "pos": pos, "pace": 60, "def": 60, "ref": 60, "pas": 60}
    return sorted(candidates, key=key, reverse=True)[0]

def build_starting_xi(home_team, away_team):
    if home_team not in PLAYER_DB: 
        return {}, False, 0.0, None, None

    roster = PLAYER_DB[home_team]
    threat_pace = opponent_winger_pace(away_team)
    high_threat = threat_pace >= 85
    
    assigned = []

    # 1. Goalkeeper
    gk = pick_best_by(roster, "GK", key=lambda p: p["ref"])
    assigned.append(gk["name"])

    # 2. Defending Line (CB1, CB2, RB, LB)
    cb1 = pick_best_by(roster, "CB", key=lambda p: p["def"], exclude=assigned)
    assigned.append(cb1["name"])
    cb2 = pick_best_by(roster, "CB", key=lambda p: p["def"], exclude=assigned)
    assigned.append(cb2["name"])

    if high_threat:
        rb = pick_best_by(roster, "RB", key=fullback_score, exclude=assigned)
        assigned.append(rb["name"])
        lb = pick_best_by(roster, "LB", key=fullback_score, exclude=assigned)
        assigned.append(lb["name"])
    else:
        rb = pick_best_by(roster, "RB", key=lambda p: p["def"], exclude=assigned)
        assigned.append(rb["name"])
        lb = pick_best_by(roster, "LB", key=lambda p: p["def"], exclude=assigned)
        assigned.append(lb["name"])

    # 3. Midfield Core (CDM, CM, CAM)
    cdm = pick_best_by(roster, "CDM", key=lambda p: p["def"] * 0.5 + p["pas"] * 0.5, exclude=assigned)
    assigned.append(cdm["name"])
    cm = pick_best_by(roster, "CM", key=lambda p: p["pas"], exclude=assigned)
    assigned.append(cm["name"])
    cam = pick_best_by(roster, "CAM", key=lambda p: p["pas"], exclude=assigned)
    assigned.append(cam["name"])

    # 4. Attacking Frontline (RW, LW, ST)
    rw = pick_best_by(roster, "RW", key=lambda p: p["pace"], exclude=assigned)
    assigned.append(rw["name"])
    lw = pick_best_by(roster, "LW", key=lambda p: p["pace"], exclude=assigned)
    assigned.append(lw["name"])
    st_player = pick_best_by(roster, "ST", key=lambda p: p["pace"] * 0.5 + p["pas"] * 0.5, exclude=assigned)

    xi = {
        "GK": gk, "RB": rb, "CB1": cb1, "CB2": cb2, "LB": lb,
        "CDM": cdm, "CM": cm, "CAM": cam, "RW": rw, "LW": lw, "ST": st_player,
    }
    return xi, high_threat, round(threat_pace, 1), rb, lb

# ============================================================================
# PITCH COORDINATES (EXACT 11 DISTINCT PLOTS)
# ============================================================================
PITCH_COORDS = {
    "GK": (50, 90),
    "RB": (15, 74), "CB1": (38, 76), "CB2": (62, 76), "LB": (85, 74),
    "CDM": (50, 58), "CM": (30, 46), "CAM": (70, 46),
    "RW": (18, 20), "LW": (82, 20), "ST": (50, 10),
}
PITCH_STAT_LABEL = {
    "GK": "REF", "RB": "PAC", "CB1": "DEF", "CB2": "DEF", "LB": "PAC",
    "CDM": "DEF", "CM": "PAS", "CAM": "PAS", "RW": "PAC", "LW": "PAC", "ST": "PAC",
}
PITCH_STAT_KEY = {
    "GK": "ref", "RB": "pace", "CB1": "def", "CB2": "def", "LB": "pace",
    "CDM": "def", "CM": "pas", "CAM": "pas", "RW": "pace", "LW": "pace", "ST": "pace",
}

def render_pitch_html(xi):
    boxes = []
    for slot, player in xi.items():
        if player is None: continue
        x, y = PITCH_COORDS[slot]
        stat_key = PITCH_STAT_KEY[slot]
        stat_label = PITCH_STAT_LABEL[slot]
        boxes.append(
            f'<div class="tx-player-box" style="left:{x}%; top:{y}%;">'
            f'<div class="tx-player-name">{player["name"]}</div>'
            f'<div class="tx-player-pos">{slot}</div>'
            f'<div class="tx-player-stat">{stat_label} {player[stat_key]}</div>'
            f'</div>'
        )
    lines = (
        '<div class="tx-pitch-line" style="left:5%; top:50%; width:90%; height:0;"></div>'
        '<div class="tx-pitch-line" style="left:50%; top:5%; width:0; height:90%;"></div>'
    )
    return f'<div class="tx-pitch">{lines}{"".join(boxes)}</div>'

# ============================================================================
# ANALYSIS LOG NARRATIVE
# ============================================================================
def build_analysis_log(home_team, away_team, high_threat, threat_pace, rb, lb, pred):
    log_items = [
        f"Formation set to a 4-3-3 base shape for {home_team}, balancing defensive solidity.",
    ]
    if high_threat:
        log_items.append(
            f"{away_team}'s wide attackers carry an average pace rating of {threat_pace}, "
            f"identified as a high-velocity wing threat. Fullbacks were weighted "
            f"toward Pace to avoid tracking errors against vertical runs."
        )
    else:
        log_items.append(
            f"{away_team} does not present an elevated wide-pace threat ({threat_pace}), "
            f"so fullback selection was weighted toward raw defensive solidity."
        )
    log_items.append(
        f"Predictive engine outputs {pred['win']}% Win / {pred['draw']}% Draw / {pred['loss']}% Loss "
        f"for {home_team} based on historical dataset baselines."
    )
    return log_items

# ============================================================================
# PATHWAY SIMULATION NARRATIVE
# ============================================================================
def build_pathway_narrative(team, pathway):
    critical_fixture = next((f for f in pathway if f["critical"]), None)
    narrative = f"{team}'s predicted tournament pathway sees an optimized passage through early rounds against {pathway[0]['opponent']}. "
    if critical_fixture:
        narrative += f"However, the model flags the {critical_fixture['round']} match against {critical_fixture['opponent']} as a [CRITICAL BOTTLENECK] with a win probability of {critical_fixture['win_pct']}%."
    return narrative

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## ⚽ TactixAI")
view = st.sidebar.radio(
    "Select Module",
    ["Tactical Support Assistant", "Pathway Simulation & Bottleneck Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

# ============================================================================
# MODULE 1: TACTICAL SUPPORT ASSISTANT
# ============================================================================
if view == "Tactical Support Assistant":
    st.sidebar.markdown("### Match Setup")
    tournament = st.sidebar.selectbox("Select Tournament", ["ACM Code Cup 2026"])
    
    home_team = st.sidebar.selectbox("Choose Home Team", TEAMS if TEAMS else ["No teams loaded"], index=0)
    away_options = [t for t in TEAMS if t != home_team]
    away_team = st.sidebar.selectbox("Choose Away Team", away_options if away_options else ["No Opponents Available"], index=0)
    
    generate = st.sidebar.button("Generate Tactics & Prediction", type="primary", use_container_width=True)

    if "tx_home" not in st.session_state:
        st.session_state.tx_home = TEAMS[0] if TEAMS else ""
        st.session_state.tx_away = TEAMS[1] if len(TEAMS) > 1 else (TEAMS[0] if TEAMS else "")

    if generate and away_team != "No Opponents Available":
        st.session_state.tx_home = home_team
        st.session_state.tx_away = away_team

    active_home = st.session_state.tx_home
    active_away = st.session_state.tx_away

    st.title("TactixAI: Tactical Support Assistant")
    st.caption(f"{tournament} — {active_home} (Home) vs {active_away} (Away)")

    if active_home in PLAYER_DB and active_away in PLAYER_DB:
        pred = predict_match_outcome(active_home, active_away)
        
        m1, m2, m3 = st.columns(3)
        for col, label, val, cls in zip([m1, m2, m3], ["Win", "Draw", "Loss"], [pred['win'], pred['draw'], pred['loss']], ["win", "draw", "loss"]):
            with col:
                st.markdown(
                    f'<div class="tx-metric-box"><div class="tx-metric-label">{label}</div>'
                    f'<div class="tx-metric-value">{val}%</div>'
                    f'<div class="tx-progress-outer"><div class="tx-progress-inner-{cls}" style="width:{val}%;"></div></div></div>',
                    unsafe_allow_html=True
                )

        st.write("")
        col_pitch, col_log = st.columns([1.3, 1])
        xi, high_threat, threat_pace, rb, lb = build_starting_xi(active_home, active_away)

        with col_pitch:
            st.markdown("#### Lineup Optimizer — Pitch View")
            st.markdown(render_pitch_html(xi), unsafe_allow_html=True)

        with col_log:
            st.markdown("#### TactixAI Analysis Log")
            log_items = build_analysis_log(active_home, active_away, high_threat, threat_pace, rb, lb, pred)
            log_html = "".join([f'<div class="tx-log-item">{item}</div>' for item in log_items])
            st.markdown(f'<div class="tx-card">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Please populate data profiles in '{PLAYERS_FILE}' to unlock tactical simulations.")

# ============================================================================
# MODULE 2: PATHWAY SIMULATION & BOTTLENECK ANALYSIS
# ============================================================================
else:
    st.sidebar.markdown("### Simulation Setup")
    focus_team = st.sidebar.selectbox("Choose Focus Team", TEAMS, index=0 if len(TEAMS)>0 else 0)
    run_sim = st.sidebar.button("Run Pathway Simulation", type="primary", use_container_width=True)

    if "tx_focus" not in st.session_state:
        st.session_state.tx_focus = focus_team if focus_team else "Brazil"
    if run_sim and focus_team:
        st.session_state.tx_focus = focus_team

    active_focus = st.session_state.tx_focus

    st.title("TactixAI: Pathway Simulation & Bottleneck Analysis")
    
    if active_focus in TEAMS:
        st.caption(f"Predicted Dynamic Tournament Journey for {active_focus}")
        
        # Build dynamic rounds using other teams in your dropdown list
        opponents = [t for t in TEAMS if t != active_focus]
        if not opponents:
            opponents = ["Uruguay", "Croatia", "Germany", "France"]
        
        rounds = ["Round of 16", "Quarter-Final", "Semi-Final", "Final"]
        pathway = []
        
        for i, r in enumerate(rounds):
            opp = opponents[i % len(opponents)]
            pred = predict_match_outcome(active_focus, opp)
            # Set critical bottleneck flag dynamically only if win prob is less than 50%
            is_critical = pred["win"] < 50
            pathway.append({
                "round": r, "opponent": opp, "win_pct": pred["win"], "critical": is_critical
            })

        cols = st.columns(len(pathway))
        for col, fixture in zip(cols, pathway):
            with col:
                css_class = "tx-bracket-node tx-bracket-node-critical" if fixture["critical"] else "tx-bracket-node tx-bracket-node-highlight"
                warning_html = '<div class="tx-warning-icon">&#9888; [CRITICAL BOTTLENECK]</div>' if fixture["critical"] else ""
                
                st.markdown(
                    f'<div class="tx-bracket-round-label">{fixture["round"]}</div>'
                    f'<div class="{css_class}"><div style="font-weight:700; font-size:13px;">{active_focus} vs {fixture["opponent"]}</div>'
                    f'{warning_html}<div style="font-weight:800; margin-top:6px;">Win {fixture["win_pct"]}%</div></div>',
                    unsafe_allow_html=True
                )

        st.write("")
        st.markdown("#### Pathway Narrative")
        st.markdown(f'<div class="tx-narrative-box">{build_pathway_narrative(active_focus, pathway)}</div>', unsafe_allow_html=True)