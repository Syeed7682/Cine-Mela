import streamlit as st
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub
import os
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="Cine-Mela · Netflix Recommender",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --red:       #0984e3;
  --red-glow:  rgba(9,132,227,0.35);
  --red-dim:   rgba(9,132,227,0.12);
  --gold:      #74b9ff;
  --green:     #00b894;
  --bg:        #1b1e1f;
  --surface:   #2d3436;
  --surface2:  #3b4446;
  --border:    rgba(178,190,195,0.15);
  --text:      #E8E8F0;
  --muted:     #b2bec3;
  --card-h:    240px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg);
  color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1400px; }
.stApp { background: var(--bg); }

/* ── Hero Banner ── */
.hero-banner {
  position: relative;
  background: linear-gradient(135deg, #1b1e1f 0%, #2d3436 50%, #1b1e1f 100%);
  border-bottom: 1px solid var(--border);
  padding: 28px 40px 22px;
  margin: 0 -2rem 2.5rem;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 200% at 80% 50%, rgba(9,132,227,0.08) 0%, transparent 65%),
    radial-gradient(ellipse 40% 150% at 5% 50%, rgba(9,132,227,0.05) 0%, transparent 60%);
}
.hero-banner::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--red), transparent);
  opacity: 0.6;
}
.hero-inner { position: relative; display: flex; align-items: center; gap: 18px; }
.logo-mark {
  width: 52px; height: 52px;
  background: var(--red);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-size: 1.4rem; font-weight: 800; color: #fff;
  box-shadow: 0 0 24px var(--red-glow);
  flex-shrink: 0;
  letter-spacing: -1px;
}
.hero-titles h1 {
  font-family: 'Syne', sans-serif;
  font-size: 1.7rem;
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.5px;
  margin: 0;
  line-height: 1;
}
.hero-titles p {
  font-size: 0.82rem;
  color: var(--muted);
  margin: 4px 0 0;
  font-weight: 300;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.hero-badge {
  margin-left: auto;
  background: var(--red-dim);
  border: 1px solid rgba(9,132,227,0.3);
  border-radius: 20px;
  padding: 5px 14px;
  font-size: 0.75rem;
  color: #74b9ff;
  font-weight: 500;
  letter-spacing: 0.04em;
  display: flex; align-items: center; gap: 6px;
}
.live-dot {
  width: 6px; height: 6px;
  background: var(--red);
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:0.4; transform:scale(1.4); }
}

/* ── Section Titles ── */
.section-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--red);
  margin-bottom: 4px;
}
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: #fff;
  margin: 0 0 1.4rem;
  display: flex; align-items: center; gap: 10px;
}
.section-title span.icon {
  font-size: 1.1rem;
}

/* ── Stat Cards ── */
.stat-row { display:flex; gap:14px; margin-bottom:2rem; flex-wrap:wrap; }
.stat-card {
  flex: 1; min-width: 140px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  position: relative;
  overflow: hidden;
}
.stat-card::before {
  content:'';
  position:absolute;
  top:0;left:0;right:0;height:2px;
  background: linear-gradient(90deg, var(--red), transparent);
}
.stat-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.8rem;
  font-weight: 800;
  color: #fff;
  line-height: 1;
}
.stat-lbl {
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.stat-delta {
  position: absolute;
  top: 16px; right: 16px;
  font-size: 0.75rem;
  color: var(--green);
  font-weight: 600;
}

/* ── Movie Card ── */
.card-grid { display:flex; gap:14px; flex-wrap:wrap; }
.movie-card {
  flex: 1; min-width: 180px; max-width: 240px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
  cursor: default;
}
.movie-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(9,132,227,0.25);
  border-color: rgba(9,132,227,0.3);
}
.movie-card.hybrid {
  background: linear-gradient(160deg, #2d3436 0%, #1b1e1f 100%);
  border-color: rgba(9,132,227,0.2);
}
.movie-card.hybrid::after {
  content: '🌟 TOP PICK';
  position: absolute;
  top: 10px; right: -24px;
  background: var(--red);
  color: #fff;
  font-size: 0.6rem;
  font-weight: 700;
  padding: 3px 28px;
  transform: rotate(35deg);
  letter-spacing: 0.08em;
}
.card-type-badge {
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  padding: 2px 7px;
  border-radius: 4px;
  margin-bottom: 8px;
}
.badge-movie  { background: rgba(245,197,24,0.12); color: var(--gold); border: 1px solid rgba(245,197,24,0.25); }
.badge-tv     { background: rgba(70,211,105,0.10); color: var(--green); border: 1px solid rgba(70,211,105,0.25); }
.card-title { font-family:'Syne',sans-serif; font-size:0.98rem; font-weight:700; color:#fff; line-height:1.25; margin-bottom:8px; }
.card-meta  { font-size:0.75rem; color:var(--muted); margin-bottom:3px; display:flex; align-items:center; gap:5px; }
.card-score {
  margin-top: 12px;
  font-size: 0.8rem; font-weight: 700;
  padding: 5px 10px; border-radius: 6px;
  display: inline-flex; align-items: center; gap: 5px;
}
.score-hybrid { background:rgba(9,132,227,0.12); color:#74b9ff; border:1px solid rgba(9,132,227,0.2); }
.score-kb     { background:rgba(70,211,105,0.10); color:var(--green); border:1px solid rgba(70,211,105,0.2); }
.score-rl     { background:rgba(245,197,24,0.10); color:var(--gold);  border:1px solid rgba(245,197,24,0.2); }
.score-watch  { background:rgba(255,255,255,0.06); color:#ccc; border:1px solid var(--border); }

/* ── Controls Panel ── */
.controls-panel {
  padding: 10px 0;
  margin-bottom: 2rem;
}

/* ── Divider ── */
.fancy-divider {
  display: flex; align-items: center; gap: 16px;
  margin: 2.5rem 0;
  color: var(--muted);
  font-size: 0.7rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.fancy-divider::before, .fancy-divider::after {
  content:''; flex:1; height:1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ── Info Box ── */
.info-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--red);
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

/* ── Streamlit widget polish ── */
.stSelectbox label, .stMultiSelect label, .stSlider label, .stTextInput label {
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
div[data-baseweb="select"] > div {
  background-color: rgba(255, 255, 255, 0.9) !important;
  border: 1px solid rgba(255, 255, 255, 0.9) !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] * {
  color: #1b1e1f !important;
}
.stTextInput > div > div > input {
  background-color: rgba(255, 255, 255, 0.9) !important;
  border: 1px solid rgba(255, 255, 255, 0.9) !important;
  border-radius: 10px !important;
  color: #1b1e1f !important;
}
.stTextInput > div > div > input::placeholder {
  color: #7A7A8C !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--red) !important;
  box-shadow: 0 0 0 2px var(--red-dim) !important;
  background-color: #ffffff !important;
}
div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
}
.stSpinner > div { border-top-color: var(--red) !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: var(--muted) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: var(--red) !important;
  color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Data Loading ───────────────────────────
@st.cache_data
def load_data():
    try:
        path = kagglehub.dataset_download('shivamb/netflix-shows')
        csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
        netflix_df = pd.read_csv(os.path.join(path, csv_file))
        if 'release_year' not in netflix_df.columns:
            netflix_df['release_year'] = 2020
    except Exception:
        np.random.seed(42)
        n_movies = 300
        movie_titles = ['Inception','The Dark Knight','Interstellar','Avatar','Titanic','The Matrix',
                        'Pulp Fiction','Forrest Gump','The Shawshank Redemption','The Godfather',
                        'Fight Club','Gladiator','The Avengers','Iron Man','Captain America','Thor',
                        'Black Panther','Spider-Man','The Lion King','Frozen','Toy Story',
                        'Monsters Inc','Finding Nemo','The Incredibles','Oppenheimer','Dune',
                        'Parasite','Joker','1917','Knives Out']
        genres_pool = ['Drama','Comedy','Action','Romance','Thriller','Sci-Fi','Horror',
                       'Documentary','Animation','Crime','Mystery','Adventure']
        netflix_df = pd.DataFrame({
            'show_id': [f's{i}' for i in range(n_movies)],
            'title': [f'{movie_titles[i % len(movie_titles)]} {i//len(movie_titles) or ""}' .strip() for i in range(n_movies)],
            'type': np.random.choice(['Movie','TV Show'], n_movies, p=[0.65,0.35]),
            'rating': np.random.choice(['G','PG','PG-13','R','TV-MA','TV-14'], n_movies),
            'listed_in': [', '.join(np.random.choice(genres_pool, np.random.randint(1,3), replace=False)) for _ in range(n_movies)],
            'release_year': np.random.randint(1990, 2024, n_movies),
            'duration': [f'{np.random.randint(70,180)} min' if t=='Movie' else f'{np.random.randint(1,8)} Seasons'
                         for t in np.random.choice(['Movie','TV Show'], n_movies, p=[0.65,0.35])],
            'description': ['An exciting journey unfolds.' for _ in range(n_movies)]
        })

    np.random.seed(42)
    n_users, n_ratings = 200, 3000
    ratings_data = []
    for _ in range(n_ratings):
        user_id   = np.random.randint(1, n_users+1)
        movie_idx = np.random.randint(0, len(netflix_df))
        movie_id  = netflix_df.iloc[movie_idx]['show_id']
        rating    = np.random.choice([1,2,3,4,5], p=[0.05,0.1,0.2,0.35,0.3])
        ratings_data.append({'userId': user_id, 'movieId': movie_id, 'rating': rating})

    ratings_df = pd.DataFrame(ratings_data).drop_duplicates(subset=['userId','movieId'], keep='last')
    combined_df = pd.merge(ratings_df, netflix_df, left_on='movieId', right_on='show_id', how='inner')

    if 'rating_x' in combined_df.columns:
        combined_df = combined_df.rename(columns={'rating_x':'rating_score','rating_y':'content_rating'})
    elif 'rating' in combined_df.columns:
        combined_df = combined_df.rename(columns={'rating':'rating_score'})

    combined_df['rating_score'] = pd.to_numeric(combined_df['rating_score'], errors='coerce')
    combined_df = combined_df.dropna(subset=['userId','movieId','rating_score'])

    user_counts  = combined_df['userId'].value_counts()
    active_users = user_counts[user_counts >= 3].index
    combined_df  = combined_df[combined_df['userId'].isin(active_users)]

    movieId_dict = {m: i for i, m in enumerate(combined_df['movieId'].unique())}
    combined_df['movieId_key'] = combined_df['movieId'].map(movieId_dict)

    user_movie_matrix = sp.coo_matrix(
        (combined_df['rating_score'], (combined_df['userId'], combined_df['movieId_key']))
    ).tocsr()

    return netflix_df, combined_df, user_movie_matrix, movieId_dict


# ─────────────────────────── RL Agent ───────────────────────────
class QLearningAgent:
    def __init__(self, n_movies, lr=0.1, gamma=0.9, epsilon=0.15):
        self.n_movies, self.lr, self.gamma, self.epsilon = n_movies, lr, gamma, epsilon
        self.q_table = np.random.rand(n_movies) * 0.01

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_movies)
        return np.argmax(self.q_table)

    def learn(self, action, reward):
        nr = (reward - 1) / 4
        self.q_table[action] += self.lr * (nr - self.q_table[action])

    def decay_epsilon(self, rate=0.995):
        self.epsilon = max(self.epsilon * rate, 0.01)

    def get_top_recommendations(self, n=10):
        return np.argsort(self.q_table)[::-1][:n]


@st.cache_resource(show_spinner=False)
def train_rl_agent(_combined_df, uid):
    user_watched = _combined_df[_combined_df['userId'] == uid]
    ratings_dict = dict(zip(user_watched['movieId'], user_watched['rating_score']))
    movies_list  = _combined_df['movieId'].unique().tolist()
    agent = QLearningAgent(len(movies_list))
    for _ in range(50):
        for _ in range(min(20, len(movies_list))):
            a = agent.choose_action()
            agent.learn(a, ratings_dict.get(movies_list[a], 0))
        agent.decay_epsilon(0.98)
    return agent, movies_list, ratings_dict


# ─────────────────────────── Collaborative Filtering ───────────────────────────
def collaborative_filtering(user_id, combined_df, user_movie_matrix, netflix_df, movieId_dict, n_recs=50):
    if user_id not in combined_df['userId'].unique():
        return pd.DataFrame(), []

    user_ratings = user_movie_matrix.getrow(user_id).toarray().ravel()
    sims = cosine_similarity(user_ratings.reshape(1,-1), user_movie_matrix.toarray()).ravel()
    sims[user_id] = -1
    similar_users = np.argsort(sims)[::-1][:10]
    user_watched  = combined_df[combined_df['userId']==user_id]['movieId'].unique()

    recommendations = {}
    for su in similar_users:
        if sims[su] <= 0: continue
        for _, row in combined_df[combined_df['userId']==su].iterrows():
            m = row['movieId']
            if m not in user_watched:
                if m not in recommendations: recommendations[m] = {'score':0,'count':0}
                recommendations[m]['score'] += row['rating_score'] * sims[su]
                recommendations[m]['count'] += 1

    rec_list = [{'movieId':k,'score':v['score']/v['count']} for k,v in recommendations.items()]
    rec_df = pd.DataFrame(rec_list).sort_values('score',ascending=False) if rec_list else pd.DataFrame()
    if not rec_df.empty:
        rec_df = pd.merge(rec_df, netflix_df[['show_id','title','type','listed_in','release_year']], 
                          left_on='movieId', right_on='show_id', how='left').head(n_recs)

    sim_recs = []
    for su in similar_users[:3]:
        for _, row in combined_df[combined_df['userId']==su].sort_values('rating_score',ascending=False).head(3).iterrows():
            m_info = netflix_df[netflix_df['show_id']==row['movieId']]
            if len(m_info) == 0: continue
            m_info = m_info.iloc[0]
            sim_recs.append({'similar_user':su,'title':m_info['title'],'type':m_info['type'],
                              'listed_in':m_info['listed_in'],'release_year':m_info.get('release_year','N/A'),
                              'rating_score':row['rating_score']})
    return rec_df, sim_recs


def get_movie_kb_score(uid, movie_id, combined_df, user_movie_matrix, movieId_dict):
    if uid not in combined_df['userId'].unique() or movie_id not in movieId_dict: return 0.0
    user_ratings = user_movie_matrix.getrow(uid).toarray().ravel()
    sims = cosine_similarity(user_ratings.reshape(1,-1), user_movie_matrix.toarray()).ravel()
    sims[uid] = -1
    similar_users = np.argsort(sims)[::-1][:10]
    score, count = 0, 0
    for su in similar_users:
        if sims[su] <= 0: continue
        r = combined_df[(combined_df['userId']==su) & (combined_df['movieId']==movie_id)]
        if not r.empty:
            score += r.iloc[0]['rating_score'] * sims[su]
            count += 1
    return score / count if count > 0 else 0.0


def make_card(title, type_, year, genre, score_html, extra_class=""):
    badge_class = "badge-tv" if type_ == "TV Show" else "badge-movie"
    return f"""
<div class="movie-card {extra_class}">
  <span class="card-type-badge {badge_class}">{type_}</span>
  <div class="card-title">{title}</div>
  <div class="card-meta">📅 {year}</div>
  <div class="card-meta">🎭 {genre[:40]}{'…' if len(str(genre))>40 else ''}</div>
  {score_html}
</div>"""


# ─────────────────────────── MAIN APP ───────────────────────────
with st.spinner("Initialising Cine-Mela Engine…"):
    netflix_df, combined_df, user_movie_matrix, movieId_dict = load_data()

# Genre list
all_genres = sorted({g.strip() for genres in netflix_df['listed_in'].dropna() for g in genres.split(',')})

# ── Hero Banner ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-inner">
    <div class="logo-mark">Ci</div>
    <div class="hero-titles">
      <h1>Cine-Mela</h1>
      <p>Hybrid · Collaborative · Reinforcement Learning Recommender</p>
    </div>
    <div class="hero-badge">
      <div class="live-dot"></div>
      LIVE ENGINE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls Panel ───────────────────────────────────────────────────
st.markdown('<div class="controls-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Dashboard Controls</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title"><span class="icon">⚙️</span> Personalise Your Feed</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 2])
all_users = sorted(combined_df['userId'].unique())

with c1:
    selected_user = st.selectbox("👤 User Profile", all_users)
    user_data = combined_df[combined_df['userId'] == selected_user]

with c2:
    selected_genres = st.multiselect("🎭 Genre Filter", all_genres, placeholder="All genres")

with c3:
    yr_min, yr_max = int(netflix_df['release_year'].min()), int(netflix_df['release_year'].max())
    min_year, max_year = st.slider("📅 Release Year", yr_min, yr_max, (1995, yr_max))

st.markdown('</div>', unsafe_allow_html=True)

# ── KPI Stats ───────────────────────────────────────────────────────
total_titles  = len(netflix_df)
total_movies  = len(netflix_df[netflix_df['type']=='Movie'])
total_tv      = len(netflix_df[netflix_df['type']=='TV Show'])
user_watched  = len(user_data)
user_avg_rat  = user_data['rating_score'].mean() if user_watched > 0 else 0
total_users   = combined_df['userId'].nunique()

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-val">{total_titles:,}</div>
    <div class="stat-lbl">Total Titles</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{total_movies:,}</div>
    <div class="stat-lbl">Movies</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{total_tv:,}</div>
    <div class="stat-lbl">TV Shows</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{total_users}</div>
    <div class="stat-lbl">Active Users</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{user_watched}</div>
    <div class="stat-lbl">User #{selected_user} Watched</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">{user_avg_rat:.1f}<span style="font-size:1rem;color:var(--muted)">/5</span></div>
    <div class="stat-lbl">Avg Rating</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 Analytics", "🔍 Search"])

# ════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════
with tab1:

    with st.spinner("🧠 Running Collaborative Filtering…"):
        kb_recs, similar_user_recs = collaborative_filtering(selected_user, combined_df, user_movie_matrix, netflix_df, movieId_dict, 50)

    with st.spinner("🤖 Training Q-Learning Agent…"):
        agent, movies_list, user_ratings_dict = train_rl_agent(combined_df, selected_user)
        top_actions = agent.get_top_recommendations(50)
        rl_recs = []
        for a in top_actions:
            m_id  = movies_list[a]
            q_val = agent.q_table[a]
            m_info = netflix_df[netflix_df['show_id']==m_id]
            if len(m_info):
                rl_recs.append({'movieId':m_id,'title':m_info.iloc[0]['title'],
                                'type':m_info.iloc[0]['type'],'listed_in':m_info.iloc[0]['listed_in'],
                                'release_year':m_info.iloc[0].get('release_year',2020),'q_value':q_val})
        rl_df = pd.DataFrame(rl_recs)

    def filter_recs(df):
        if df.empty: return df
        if selected_genres:
            df = df[df['listed_in'].str.contains('|'.join(selected_genres), case=False, na=False)]
        df = df[(df['release_year'] >= min_year) & (df['release_year'] <= max_year)]
        return df

    kb_recs_f = filter_recs(kb_recs)
    rl_df_f   = filter_recs(rl_df)

    # ── Hybrid Top Picks ──────────────────────────────────────
    if not kb_recs_f.empty and not rl_df_f.empty:
        kb_recs_f = kb_recs_f.copy(); kb_recs_f['kb_norm'] = kb_recs_f['score'] / (kb_recs_f['score'].max() or 1)
        rl_df_f   = rl_df_f.copy();   rl_df_f['rl_norm']   = rl_df_f['q_value']  / (rl_df_f['q_value'].max() + 1e-8)

        all_ids = set(kb_recs_f['movieId'].values) | set(rl_df_f['movieId'].values)
        hybrid_recs = []
        for m in all_ids:
            kb_s = kb_recs_f[kb_recs_f['movieId']==m]['kb_norm'].values
            rl_s = rl_df_f[rl_df_f['movieId']==m]['rl_norm'].values
            kb_s = kb_s[0] if len(kb_s) else 0
            rl_s = rl_s[0] if len(rl_s) else 0
            mi   = netflix_df[netflix_df['show_id']==m]
            if len(mi):
                hybrid_recs.append({'movieId':m,'title':mi.iloc[0]['title'],'type':mi.iloc[0]['type'],
                                    'listed_in':mi.iloc[0]['listed_in'],'release_year':mi.iloc[0].get('release_year','N/A'),
                                    'hybrid_score':0.4*kb_s+0.6*rl_s,'kb_score':kb_s,'rl_score':rl_s})
        hybrid_df = pd.DataFrame(hybrid_recs).sort_values('hybrid_score',ascending=False).head(5)

        st.markdown('<div class="section-label">Hybrid Engine Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon">🌟</span> Top Picks For You</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">These recommendations blend <b>Collaborative Filtering (40%)</b> — what similar users love — with <b>Q-Learning RL (60%)</b> — what the adaptive agent has learned about your taste preferences.</div>', unsafe_allow_html=True)

        h_cols = st.columns(5)
        for i, (_, row) in enumerate(hybrid_df.iterrows()):
            with h_cols[i]:
                pct = int(row['hybrid_score']*100)
                score_html = f'<div class="card-score score-hybrid">🔥 {pct}% Match</div>'
                st.markdown(make_card(row['title'],row['type'],row['release_year'],row['listed_in'],score_html,"hybrid"), unsafe_allow_html=True)

        # Hybrid score bar chart
        fig_hybrid = go.Figure(go.Bar(
            x=hybrid_df['title'].str[:22]+'…', y=hybrid_df['hybrid_score'],
            marker=dict(color=hybrid_df['hybrid_score'], colorscale=[[0,'#3a0008'],[0.5,'#a00015'],[1,'#0984e3']],
                        line=dict(color='rgba(0,0,0,0)',width=0)),
            text=[f"{v:.2f}" for v in hybrid_df['hybrid_score']], textposition='outside',
            textfont=dict(color='#E8E8F0', size=11)
        ))
        fig_hybrid.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7A7A8C', family='DM Sans'), height=220,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', range=[0,1.1])
        )
        st.plotly_chart(fig_hybrid, use_container_width=True, config={'displayModeBar':False})

    st.markdown('<div class="fancy-divider">Recommendation Sources</div>', unsafe_allow_html=True)

    # ── KB + RL Side by Side ──────────────────────────────────
    col_kb, col_rl = st.columns(2)

    with col_kb:
        st.markdown('<div class="section-label">Collaborative Filtering</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon">🧠</span> Knowledge-Based</div>', unsafe_allow_html=True)
        if not kb_recs_f.empty:
            for _, row in kb_recs_f.head(5).iterrows():
                score_html = f'<div class="card-score score-kb">✅ Sim: {row["score"]:.2f}</div>'
                st.markdown(make_card(row['title'],row['type'],row.get('release_year','N/A'),row['listed_in'],score_html), unsafe_allow_html=True)
        else:
            st.info("No KB recommendations match current filters.")

    with col_rl:
        st.markdown('<div class="section-label">Reinforcement Learning</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon">🤖</span> Q-Learning Agent</div>', unsafe_allow_html=True)
        if not rl_df_f.empty:
            for _, row in rl_df_f.head(5).iterrows():
                score_html = f'<div class="card-score score-rl">⚡ Q: {row["q_value"]:.4f}</div>'
                st.markdown(make_card(row['title'],row['type'],row['release_year'],row['listed_in'],score_html), unsafe_allow_html=True)
        else:
            st.info("No RL recommendations match current filters.")

    st.markdown('<div class="fancy-divider">Social Signals</div>', unsafe_allow_html=True)

    # ── Similar Users ──────────────────────────────────────────
    st.markdown('<div class="section-label">Taste Matching</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">👥</span> What Similar Users Are Watching</div>', unsafe_allow_html=True)
    if similar_user_recs:
        seen, shown = set(), 0
        sim_cards = ""
        for rec in similar_user_recs:
            if rec['title'] not in seen and shown < 4:
                seen.add(rec['title']); shown += 1
                score_html = f'<div class="card-score score-watch">User {rec["similar_user"]} · ⭐ {rec["rating_score"]}/5</div>'
                sim_cards += make_card(rec['title'],rec['type'],rec.get('release_year','N/A'),rec['listed_in'],score_html)
        st.markdown(f'<div class="card-grid">{sim_cards}</div>', unsafe_allow_html=True)
    else:
        st.info("No similar users data available.")

    # ── User's Watched History ─────────────────────────────────
    st.markdown('<div class="fancy-divider">Watch History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Your Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">👁️</span> Top Rated by You</div>', unsafe_allow_html=True)
    top_watched = user_data.sort_values('rating_score', ascending=False).head(4)
    w_cards = ""
    for _, row in top_watched.iterrows():
        score_html = f'<div class="card-score score-watch">⭐ {row["rating_score"]}/5.0</div>'
        w_cards += make_card(row['title'],row['type'],row.get('release_year','N/A'),row['listed_in'],score_html)
    st.markdown(f'<div class="card-grid">{w_cards}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Data Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">📊</span> Catalog & User Analytics</div>', unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    # Genre distribution
    with a1:
        genre_counts = {}
        for genres in netflix_df['listed_in'].dropna():
            for g in genres.split(','):
                g = g.strip()
                genre_counts[g] = genre_counts.get(g, 0) + 1
        gdf = pd.DataFrame(sorted(genre_counts.items(), key=lambda x:-x[1])[:12], columns=['Genre','Count'])

        fig_genre = go.Figure(go.Bar(
            y=gdf['Genre'], x=gdf['Count'], orientation='h',
            marker=dict(color=gdf['Count'], colorscale=[[0,'#3a0008'],[0.6,'#a00015'],[1,'#0984e3']],
                        line=dict(width=0)),
            text=gdf['Count'], textposition='outside', textfont=dict(color='#9090a0', size=10)
        ))
        fig_genre.update_layout(
            title=dict(text="Top Genres in Catalog", font=dict(color='#c0c0d0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7A7A8C'), height=380, margin=dict(l=0,r=40,t=40,b=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_genre, use_container_width=True, config={'displayModeBar':False})

    # Type pie
    with a2:
        type_counts = netflix_df['type'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=type_counts.index, values=type_counts.values,
            hole=0.55,
            marker=dict(colors=['#0984e3','#74b9ff'], line=dict(color='#1b1e1f', width=3)),
            textfont=dict(color='#fff', size=12)
        ))
        fig_pie.update_layout(
            title=dict(text="Content Type Split", font=dict(color='#c0c0d0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#7A7A8C'),
            height=380, margin=dict(l=0,r=0,t=40,b=0),
            legend=dict(font=dict(color='#9090a0'), bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar':False})

    b1, b2 = st.columns(2)

    # Releases over time
    with b1:
        year_counts = netflix_df['release_year'].value_counts().sort_index()
        fig_year = go.Figure(go.Scatter(
            x=year_counts.index, y=year_counts.values,
            mode='lines+markers',
            line=dict(color='#0984e3', width=2),
            marker=dict(color='#0984e3', size=5),
            fill='tozeroy', fillcolor='rgba(9,132,227,0.07)'
        ))
        fig_year.update_layout(
            title=dict(text="Releases Per Year", font=dict(color='#c0c0d0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7A7A8C'), height=280, margin=dict(l=0,r=0,t=40,b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)')
        )
        st.plotly_chart(fig_year, use_container_width=True, config={'displayModeBar':False})

    # User rating distribution
    with b2:
        fig_rat = go.Figure(go.Histogram(
            x=combined_df['rating_score'],
            nbinsx=5,
            marker=dict(color=['#3a0008','#6a0012','#a00015','#cc0018','#0984e3'],
                        line=dict(color='#1b1e1f', width=2))
        ))
        fig_rat.update_layout(
            title=dict(text="Rating Distribution", font=dict(color='#c0c0d0', size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7A7A8C'), height=280, margin=dict(l=0,r=0,t=40,b=0),
            xaxis=dict(title='Rating', showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)')
        )
        st.plotly_chart(fig_rat, use_container_width=True, config={'displayModeBar':False})

    # User activity
    st.markdown('<div class="section-title" style="margin-top:1rem"><span class="icon">🏆</span> Most Active Users</div>', unsafe_allow_html=True)
    top_users = combined_df.groupby('userId').agg(watched=('movieId','count'), avg_rating=('rating_score','mean')).sort_values('watched',ascending=False).head(10).reset_index()
    fig_users = go.Figure()
    fig_users.add_trace(go.Bar(
        name='Titles Watched', x=top_users['userId'].astype(str), y=top_users['watched'],
        marker_color='#0984e3', opacity=0.85
    ))
    fig_users.add_trace(go.Scatter(
        name='Avg Rating', x=top_users['userId'].astype(str), y=top_users['avg_rating']*10,
        mode='lines+markers', yaxis='y2',
        line=dict(color='#74b9ff', width=2), marker=dict(size=6)
    ))
    fig_users.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7A7A8C'), height=300, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(font=dict(color='#9090a0'), bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1),
        xaxis=dict(showgrid=False, title='User ID'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', title='Watched'),
        yaxis2=dict(overlaying='y', side='right', title='Avg Rating ×10', showgrid=False),
        barmode='group'
    )
    st.plotly_chart(fig_users, use_container_width=True, config={'displayModeBar':False})


# ════════════════════════════════════════════════════════════
# TAB 3 — SEARCH
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Content Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="icon">🔍</span> Search the Catalog</div>', unsafe_allow_html=True)

    search_query = st.text_input("", placeholder="Search by title, e.g. Inception…")

    if search_query:
        results = netflix_df[netflix_df['title'].str.contains(search_query, case=False, na=False)].head(8)
        if not results.empty:
            st.markdown(f"<p style='color:var(--muted);font-size:0.82rem;margin-bottom:1rem'>Found <b style='color:#fff'>{len(results)}</b> result(s)</p>", unsafe_allow_html=True)
            r_cols = st.columns(4)
            for i, (_, row) in enumerate(results.iterrows()):
                m_id = row['show_id']
                m_ratings = combined_df[combined_df['movieId']==m_id]['rating_score']
                views, avg_r = len(m_ratings), (m_ratings.mean() if len(m_ratings) else 0)
                sim = get_movie_kb_score(selected_user, m_id, combined_df, user_movie_matrix, movieId_dict)
                sim_text = f"{sim:.2f}" if sim > 0 else "—"
                with r_cols[i % 4]:
                    st.markdown(f"""
<div class="movie-card" style="border-color:rgba(9,132,227,0.3)">
  <span class="card-type-badge {'badge-tv' if row['type']=='TV Show' else 'badge-movie'}">{row['type']}</span>
  <div class="card-title">{row['title']}</div>
  <div class="card-meta">📅 {row.get('release_year','N/A')}</div>
  <div class="card-meta">🎭 {str(row['listed_in'])[:40]}</div>
  <div class="card-meta">👁️ {views} ratings &nbsp;·&nbsp; ⭐ {avg_r:.1f}</div>
  <div class="card-score score-hybrid">Similarity: {sim_text}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.warning("No titles found. Try a different keyword.")
    else:
        # Trending  (most rated)
        st.markdown('<div class="section-title" style="margin-top:1rem"><span class="icon">🔥</span> Trending Now</div>', unsafe_allow_html=True)
        trending = (combined_df.groupby('movieId')
                    .agg(ratings_count=('rating_score','count'), avg_score=('rating_score','mean'))
                    .sort_values('ratings_count', ascending=False)
                    .head(8).reset_index())
        trending = pd.merge(trending, netflix_df[['show_id','title','type','listed_in','release_year']], left_on='movieId', right_on='show_id', how='left')

        t_cols = st.columns(4)
        for i, (_, row) in enumerate(trending.iterrows()):
            with t_cols[i % 4]:
                score_html = f'<div class="card-score score-kb">🔥 {row["ratings_count"]} ratings · ⭐ {row["avg_score"]:.1f}</div>'
                st.markdown(make_card(row['title'],row['type'],row.get('release_year','N/A'),row['listed_in'],score_html), unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;border-top:1px solid rgba(255,255,255,0.05);padding-top:1.5rem;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:28px;height:28px;background:#0984e3;border-radius:6px;
         display:flex;align-items:center;justify-content:center;
         font-family:Syne,sans-serif;font-weight:800;font-size:0.75rem;color:#fff">Ci</div>
    <span style="font-family:Syne,sans-serif;font-weight:700;color:#fff;font-size:0.9rem">Cine-Mela</span>
  </div>
  <span style="font-size:0.72rem;color:#444;letter-spacing:0.06em">
    COLLABORATIVE FILTERING · Q-LEARNING · HYBRID RANKING ENGINE
  </span>
  <span style="font-size:0.72rem;color:#444">Built with Streamlit + Plotly</span>
</div>
""", unsafe_allow_html=True)