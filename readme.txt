# 🎬 CINE-MELA: Netflix Hybrid Recommendation Dashboard

A sleek, high-performance **Streamlit** web dashboard that delivers highly personalized Netflix movie and TV show recommendations using a powerful **Hybrid Recommendation Engine**.

---

## ✨ Overview

**CINE-MELA** combines the wisdom of the crowd with adaptive artificial intelligence to provide intelligent, dynamic, and personalized content recommendations.

The system features a sophisticated **Hybrid Recommendation Engine** that intelligently blends:

- **Knowledge-Based Collaborative Filtering** (40%) — Leverages viewing patterns of similar users  
- **Reinforcement Learning (Q-Learning)** (60%) — An intelligent AI agent that continuously learns and adapts to individual user preferences over time

This hybrid approach ensures recommendations are both **community-driven** and **highly personalized**, improving over time as the RL agent learns from user interactions.

---

## 🚀 Key Features

- **Advanced Hybrid Recommendation Engine** — Best of collaborative filtering and reinforcement learning
- **Real-Time Search & Discovery** — Instant search across the Netflix catalog with live viewership metrics, average ratings, and personalized **Similarity Score**
- **Interactive Analytics** — Stunning Plotly visualizations including genre distributions, release year trends, and community insights
- **Premium Dark UI/UX** — Modern glassmorphism design with fluid micro-animations and a refined blue & charcoal color palette
- **Granular Filtering** — Filter recommendations and analytics by **User Profile**, **Genre**, and **Release Year**
- **User Profile Management** — Create and switch between multiple user profiles
- **Feedback Loop** — RL agent improves recommendations based on user interactions

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with custom CSS for glassmorphism)
- **Backend**: Python 3.8+
- **Recommendation Engine**:
  - Collaborative Filtering
  - Reinforcement Learning (Q-Learning)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Others**: scikit-learn, Matplotlib

---

## 📥 Installation

1. Ensure you have **Python 3.8 or higher** installed.
2. Clone the repository:

   ```bash
   git clone https://github.com/Syeed7682/Cine-Mela.git
   cd Cine-Mela
   ```

3. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Place your Netflix dataset inside the `data/` folder:
   - `netflix_titles.csv`
   - (Optional) `ratings.csv` for collaborative filtering
   *(Note: If no dataset is found, the app automatically downloads it via kagglehub or creates a synthetic dataset for demonstration!)*

---

## ▶️ How to Run

```bash
python -m streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`
Website : https://cine-mela.streamlit.app/
Dataset: kaggle kernels pull radiantmohit/movie-recomendation-netflix create with this dataset
---

## 📂 Project Structure

```
Cine-Mela/
├── app.py                             # Main Streamlit application
├── netflix_recommendation_system.py   # Hybrid recommendation engine core logic
├── requirements.txt                   # Project dependencies
├── readme.txt                         # Project Documentation
└── .gitignore                         # Git ignore file
```

---

## 🧠 How the Hybrid Engine Works

The recommendation score is calculated as:

**Final Score = (0.4 × Collaborative Filtering Score) + (0.6 × Q-Learning Score)**

- **Collaborative Filtering**: Finds similar users and recommends what they liked.
- **Q-Learning Agent**: Learns optimal recommendations through trial and error using rewards from user feedback (likes, watches, skips).

---

## 🎨 UI/UX Highlights

- Modern dark theme with glassmorphism effects
- Smooth animations and transitions
- Fully responsive design
- Interactive charts and filters

---

## 🔮 Future Enhancements

- Content-based filtering using embeddings
- Deep Q-Network (DQN) upgrade
- User authentication and persistent profiles
- Docker support
- Recommendation export feature

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- Netflix for the inspiration
- Streamlit team
- Kaggle Netflix Datasets
- Reinforcement Learning community

---

**Made with ❤️ for movie lovers**

*Enjoy your personalized cinematic journey with CINE-MELA! 🎥*
