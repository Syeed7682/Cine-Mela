=========================================================
CINE-MELA : NETFLIX HYBRID RECOMMENDATION DASHBOARD
=========================================================

1. OVERVIEW
-----------
Cine-Mela is a professional, high-performance web dashboard built with Streamlit that provides highly personalized Netflix movie and TV show recommendations.

It utilizes a powerful Hybrid Recommendation Engine that blends:
- Knowledge-Based Collaborative Filtering (40%): Suggests content based on the viewing habits of similar users.
- Reinforcement Learning / Q-Learning (60%): An AI agent that dynamically learns and explores user preferences over time.


2. KEY FEATURES
---------------
* Advanced Hybrid Engine: Blends community taste with an adaptive AI agent.
* Real-Time Search & Discovery: Instantly search the Netflix catalog. See live viewership metrics, average ratings, and a personalized "Similarity Score" for every search result.
* Interactive Analytics: Beautiful, dynamic Plotly charts showcasing catalog genre distributions, release trends, and user community insights.
* Premium UI/UX: A highly polished dark-mode interface featuring glassmorphism, fluid micro-animations, and a custom blue/charcoal color palette.
* Granular Controls: Instantly filter the entire dashboard by User Profile, Genre, and Release Year.


3. INSTALLATION
---------------
Ensure you have Python 3.8+ installed. 
Install the required dependencies using the provided requirements.txt file:

    pip install -r requirements.txt

The main dependencies include:
- streamlit
- pandas & numpy
- scipy & scikit-learn
- plotly
- kagglehub (for dataset downloading)


4. HOW TO RUN
-------------
Navigate to the project folder in your terminal and run:

    python -m streamlit run app.py

The dashboard will open automatically in your default web browser (usually at http://localhost:8501).


5. DATASET
----------
The application uses the "Netflix Movies and TV Shows" dataset from Kaggle via `kagglehub`. If the dataset cannot be downloaded, the app will automatically fall back to generating a robust synthetic dataset so you can immediately interact with the dashboard.
