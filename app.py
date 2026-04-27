from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix
import ast
import pickle
import os

app = Flask(__name__)
CORS(app)

CACHE_FILE = "model_cache.pkl"

if os.path.exists(CACHE_FILE):
    print("⚡ Loading from cache...")
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    content_df        = cache["content_df"]
    tfidf_matrix      = cache["tfidf_matrix"]
    movie_user_matrix = cache["movie_user_matrix"]
    m                 = cache["m"]
    C                 = cache["C"]
    print("✅ Cache loaded!")

else:
    print("⏳ First run, building cache...")

    # 1. Load CSVs
    movies  = pd.read_csv("movies.csv", low_memory=False)
    movies  = movies[['title', 'genres', 'id', 'overview', 'vote_average', 'vote_count']]
    ratings = pd.read_csv("ratings.csv")
    ratings = ratings[['userId', 'movieId', 'rating']]
    credits = pd.read_csv("credits.csv")

    # 2. Type conversions
    movies['id']           = pd.to_numeric(movies['id'], errors='coerce')
    ratings['movieId']     = pd.to_numeric(ratings['movieId'], errors='coerce')
    credits['id']          = pd.to_numeric(credits['id'], errors='coerce')
    movies['vote_count']   = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)
    movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)

    movies = movies.sort_values(by='vote_count', ascending=False).head(12000)

    # 3. Parse genres, cast, crew
    def convert_genre(obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []

    def convert_cast(obj):
        try:
            C = []
            for i in ast.literal_eval(obj)[:3]:
                C.append(i['name'])
            return C
        except:
            return []

    def convert_crew(obj):
        try:
            crew_list = ast.literal_eval(obj)
            for i in crew_list:
                if i['job'] == 'Director':
                    return [i['name']]
            return []
        except:
            return []

    movies['genres']  = movies['genres'].apply(convert_genre)
    credits['cast']   = credits['cast'].apply(convert_cast)
    credits['crew']   = credits['crew'].apply(convert_crew)

    # 4. Merge & build content_df
    content_df = pd.merge(movies, credits, on='id')
    ratings    = ratings[ratings['movieId'].isin(content_df['id'])]

    content_df['genres']  = content_df['genres'].apply(lambda x: " ".join(x) * 4)
    content_df['cast']    = content_df['cast'].apply(lambda x: " ".join(x))
    content_df['crew']    = content_df['crew'].apply(lambda x: " ".join(x))

    for col in ['genres', 'crew', 'cast', 'overview', 'title', 'vote_average', 'vote_count']:
        content_df[col] = content_df[col].fillna('')

    content_df['tags'] = (
        content_df['title'] + " " +
        content_df['genres'] + " " +
        content_df['cast'] + " " +
        content_df['crew'] + " " +
        content_df['overview']
    )
    content_df['tags']  = content_df['tags'].str.lower().str[:300]
    content_df['title'] = content_df['title'].str.lower()
    content_df = content_df.drop(columns=['genres', 'cast', 'crew', 'overview'])

    content_df['id']           = content_df['id'].astype(int)
    content_df['vote_count']   = pd.to_numeric(content_df['vote_count'], errors='coerce')
    content_df['vote_average'] = pd.to_numeric(content_df['vote_average'], errors='coerce')

    # 5. TF-IDF
    print("⏳ Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=6000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(content_df['tags'])

    # 6. Collaborative filtering
    print("⏳ Building collaborative filter...")
    collab_df = movies[['title', 'id']].copy()
    collab_df = pd.merge(collab_df, ratings, left_on='id', right_on='movieId')
    collab_df['movieId'] = collab_df['movieId'].astype(int)

    movie_user_matrix = collab_df.pivot_table(
        index='movieId', columns='userId', values='rating'
    ).fillna(0)

    # 7. IMDB Weighted Score
    C = content_df['vote_average'].mean()
    m = content_df['vote_count'].quantile(0.7)

    def weighted_rating(row):
        v = row['vote_count']
        R = row['vote_average']
        return (v / (v + m) * R) + (m / (v + m) * C)

    content_df['weighted_score'] = content_df.apply(weighted_rating, axis=1)

    # 8. Save cache
    print("💾 Saving cache...")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "content_df":        content_df,
            "tfidf_matrix":      tfidf_matrix,
            "movie_user_matrix": movie_user_matrix,
            "m":                 m,
            "C":                 C
        }, f)
    print("✅ Cache saved!")

print("✅ Models ready!")

# ── Normalize ─────────────────────────────────────────────────
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

# ── Recommend ─────────────────────────────────────────────────
def recommend(movie_name):
    movie_name = movie_name.lower().strip()

    match = content_df[content_df['title'].str.lower() == movie_name]
    if match.empty:
        return None

    idx      = match.index[0]
    movie_id = content_df.iloc[idx]['id']

    # Content similarity
    content_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    content_vals   = normalize(content_scores)

    # Collaborative similarity
    collab_vals = np.zeros(len(content_df))
    if movie_id in movie_user_matrix.index:
        target_vector = movie_user_matrix.loc[movie_id].values.reshape(1, -1)
        collab_scores = cosine_similarity(target_vector, movie_user_matrix.values).flatten()
        id_to_index   = {id_: i for i, id_ in enumerate(content_df['id'])}
        for j, mid in enumerate(movie_user_matrix.index):
            if mid in id_to_index:
                collab_vals[id_to_index[mid]] = collab_scores[j]
        collab_vals = normalize(collab_vals)

    # Dynamic weights
    votes     = content_df.iloc[idx]['vote_count']
    content_w = 0.5 if votes > m else 0.8
    collab_w  = 0.5 if votes > m else 0.2

    # Hybrid scores
    hybrid_scores = []
    for i in range(len(content_df)):
        score = (content_vals[i] * content_w) + (collab_vals[i] * collab_w)
        score += 0.02 * content_df.iloc[i]['weighted_score']
        hybrid_scores.append((i, score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[0:20]

    result = []
    for i, score in hybrid_scores:
        result.append(content_df.iloc[i]['title'])
        if len(result) == 10:
            break

    return result if result else None

# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend")
def get_recommendations():
    movie = request.args.get("movie", "").strip()
    if not movie:
        return jsonify({"error": "Please provide a movie name"}), 400

    result = recommend(movie)
    if result is None:
        return jsonify({"error": f"Movie '{movie}' not found in database"}), 404

    return jsonify({"recommendations": result})

@app.route("/search")
def search_movies():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
    matches = content_df[content_df['title'].str.lower().str.contains(query, na=False)]['title'].head(10).tolist()
    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
