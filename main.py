import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies_data.csv",
                     low_memory=False)  # forces pandas to process the wntire file before deciding on the data types
# since cols like popularity have many types of data
movies = movies[['title', 'genres', 'id', 'overview', 'vote_average', 'vote_count']]
ratings = pd.read_csv("ratings.csv")
ratings = ratings[['userId', 'movieId', 'rating']]
credits = pd.read_csv("credits.csv")

movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)
movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(0)

movies = movies.sort_values(by='vote_count', ascending=False).head(12000)

# ===============================
# 5. MERGE WITH CREDITS (ONLY 25K MOVIES)
# ===============================
content_df = pd.merge(movies, credits, on='id')
# 6. FILTER RATINGS (ONLY EXISTING MOVIES)
# ===============================
ratings = ratings[ratings['movieId'].isin(content_df['id'])]
import ast


def convert_genre(obj):
    L = []
    for i in ast.literal_eval(
            obj):  # each i represents one dictionary...to conert to real list...given is not a real list...python read it as a string
        L.append(i['name'])  # only push names of each i into L
    return L


movies['genres'] = movies['genres'].apply(convert_genre)


def convert_cast(obj):
    C = []
    for i in ast.literal_eval(obj)[:3]:
        C.append(i['name'])
    return C


credits['cast'] = credits['cast'].apply(convert_cast)


def convert_crew(obj):
    try:
        crew_list = ast.literal_eval(obj)
        for i in crew_list:
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except:
        return []


credits['crew'] = credits['crew'].apply(convert_crew)

content_df = pd.merge(movies, credits, on='id')
content_df.head(5)
content_df['genres'] = content_df['genres'].apply(lambda x: " ".join(x))

content_df['cast'] = content_df['cast'].apply(lambda x: " ".join(x))
content_df['crew'] = content_df['crew'].apply(lambda x: " ".join(x))

content_df['genres'] = content_df['genres'].fillna('')
content_df['crew'] = content_df['crew'].fillna('')
content_df['cast'] = content_df['cast'].fillna('')
content_df['overview'] = content_df['overview'].fillna('')
content_df['title'] = content_df['title'].fillna('')
content_df['vote_average'] = content_df['vote_average'].fillna('')
content_df['vote_count'] = content_df['vote_count'].fillna('')

content_df['tags'] = content_df['title'] + " " + content_df['genres']*3 + " " + content_df['cast'] + " " + content_df[
    'crew'] + " " + content_df['overview']

content_df['tags'] = content_df['tags'].str.lower()
content_df['title'] = content_df['title'].str.lower()
content_df['tags'] = content_df['tags'].str[:300]

content_df = content_df.drop(columns=['genres', 'cast', 'crew', 'overview'])
content_df['id'] = content_df['id'].astype(int)
content_df['vote_count'] = pd.to_numeric(content_df['vote_count'], errors='coerce')
content_df['vote_average'] = pd.to_numeric(content_df['vote_average'], errors='coerce')

collab_df = movies[['title', 'id']]
collab_df = pd.merge(collab_df, ratings, left_on='id', right_on='movieId')
collab_df['movieId'] = collab_df['movieId'].astype(int)

# reduce users
user_counts = collab_df['userId'].value_counts()
active_users = user_counts[user_counts > 50].index
collab_df = collab_df[collab_df['userId'].isin(active_users)]

# reduce movies
movie_counts = collab_df['movieId'].value_counts()
popular_movies = movie_counts[movie_counts > 20].index
collab_df = collab_df[collab_df['movieId'].isin(popular_movies)]
print(content_df.isnull().sum())

# Content based
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 2. CONTENT-BASED (TF-IDF)
# ===============================
tfidf = TfidfVectorizer(
    max_features=12000,
    stop_words='english',
    ngram_range=(1,3),   # BIG improvement
    min_df=2
)
tfidf_matrix = tfidf.fit_transform(content_df['tags'])

content_similarity = cosine_similarity(tfidf_matrix)

# 3. COLLABORATIVE FILTERING
from scipy.sparse import csr_matrix

# Create pivot
movie_user_matrix = collab_df.pivot_table(
    index='movieId', columns='userId', values='rating'
).fillna(0)






# 5. WEIGHTED RATING (IMDB)
# ===============================
C = content_df['vote_average'].mean()
m = content_df['vote_count'].quantile(0.7)


def weighted_rating(row):
    v = row['vote_count']
    R = row['vote_average']
    return (v / (v + m) * R) + (m / (v + m) * C)


content_df['weighted_score'] = content_df.apply(weighted_rating, axis=1)


# 6. NORMALIZE FUNCTION
# ===============================
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


# 7. RECOMMEND FUNCTION (FULLY OPTIMIZED)
# ===============================
def recommend(movie_name):
    movie_name = movie_name.lower().strip()

    match = content_df[content_df['title'].str.lower() == movie_name]
    if match.empty:
        return "Movie not found"

    idx = match.index[0]
    movie_id = content_df.iloc[idx]['id']

    # -------- CONTENT SIMILARITY (ON DEMAND) --------
    content_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    content_vals = normalize(content_scores)

    # -------- COLLABORATIVE SIMILARITY (FIXED) --------
    collab_vals = np.zeros(len(content_df))

    if movie_id in movie_user_matrix.index:

        target_vector = movie_user_matrix.loc[movie_id].values.reshape(1, -1)

        collab_scores = cosine_similarity(target_vector, movie_user_matrix.values).flatten()

        # Create mapping
        id_to_index = {id_: i for i, id_ in enumerate(content_df['id'])}

        # Map scores correctly
        for j, mid in enumerate(movie_user_matrix.index):
            if mid in id_to_index:
                collab_vals[id_to_index[mid]] = collab_scores[j]

        collab_vals = normalize(collab_vals)

    # -------- DYNAMIC WEIGHTS --------
    votes = content_df.iloc[idx]['vote_count']

    if votes > m:
        content_w = 0.6
        collab_w = 0.4
    else:
        content_w = 0.9
        collab_w = 0.1

    # -------- HYBRID SCORE --------
    hybrid_scores = []

    for i in range(len(content_df)):
        score = (content_vals[i] * content_w) + (collab_vals[i] * collab_w)

        # add weighted rating boost
        score += 0.02 * content_df.iloc[i]['weighted_score']

        hybrid_scores.append((i, score))

    # Sort scores
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)

    # Remove same movie
    hybrid_scores = hybrid_scores[1:20]

    # Return top 10
    result = []
    for i, score in hybrid_scores:
        result.append(content_df.iloc[i]['title'])
        if len(result) == 10:
            break

    return result




