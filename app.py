from flask import Flask, request, render_template
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)



from dotenv import load_dotenv
import os

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")



movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')


movies_df = movies_df.merge(credits_df[['title', 'cast', 'crew']], on='title', how='left')


def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return " ".join([genre['name'] for genre in genres])
    except:
        return ""

movies_df['parsed_genres'] = movies_df['genres'].apply(extract_genres)
movies_df['content'] = movies_df['overview'].fillna('') + " " + movies_df['parsed_genres'].fillna('')


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['content'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


title_to_index = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()


def fetch_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get('results') and len(data['results']) > 0:
            return data['results'][0].get('poster_path')  # Just the path like /abc.jpg
    except Exception as e:
        print("Error fetching poster:", e)

    return None

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = title_to_index.get(title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    seen_titles = set()
    recommendations = []

    for i, _ in sim_scores:
        movie_title = movies_df['title'].iloc[i]
        if movie_title != title and movie_title not in seen_titles:
            poster_path = fetch_poster(movie_title)
            recommendations.append({
                'title': movie_title,
                'poster_path': poster_path
            })
            seen_titles.add(movie_title)
        if len(recommendations) == 5:
            break

    return recommendations




@app.route('/')
def home():
    all_titles = sorted(movies_df['title'].unique())
    return render_template('index.html', all_titles=all_titles)
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    recommendations = get_recommendations(movie_title)
    all_titles = sorted(movies_df['title'].unique())
    return render_template('index.html', movie=movie_title, recommendations=recommendations, all_titles=all_titles)

if __name__ == '__main__':
    app.run(debug=True)
