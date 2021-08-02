import pandas as pd
import numpy as np
import pickle


def prepare_data():
    # read two data files
    credits_df = pd.read_csv('/Users/Faisal/Development/recommender-storage/data/tmdb_credits.csv')
    movies_df = pd.read_csv('/Users/Faisal/Development/recommender-storage/data/tmdb_movies.csv')

    # Join two dataset based on id (movie_id)
    credits_df.columns = ['id', 'tittle', 'cast', 'crew']
    df = movies_df.merge(credits_df, on='id')

    # drop unnecessary columns
    df = df.drop(['budget', 'homepage',
                  'original_title', 'production_companies',
                  'release_date', 'revenue', 'runtime',
                  'spoken_languages', 'status', 'tagline', 'tittle'], axis=1)

    # rename columns
    df.columns = ['genres', 'id', 'keywords',
                  'language', 'overview',
                  'popularity', 'countries',
                  'title', 'vote_average',
                  'vote_count', 'cast', 'crew']

    # fill null values with empty string
    df['overview'] = df['overview'].fillna('')

    # find mean vote accross the whole dataset
    C = df['vote_average'].mean()

    # find minimum vote requires to be listed
    m = df['vote_count'].quantile(.9)

    # all all the movies that matches minimum vote requirements
    movies_filter_by_vote_count = df.copy().loc[df['vote_count'] >= m]

    # weighted rating calculation
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    # Add a new feature 'score' and calculate its value with `weighted_rating()`
    movies_filter_by_vote_count['score'] = movies_filter_by_vote_count.apply(weighted_rating, axis=1)

    # find top trending movies based on score
    trending_movies = movies_filter_by_vote_count.sort_values('score', ascending=False)

    # find top movies based on popularity
    popular_movies = df.copy().sort_values('popularity', ascending=False)

    content_based_movies = df.copy()

    from ast import literal_eval
    features = ['cast', 'crew', 'keywords', 'genres']
    for feature in features:
        content_based_movies[feature] = content_based_movies[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # Returns the list top 3 elements or entire list; whichever is more.
    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names
        # Return empty list in case of missing/malformed data
        return []

    content_based_movies['director'] = content_based_movies['crew'].apply(get_director)

    features = ['cast', 'keywords', 'genres']
    for feature in features:
        content_based_movies[feature] = content_based_movies[feature].apply(get_list)

    # clean data
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # clean data
    for feature in ['cast', 'director', 'keywords', 'genres']:
        content_based_movies[feature] = content_based_movies[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(
            x['genres']) + ' ' + x['overview']

    content_based_movies['soup'] = content_based_movies.apply(create_soup, axis=1)

    # Import CountVectorizer and create the count matrix
    from sklearn.feature_extraction.text import CountVectorizer
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(content_based_movies['soup'])

    # Compute the Cosine Similarity matrix based on the count_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity_matrix = cosine_similarity(count_matrix, count_matrix)

    # SVD

    # imports
    from surprise import Reader, Dataset, SVD
    from surprise.model_selection import cross_validate

    # read data file
    ratings = pd.read_csv('/Users/Faisal/Development/recommender-storage/data/user_ratings.csv')

    # load data
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], Reader())

    # create svd
    svd = SVD()

    # cross validation
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # fit data
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    movies_for_prediction = df[['id', 'title']]
    all_users = ratings['userId'].unique()
    predicted_movies_with_rating = []

    for user in all_users[0:100]:
        for movie in movies_for_prediction.values[0:20]:
            users_ratings = ratings[(ratings['userId'] == user) & (ratings['movieId'] == movie[0])]
            if len(users_ratings.index) == 0:
                predicted_movies_with_rating.append([user, movie[0], movie[1], svd.predict(user, movie[0]).est])

    svd_lookup = pd.DataFrame(predicted_movies_with_rating)
    svd_lookup.columns = ['user', 'movie', 'title', 'prediction']

    # export all the models
    models = [
        (trending_movies, '/Users/Faisal/Development/recommender-storage/models/trending.data'),
        (popular_movies, '/Users/Faisal/Development/recommender-storage/models/popular.data'),
        (df, '/Users/Faisal/Development/recommender-storage/models/generic.data'),
        (content_based_movies, '/Users/Faisal/Development/recommender-storage/models/content_based.data'),
        (cosine_similarity_matrix, '/Users/Faisal/Development/recommender-storage/models/similarity.matrix'),
        (svd, '/Users/Faisal/Development/recommender-storage/models/svd_raw.model'),
        (svd_lookup, '/Users/Faisal/Development/recommender-storage/models/svd_lookup.data')
    ]

    return models


def serialize_models(models):
    for model in models:
        outfile = open(model[1], 'wb')
        pickle.dump(model[0], outfile)


def main():
    models = prepare_data()
    serialize_models(models)


if __name__ == '__main__':
    main()
