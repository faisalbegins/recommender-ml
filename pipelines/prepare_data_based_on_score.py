def execute(df=None):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.9)

    weighted_movies = df.copy().loc[df['vote_count'] >= m]

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    weighted_movies['score'] = weighted_movies.apply(weighted_rating, axis=1)

    return weighted_movies
