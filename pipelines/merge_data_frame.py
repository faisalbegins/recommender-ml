def execute(credits_df=None, movies_df=None):
    return movies_df.merge(credits_df, on='id')





