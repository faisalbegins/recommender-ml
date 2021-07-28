class DemographicRecommender:
    def __init__(self, df):
        self.df = df

    def get_recommendation(self):
        return self.df['month']
