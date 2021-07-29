from pipelines import read_data
from pipelines import rename_column
from pipelines import merge_data_frame
from pipelines import prepare_data_based_on_score
from pipelines import serialize_model

# read movie data from the csv file
credits_df, movies_df = read_data.execute()

# rename columns
rename_column.execute(credits_df)

# merge data frame
df = merge_data_frame.execute(credits_df, movies_df)

# prepare data based on score
weighted_moves = prepare_data_based_on_score.execute(df)

# serialize prepare model
serialize_model.execute(weighted_moves)

