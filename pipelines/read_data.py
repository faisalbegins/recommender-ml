def execute():
    from utilities import file_reader as fr
    from utilities import constants as const

    credits_df = fr.read_csv(const.CREDITS_DATA_FILE)
    movies_df = fr.read_csv(const.MOVIES_DATA_FILE)
    return credits_df, movies_df
