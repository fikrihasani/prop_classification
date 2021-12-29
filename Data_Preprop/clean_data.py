from numpy.lib.npyio import load
import pandas as pd

def load_data_to_df():
    df = pd.read_excel("Data/Labeling Skripsi S1.xlsx",sheet_name="Data")
    return df


load_data_to_df()