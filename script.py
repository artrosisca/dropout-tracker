import pandas as pd
import numpy as np

db_path = "students-database.csv"

df = pd.read_csv(db_path, encoding='utf-8', skiprows=2)

print(df['Sexo'].values)