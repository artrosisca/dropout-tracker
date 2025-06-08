import pandas as pd

db_path = "students-database.xlsx"

df_alunos = pd.read_excel(db_path, sheet_name=0, engine="openpyxl", skiprows=1)

# todo: muita coisa