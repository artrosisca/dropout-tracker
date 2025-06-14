import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

alunos_unificado_path = "alunos_unificado.xlsx"

df = pd.read_excel(alunos_unificado_path)

# Categorica = string, precisa ser transformada em numérica
colunas_categoricas = ['Sigla Cota', 'Escola Pública?', 'Sexo', 
                       'Estado', 'Situação Atual do Aluno', 'Transporte']

# Aplica Label Encoding nas colunas categoricas
le = LabelEncoder()
for col in colunas_categoricas:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        print(f"Coluna {col} não encontrada no DataFrame.")

# Salvar a coluna ID antes da normalização
id_column = df['id'].copy() if 'id' in df.columns else None

# Colunas para normalizar (todas exceto 'id')
colunas_para_normalizar = [col for col in df.columns if col != 'id']

if colunas_para_normalizar:
    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

if id_column is not None:
    df['id'] = id_column

df.to_excel("alunos_normalizado.xlsx", index=False)