import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos a serem importados
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

db_path = "alunos.xlsx"

# Armazena os dados de cada planilha em um DataFrame
relacao_alunos = pd.read_excel(db_path, sheet_name='Relacao_Alunos', skiprows=2)
historico = pd.read_excel(db_path, sheet_name='Historico', skiprows=2)
socioeconomico = pd.read_excel(db_path, sheet_name='Questionario_socioEconomico', skiprows=2)

def criar_dataframe_relacao_alunos(relacao_alunos):
    # Extraindo apenas as colunas necessárias das planilhas
    dados_relacao = relacao_alunos[['id', 'Sigla Cota', 'Escola Pública?', 'Coeficiente', 'Nota Enem', 
                                'Sexo', 'Estado', 'Situação Atual do Aluno' ]].copy()
    
    #Adiciona a idade como coluna calculada
    dados_relacao['Idade'] = relacao_alunos['Ano Ingresso'] - pd.to_datetime(
                                                                relacao_alunos['Data Nascimento'], format='%d/%m/%Y').dt.year
    
    # Tratamento de nulos/ausentes
    dados_relacao['Sigla Cota'].fillna('Nao Cotista', inplace=True)

    dados_relacao['Escola Pública?'] = dados_relacao['Escola Pública?'].astype(str).str.strip()
    dados_relacao['Escola Pública?'] = dados_relacao['Escola Pública?'].replace(['-', 'nan', 'NaN', 'None'], pd.NA)
    dados_relacao['Escola Pública?'].fillna('Escola Pública', inplace=True)
    
    dados_relacao['Coeficiente'] = pd.to_numeric(dados_relacao['Coeficiente'], errors='coerce')
    media_coeficiente = dados_relacao['Coeficiente'].mean()
    dados_relacao['Coeficiente'].fillna(media_coeficiente, inplace=True)
    
    media_enem = dados_relacao['Nota Enem'].mean()
    dados_relacao['Nota Enem'].fillna(media_enem, inplace=True)
    
    return dados_relacao

def criar_dataframe_historico(historico):
    # Extrair apenas as colunas necessárias do histórico
    dados_historico = historico[['id', 'Freq.(%)']].copy()
    
    # Substituir valores ausentes pela média
    media_frequencia = dados_historico['Freq.(%)'].mean()
    dados_historico['Freq.(%)'].fillna(media_frequencia, inplace=True)

    # Calcular a média de frequência para cada aluno
    dados_historico = dados_historico.groupby('id')['Freq.(%)'].mean().reset_index()
    dados_historico.rename(columns={'Freq.(%)': 'Frequencia Media'}, inplace=True)

    return dados_historico

def criar_dataframe_socioeconomico(socioeconomico):
    # Vamos usar apenas o transporte do questionario
    transporte = socioeconomico[socioeconomico["Cod. Questão"] == 16].copy()
    
    # Cria o dataframe com 'id', 'transporte'
    dados_transporte = transporte[['id', 'Resposta']].copy()
    dados_transporte.rename(columns={'Resposta': 'Transporte'}, inplace=True)
    
    # Trata valores nulos se houver
    dados_transporte['Transporte'].fillna('Não informado', inplace=True)
    
    return dados_transporte

def criar_dataframe_final(relacao_alunos, historico, socioeconomico):
    # Cria os dataframes individuais
    df_relacao = criar_dataframe_relacao_alunos(relacao_alunos)
    df_historico = criar_dataframe_historico(historico)
    df_socioeconomico = criar_dataframe_socioeconomico(socioeconomico)
    
    # Primeiro, combina relação de alunos com histórico
    df_intermediario = pd.merge(df_relacao, df_historico, on='id', how='left')
    
    # Em seguida, combina o resultado com os dados socioeconômicos
    df_final = pd.merge(df_intermediario, df_socioeconomico, on='id', how='left')
    
    # Trata valores ausentes que podem surgir após o merge
    df_final['Frequencia Media'].fillna(df_historico['Frequencia Media'].mean(), inplace=True)
    df_final['Transporte'].fillna('Não informado', inplace=True)
    
    return df_final

def main():
    base_df = criar_dataframe_final(relacao_alunos, historico, socioeconomico)
    print(f'DataFrame final criado com {len(base_df)} registros e {len(base_df.columns)} colunas.')
    base_df.to_excel('alunos_unificado.xlsx', index=False)

if __name__ == "__main__":
    main()