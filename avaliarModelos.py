import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Modelos a serem importados
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def carregar_e_preparar_dados():
    """
    Carrega os dados diretamente do arquivo Excel e suas abas,
    unifica e faz o pré-processamento.
    Retorna os dados prontos para o modelo (X, y) e os dados de treino/teste.
    """
    try:
        db_path = "alunos.xlsx"
        relacao_alunos = pd.read_excel(db_path, sheet_name='Relacao_Alunos', skiprows=2)
        historico = pd.read_excel(db_path, sheet_name='Historico', skiprows=2)
        socioeconomico = pd.read_excel(db_path, sheet_name='Questionario_socioEconomico', skiprows=2)
        
    except FileNotFoundError:
        print(f"Arquivo '{db_path}' não encontrado. Certifique-se de que ele está na mesma pasta do script.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo Excel: {e}")
        return None, None, None, None, None

    # Funções de Unificação
    def criar_dataframe_relacao_alunos(relacao_alunos):
        dados_relacao = relacao_alunos[['id', 'Sigla Cota', 'Escola Pública?', 'Coeficiente', 'Nota Enem', 
                                     'Sexo', 'Estado', 'Situação Atual do Aluno' ]].copy()
        dados_relacao['Idade'] = relacao_alunos['Ano Ingresso'] - pd.to_datetime(
                                                                    relacao_alunos['Data Nascimento'], format='%d/%m/%Y', errors='coerce').dt.year
        dados_relacao['Idade'].fillna(dados_relacao['Idade'].median(), inplace=True)
        dados_relacao['Sigla Cota'].fillna('Nao Cotista', inplace=True)
        dados_relacao['Escola Pública?'] = dados_relacao['Escola Pública?'].astype(str).str.strip().replace(['-', 'nan', 'NaN', 'None'], pd.NA)
        dados_relacao['Escola Pública?'].fillna('Escola Pública', inplace=True)
        dados_relacao['Coeficiente'] = pd.to_numeric(dados_relacao['Coeficiente'], errors='coerce')
        dados_relacao['Coeficiente'].fillna(dados_relacao['Coeficiente'].mean(), inplace=True)
        dados_relacao['Nota Enem'].fillna(dados_relacao['Nota Enem'].mean(), inplace=True)
        return dados_relacao

    def criar_dataframe_historico(historico):
        dados_historico = historico[['id', 'Freq.(%)']].copy()
        dados_historico['Freq.(%)'].fillna(dados_historico['Freq.(%)'].mean(), inplace=True)
        dados_historico = dados_historico.groupby('id')['Freq.(%)'].mean().reset_index()
        dados_historico.rename(columns={'Freq.(%)': 'Frequencia Media'}, inplace=True)
        return dados_historico

    def criar_dataframe_socioeconomico(socioeconomico):
        transporte = socioeconomico[socioeconomico["Cod. Questão"] == 16].copy()
        dados_transporte = transporte[['id', 'Resposta']].copy()
        dados_transporte.rename(columns={'Resposta': 'Transporte'}, inplace=True)
        dados_transporte['Transporte'].fillna('Não informado', inplace=True)
        return dados_transporte

    df_relacao = criar_dataframe_relacao_alunos(relacao_alunos)
    df_historico = criar_dataframe_historico(historico)
    df_socioeconomico = criar_dataframe_socioeconomico(socioeconomico)
    df_intermediario = pd.merge(df_relacao, df_historico, on='id', how='left')
    df = pd.merge(df_intermediario, df_socioeconomico, on='id', how='left')
    df['Frequencia Media'].fillna(df['Frequencia Media'].mean(), inplace=True)
    df['Transporte'].fillna('Não informado', inplace=True)

    # Pré-processamento Final
    target_column = 'Situação Atual do Aluno'
    le_situacao = LabelEncoder()
    df[target_column] = le_situacao.fit_transform(df[target_column].astype(str))

    colunas_categoricas_features = ['Sigla Cota', 'Escola Pública?', 'Sexo', 'Estado', 'Transporte']
    for col in colunas_categoricas_features:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
    colunas_para_normalizar = [col for col in df.columns if col not in ['id', target_column]]
    
    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

    X = df.drop(columns=['id', target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le_situacao


def avaliar_modelo_supervisionado(modelo, nome_modelo, X_test, y_test, label_encoder):
    """
    Avalia um modelo de classificação (supervisionado).
    """
    print(f"--- Avaliando: {nome_modelo} ---")
    previsoes = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, previsoes)
    print(f"Acurácia: {acuracia:.2%}\n")

    print("Relatório de Classificação:")
    nomes_classes = label_encoder.classes_
    codigos_classes = label_encoder.transform(nomes_classes)
    print(classification_report(y_test, previsoes, labels=codigos_classes, target_names=nomes_classes, zero_division=0))
    print("-" * 50)
    return acuracia


def treinar_e_avaliar_kmeans(X_train, X_test, y_test, label_encoder):
    """
    Treina e analisa o algoritmo K-Means.
    """
    print("\n--- Avaliando: K-Means (Agrupamento Não Supervisionado) ---")
    n_clusters = len(label_encoder.classes_)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    clusters_preditos = kmeans.predict(X_test)
    
    # Chama a função para plotar o gráfico do K-Means
    plotar_analise_kmeans(y_test, clusters_preditos, label_encoder)


def plotar_grafico_comparativo(resultados):
    """
    Plota um gráfico de barras para modelos supervisionados.
    """
    df_resultados = pd.DataFrame(list(resultados.items()), columns=['Modelo', 'Acurácia']).sort_values('Acurácia', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Acurácia', y='Modelo', data=df_resultados, palette='viridis', orient='h')

    for index, value in enumerate(df_resultados['Acurácia']):
        plt.text(value, index, f' {value:.2%}', va='center', fontsize=11)

    plt.xlabel('Acurácia', fontsize=12)
    plt.ylabel('Modelo', fontsize=12)
    plt.title('Comparação de Acurácia entre Modelos Supervisionados', fontsize=16)
    plt.xlim(0, 1.05)
    plt.tight_layout()
    
    if not os.path.exists('graficos'):
        os.makedirs('graficos')
        
    plt.savefig('graficos/comparacao_modelos_acuracia.png')
    print("\nGráfico 'graficos/comparacao_modelos_acuracia.png' salvo com sucesso!")

## NOVA FUNÇÃO PARA PLOTAR O GRÁFICO DO K-MEANS
def plotar_analise_kmeans(y_test, clusters_preditos, label_encoder):
    """
    Cria e salva um mapa de calor (heatmap) da análise do K-Means.
    """
    df_analise = pd.DataFrame({'Situação Real': y_test, 'Cluster Atribuído': clusters_preditos})
    df_analise['Situação Real'] = label_encoder.inverse_transform(df_analise['Situação Real'])
    
    tabela_cruzada = pd.crosstab(df_analise['Situação Real'], df_analise['Cluster Atribuído'])
    
    print("Análise dos Clusters vs. Situação Real do Aluno:")
    print(tabela_cruzada)
    print("\nProcure por um cluster (coluna) que concentre um tipo específico de aluno (linha), como 'Evadido'.")
    print("-" * 50)

    # Criação do Gráfico (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(tabela_cruzada, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Mapa de Calor: Situação Real vs. Cluster K-Means', fontsize=16)
    plt.ylabel('Situação Real do Aluno', fontsize=12)
    plt.xlabel('Cluster Atribuído pelo K-Means', fontsize=12)
    plt.tight_layout()

    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    plt.savefig('graficos/analise_kmeans_heatmap.png')
    print("Gráfico 'graficos/analise_kmeans_heatmap.png' salvo com sucesso!")


def main():
    X_train, X_test, y_train, y_test, le_situacao = carregar_e_preparar_dados()

    if X_train is not None:
        modelos_supervisionados = {
            "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
            "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
            "KNN (5 Vizinhos)": KNeighborsClassifier(n_neighbors=5)
        }

        resultados_finais = {}

        for nome, modelo_obj in modelos_supervisionados.items():
            modelo_obj.fit(X_train, y_train)
            acuracia = avaliar_modelo_supervisionado(modelo_obj, nome, X_test, y_test, le_situacao)
            resultados_finais[nome] = acuracia
        
        plotar_grafico_comparativo(resultados_finais)

        treinar_e_avaliar_kmeans(X_train, X_test, y_test, le_situacao)

if __name__ == "__main__":
    main()