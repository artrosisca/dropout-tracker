"""
random_forest.py

Este script realiza a análise preditiva da evasão de alunos utilizando o algoritmo
de Random Forest (Floresta Aleatória).

Estrutura:
1.  Carrega e pré-processa os dados utilizando as funções do 'projeto_am.py'.
2.  Divide os dados em conjuntos de treinamento e teste.
3.  Treina o modelo de Random Forest.
4.  Avalia o modelo e gera métricas de desempenho.
5.  Visualiza a importância das features.
6.  Salva todos os resultados em uma pasta dedicada na raiz do projeto.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # <-- MUDANÇA: Importa o RandomForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Bloco para garantir a importação do módulo de pré-processamento ---
try:
    from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
    print("Módulos de 'projeto_am.py' importados com sucesso.")
except ImportError:
    print("Erro: 'projeto_am.py' não encontrado.")
    sys.exit("Certifique-se de que 'projeto_am.py' está no mesmo diretório.")

# --- Configurações ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
CAMINHO_BASE_ORIGINAL = os.path.join(project_root, 'dados', 'base.xlsx')

# <-- MUDANÇA: Pasta de resultados específica para o Random Forest
PASTA_RELATORIO_RF = os.path.join(project_root, 'resultados_random_forest')
NOME_CLASSES = ['Não Evadiu', 'Evadiu']

def gerar_dados_para_classificacao(caminho_base):
    """
    Orquestra o pré-processamento e retorna os dados prontos para modelagem.
    (Função reutilizada)
    """
    print("\n--- Etapa 1: Pré-processamento dos Dados ---")
    df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
    df_historico_agg = processar_historico(df_historico)
    df_final = preparar_dados_finais(df_alunos, df_historico_agg)

    if 'id' in df_final.columns:
        X = df_final.drop(columns=['Evasao', 'id'])
    else:
        X = df_final.drop(columns=['Evasao'])
        
    y = df_final['Evasao']
    print("Dados pré-processados e separados em features (X) e alvo (y).")
    return X, y

def treinar_e_avaliar_random_forest(X, y): # <-- MUDANÇA: Nome da função
    """
    Divide os dados, treina o modelo Random Forest e faz as predições.
    """
    print("\n--- Etapa 2: Treinamento do Modelo Random Forest ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101, stratify=y)
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # <-- MUDANÇA PRINCIPAL: Trocamos o classificador para RandomForestClassifier
    # n_estimators=100 significa que vamos criar 100 árvores na nossa floresta.
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=101, n_jobs=-1)

    print("Treinando o modelo...")
    modelo_rf.fit(X_train, y_train)

    print("Realizando predições no conjunto de teste...")
    y_pred = modelo_rf.predict(X_test)

    return modelo_rf, X_train, X_test, y_train, y_test, y_pred

def salvar_resultados(y_test, y_pred, modelo, feature_names, pasta_saida):
    """
    Gera e salva o relatório de classificação, matriz de confusão e importância das features.
    """
    print("\n--- Etapa 3: Geração de Relatórios e Gráficos ---")
    # 1. Relatório de Classificação
    report = classification_report(y_test, y_pred, target_names=NOME_CLASSES)
    caminho_report = os.path.join(pasta_saida, 'relatorio_classificacao.txt')
    with open(caminho_report, 'w', encoding='utf-8') as f:
        f.write("Relatório de Classificação - Random Forest\n") # <-- MUDANÇA
        f.write("="*50 + "\n")
        f.write(f"Acurácia Global: {accuracy_score(y_test, y_pred):.2%}\n\n")
        f.write(report)
    print(f"Relatório de classificação salvo em: {caminho_report}")

    # 2. Matriz de Confusão
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                  xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES)
    plt.title('Matriz de Confusão - Random Forest') # <-- MUDANÇA
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    caminho_matriz = os.path.join(pasta_saida, 'matriz_confusao.png')
    plt.savefig(caminho_matriz)
    plt.close()
    print(f"Matriz de confusão salva em: {caminho_matriz}")

    # 3. Importância das Features (Funciona igual à Árvore de Decisão)
    importancias = pd.Series(modelo.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importancias.head(15), y=importancias.head(15).index, palette='viridis', hue=importancias.head(15).index, legend=False)
    plt.title('Top 15 Variáveis Mais Importantes')
    plt.xlabel('Importância')
    plt.ylabel('Variável')
    plt.tight_layout()
    caminho_importancia = os.path.join(pasta_saida, 'importancia_features.png')
    plt.savefig(caminho_importancia)
    plt.close()
    print(f"Gráfico de importância das features salvo em: {caminho_importancia}")

# A função de PDP também funciona aqui, então a mantemos.
def salvar_grafico_pdp(modelo, X_train, features_para_plotar, pasta_saida):
    """
    Cria e salva Gráficos de Dependência Parcial (PDP).
    """
    print("\n--- Etapa 4: Gerando Gráficos de Dependência Parcial (PDP) ---")
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        display = PartialDependenceDisplay.from_estimator(
            modelo, X_train, features=features_para_plotar,
            kind="average", ax=ax
        )
        fig.suptitle("Relação entre Variáveis e a Probabilidade de Evasão (PDP)")
        plt.tight_layout()
        caminho_pdp = os.path.join(pasta_saida, f'pdp_plot.png')
        plt.savefig(caminho_pdp)
        plt.close()
        print(f"Gráfico PDP salvo em: {caminho_pdp}")
    except Exception as e:
        print(f"Não foi possível gerar o gráfico PDP. Erro: {e}")

if __name__ == '__main__':
    if not os.path.exists(PASTA_RELATORIO_RF):
        os.makedirs(PASTA_RELATORIO_RF)

    X, y = gerar_dados_para_classificacao(CAMINHO_BASE_ORIGINAL)
    
    modelo, X_train, X_test, y_train, y_test, y_pred = treinar_e_avaliar_random_forest(X, y)
    
    salvar_resultados(y_test, y_pred, modelo, X.columns, PASTA_RELATORIO_RF)
    
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    features_pdp = importancias.nlargest(2).index.tolist()
    
    salvar_grafico_pdp(modelo, X_train, features_pdp, PASTA_RELATORIO_RF)

    print("\nAnálise preditiva com Random Forest concluída com sucesso!")