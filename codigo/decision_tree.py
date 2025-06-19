
"""
decision_tree.py

Este script realiza a análise preditiva da evasão de alunos utilizando o algoritmo
de Árvore de Decisão.

Estrutura:
1.  Carrega e pré-processa os dados utilizando as funções do 'projeto_am.py'.
2.  Divide os dados em conjuntos de treinamento e teste.
3.  Treina o modelo de Árvore de Decisão.
4.  Avalia o modelo e gera métricas de desempenho (relatório, matriz de confusão).
5.  Visualiza a árvore de decisão e a importância das features.
6.  Salva todos os resultados em uma pasta dedicada.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

PASTA_RELATORIO_DECISION_TREE = os.path.join(script_dir, 'resultados_decision_tree')
NOME_CLASSES = ['Não Evadiu', 'Evadiu'] # 0 e 1, respectivamente

def gerar_dados_para_classificacao(caminho_base):
    """
    Orquestra o pré-processamento e retorna os dados prontos para modelagem.
    """
    print("\n--- Etapa 1: Pré-processamento dos Dados ---")
    df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
    df_historico_agg = processar_historico(df_historico)
    df_final = preparar_dados_finais(df_alunos, df_historico_agg)

    # Separa as features (X) da variável-alvo (y)
    # Remove tanto a coluna 'Evasao' (alvo) quanto a 'id' (identificador)
    if 'id' in df_final.columns:
        X = df_final.drop(columns=['Evasao', 'id'])
    else:
        X = df_final.drop(columns=['Evasao'])
        
    y = df_final['Evasao']
    print("Dados pré-processados e separados em features (X) e alvo (y).")
    return X, y

def treinar_e_avaliar_arvore(X, y):
    """
    Divide os dados, treina o modelo e faz as predições.
    """
    print("\n--- Etapa 2: Treinamento do Modelo de Árvore de Decisão ---")
    # Divisão em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # Criação do modelo. max_depth ajuda a evitar overfitting e simplifica a árvore.
    # Um bom valor inicial para max_depth é entre 3 e 7.
    modelo_arvore = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Treinamento
    print("Treinando o modelo...")
    modelo_arvore.fit(X_train, y_train)

    # Predição
    print("Realizando predições no conjunto de teste...")
    y_pred = modelo_arvore.predict(X_test)

    return modelo_arvore, X_train, X_test, y_train, y_test, y_pred

def salvar_resultados(y_test, y_pred, modelo, feature_names, pasta_saida):
    """
    Gera e salva o relatório de classificação, matriz de confusão e importância das features.
    """
    print("\n--- Etapa 3: Geração de Relatórios e Gráficos ---")
    # 1. Relatório de Classificação
    report = classification_report(y_test, y_pred, target_names=NOME_CLASSES)
    caminho_report = os.path.join(pasta_saida, 'relatorio_classificacao.txt')
    with open(caminho_report, 'w', encoding='utf-8') as f:
        f.write("Relatório de Classificação - Árvore de Decisão\n")
        f.write("="*50 + "\n")
        f.write(f"Acurácia Global: {accuracy_score(y_test, y_pred):.2%}\n\n")
        f.write(report)
    print(f"Relatório de classificação salvo em: {caminho_report}")

    # 2. Matriz de Confusão
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    caminho_matriz = os.path.join(pasta_saida, 'matriz_confusao.png')
    plt.savefig(caminho_matriz)
    plt.close()
    print(f"Matriz de confusão salva em: {caminho_matriz}")

    # 3. Importância das Features
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

def visualizar_arvore(modelo, feature_names, pasta_saida):
    """
    Gera e salva uma imagem da árvore de decisão treinada.
    """
    print("\n--- Etapa 4: Visualização da Árvore de Decisão ---")
    plt.figure(figsize=(40, 20)) # Tamanho grande para melhor visualização
    plot_tree(modelo,
              feature_names=feature_names,
              class_names=NOME_CLASSES,
              filled=True,
              rounded=True,
              proportion=False,
              precision=2,
              fontsize=10)
    plt.title("Visualização da Árvore de Decisão", fontsize=20)
    caminho_arvore = os.path.join(pasta_saida, 'arvore_de_decisao.png')
    plt.savefig(caminho_arvore)
    plt.close()
    print(f"Visualização da árvore salva em: {caminho_arvore}")


if __name__ == '__main__':
    # Cria a pasta de resultados se ela não existir
    if not os.path.exists(PASTA_RELATORIO_DECISION_TREE):
        os.makedirs(PASTA_RELATORIO_DECISION_TREE)

    # 1. Preparar os dados
    X, y = gerar_dados_para_classificacao(CAMINHO_BASE_ORIGINAL)

    # 2. Treinar e avaliar o modelo
    modelo, X_train, X_test, y_train, y_test, y_pred = treinar_e_avaliar_arvore(X, y)

    # 3. Salvar relatórios e gráficos de avaliação
    salvar_resultados(y_test, y_pred, modelo, X.columns, PASTA_RELATORIO_DECISION_TREE)

    # 4. Salvar a visualização da árvore
    visualizar_arvore(modelo, X.columns, PASTA_RELATORIO_DECISION_TREE)

    print("\nAnálise preditiva com Árvore de Decisão concluída com sucesso!")
