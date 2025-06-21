
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

PASTA_RELATORIO_DECISION_TREE = os.path.join(project_root, 'resultados_decision_tree')
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
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

def salvar_resultados_basicos(y_test, y_pred, modelo, feature_names, pasta_saida):
    """
    Gera e salva o relatório de classificação, matriz de confusão e um gráfico
    APENAS com as features mais importantes (importância > 0).
    """
    print("\n--- Etapa 3: Geração de Relatórios Básicos ---")
    
    # --- Relatório de Classificação (sem alterações) ---
    report = classification_report(y_test, y_pred, target_names=NOME_CLASSES)
    caminho_report = os.path.join(pasta_saida, 'relatorio_classificacao.txt')
    with open(caminho_report, 'w', encoding='utf-8') as f:
        f.write(f"Acurácia Global: {accuracy_score(y_test, y_pred):.2%}\n\n" + report)
    print(f"Relatório de classificação salvo em: {caminho_report}")

    # --- Matriz de Confusão (sem alterações) ---
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES)
    plt.title('Matriz de Confusão'); plt.ylabel('Verdadeiro'); plt.xlabel('Predito')
    caminho_matriz = os.path.join(pasta_saida, 'matriz_confusao.png')
    plt.savefig(caminho_matriz); plt.close()
    print(f"Matriz de confusão salva em: {caminho_matriz}")

    # --- Bloco de Importância das Features (LÓGICA ALTERADA) ---
    importancias = pd.Series(modelo.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    # Filtra para manter apenas as features com importância maior que zero
    importancias_relevantes = importancias[importancias > 0]
    
    print(f"Encontradas {len(importancias_relevantes)} features com importância > 0.")

    plt.figure(figsize=(10, 8))
    # Plota apenas as features que passaram no filtro
    sns.barplot(x=importancias_relevantes, y=importancias_relevantes.index, palette='viridis', hue=importancias_relevantes.index, legend=False)
    
    # O título agora é dinâmico, refletindo o número de features mostradas
    plt.title(f'As {len(importancias_relevantes)} Variáveis Mais Importantes', fontsize=16)
    plt.xlabel('Importância (Gini Importance)')
    plt.ylabel('Variável')
    plt.tight_layout()
    
    caminho_importancia = os.path.join(pasta_saida, 'importancia_features.png')
    plt.savefig(caminho_importancia); plt.close()
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

def salvar_grafico_pdp(modelo, X_train, features_para_plotar, pasta_saida):
    """
    Cria e salva Gráficos de Dependência Parcial (PDP).
    """
    print("\n--- Etapa 6: Gerando Gráficos de Dependência Parcial (PDP) ---")
    fig, ax = plt.subplots(figsize=(10, 5))
    display = PartialDependenceDisplay.from_estimator(
        modelo,
        X_train,
        features=features_para_plotar, # Lista de features para analisar
        kind="average", # Mostra a média das predições
        ax=ax
    )
    fig.suptitle("Relação entre Variáveis e a Probabilidade de Evasão (PDP)")
    plt.tight_layout()
    
    caminho_pdp = os.path.join(pasta_saida, f'pdp_plot.png')
    plt.savefig(caminho_pdp)
    plt.close()
    print(f"Gráfico PDP salvo em: {caminho_pdp}")

def salvar_grafico_perfil_comparativo(modelo, X_test, y_pred, pasta_saida):
    """
    Cria um painel de gráficos de barras, cada um comparando uma feature importante
    entre os grupos previstos como evadidos versus não evadidos.
    """
    print("\n--- Etapa 7: Gerando Gráfico de Perfil Comparativo (Versão Melhorada) ---")
    
    # Pega dinamicamente as 3 features mais importantes
    feature_names = X_test.columns
    importancias = pd.Series(modelo.feature_importances_, index=feature_names)
    features_comparar = importancias.nlargest(3).index.tolist()
    
    print(f"As 3 features mais importantes selecionadas para o gráfico: {features_comparar}")

    # Junta as features de teste com as previsões do modelo
    df_perfil = X_test.copy()
    df_perfil['Previsao'] = y_pred
    df_perfil['Previsao'] = df_perfil['Previsao'].map({0: 'Previsto como Não Evadiu', 1: 'Previsto como Evadiu'})
    
    # --- Lógica de Plotagem Melhorada ---
    # Cria uma figura com 3 subplots (1 linha, 3 colunas)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False) # sharey=False permite eixos Y independentes
    fig.suptitle('Perfil Comparativo: Alunos em Risco vs. Alunos Seguros', fontsize=18)

    # Itera sobre cada feature e cria um gráfico para ela
    for i, feature in enumerate(features_comparar):
        # Desenha o gráfico de barras no subplot correspondente (axes[i])
        sns.barplot(data=df_perfil, x='Previsao', y=feature, ax=axes[i], palette=['#3498db', '#e74c3c'], hue='Previsao', legend=False)
        axes[i].set_title(f'Média de: {feature}', fontsize=14)
        axes[i].set_xlabel('') # Remove o rótulo do eixo X para não poluir
        axes[i].set_ylabel('Valor Médio')
        axes[i].tick_params(axis='x', rotation=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajusta o layout para o título principal caber
    
    caminho_grafico = os.path.join(pasta_saida, 'perfil_comparativo.png')
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico de perfil comparativo detalhado salvo em: {caminho_grafico}")

def salvar_grafico_distribuicao_features(modelo, X_test, y_test, pasta_saida):
    """
    Cria um painel de gráficos de densidade (KDE) para visualizar a distribuição
    das features mais importantes para cada classe (Evadiu vs. Não Evadiu).
    """
    print("\n--- Etapa 9: Gerando Gráfico de Distribuição de Features ---")

    # Pega dinamicamente as 3 features mais importantes do modelo
    feature_names = X_test.columns
    importancias = pd.Series(modelo.feature_importances_, index=feature_names)
    features_plotar = importancias.nlargest(3).index.tolist()

    print(f"As 3 features mais importantes selecionadas para o gráfico: {features_plotar}")

    # Cria um DataFrame temporário com as features e o resultado REAL
    df_dist = X_test[features_plotar].copy()
    df_dist['Evasao_Real'] = y_test.map({0: 'Não Evadiu', 1: 'Evadiu'})

    # Cria uma figura com 3 subplots (1 linha, 3 colunas)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Distribuição dos Grupos por Característica Principal', fontsize=18)

    # Itera para criar um gráfico para cada feature
    for i, feature in enumerate(features_plotar):
        # Desenha o gráfico de densidade no subplot correspondente
        sns.kdeplot(data=df_dist, x=feature, hue='Evasao_Real',
                    ax=axes[i], fill=True, common_norm=False, palette=['#3498db', '#e74c3c'])
        axes[i].set_title(f'Distribuição por: {feature}', fontsize=14)
        axes[i].set_xlabel('Valor da Característica')
        axes[i].set_ylabel('Densidade')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para o título

    caminho_grafico = os.path.join(pasta_saida, 'distribuicao_por_feature.png')
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico de distribuição salvo em: {caminho_grafico}")

if __name__ == '__main__':
    # Cria a pasta de resultados se ela não existir
    if not os.path.exists(PASTA_RELATORIO_DECISION_TREE):
        os.makedirs(PASTA_RELATORIO_DECISION_TREE)

    # 1. Preparar os dados
    X, y = gerar_dados_para_classificacao(CAMINHO_BASE_ORIGINAL)

    # 2. Treinar e avaliar o modelo
    modelo, X_train, X_test, y_train, y_test, y_pred = treinar_e_avaliar_arvore(X, y)
    
    # 3. Salvar relatórios e gráficos de avaliação
    salvar_resultados_basicos(y_test, y_pred, modelo, X.columns, PASTA_RELATORIO_DECISION_TREE)

    # 4. Salvar a visualização da árvore
    visualizar_arvore(modelo, X.columns, PASTA_RELATORIO_DECISION_TREE)

    salvar_grafico_perfil_comparativo(modelo, X_test, y_pred, PASTA_RELATORIO_DECISION_TREE)

    salvar_grafico_distribuicao_features(modelo, X_test, y_test, PASTA_RELATORIO_DECISION_TREE)

    print("\nAnálise preditiva com Árvore de Decisão concluída com sucesso!")
