"""
logistic_regression.py

Este script realiza a análise preditiva da evasão de alunos utilizando o algoritmo
de Regressão Logística.

Estrutura:
1.  Carrega e pré-processa os dados utilizando as funções do 'projeto_am.py'.
2.  Divide os dados em conjuntos de treinamento e teste.
3.  Treina o modelo de Regressão Logística.
4.  Avalia o modelo e gera métricas de desempenho (relatório, matriz de confusão).
5.  Visualiza os coeficientes do modelo (importância das features).
6.  Salva todos os resultados em uma pasta dedicada.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
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

# <-- MUDANÇA: Pasta de resultados específica para este modelo
PASTA_RELATORIO_LOGISTICA = os.path.join(project_root, 'resultados_regressao_logistica')
NOME_CLASSES = ['Não Evadiu', 'Evadiu'] # 0 e 1, respectivamente

def gerar_dados_para_classificacao(caminho_base):
    """
    Orquestra o pré-processamento e retorna os dados prontos para modelagem.
    (Esta função é idêntica à do outro script, reutilização de código!)
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

def treinar_e_avaliar_regressao_logistica(X, y): # <-- MUDANÇA: Nome da função
    """
    Divide os dados, treina o modelo de Regressão Logística e faz as predições.
    """
    print("\n--- Etapa 2: Treinamento do Modelo de Regressão Logística ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101, stratify=y)
    print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

    # <-- MUDANÇA PRINCIPAL: Trocamos o classificador
    # max_iter=1000 ajuda a garantir que o modelo encontre uma solução (convirja).
    modelo_logistica = LogisticRegression(random_state=101, max_iter=1000, solver='liblinear')

    print("Treinando o modelo...")
    modelo_logistica.fit(X_train, y_train)

    print("Realizando predições no conjunto de teste...")
    y_pred = modelo_logistica.predict(X_test)

    return modelo_logistica, X_train, X_test, y_train, y_test, y_pred

def salvar_resultados(y_test, y_pred, modelo, feature_names, pasta_saida):
    """
    Gera e salva o relatório de classificação, matriz de confusão e importância das features.
    """
    print("\n--- Etapa 3: Geração de Relatórios e Gráficos ---")
    # 1. Relatório de Classificação
    report = classification_report(y_test, y_pred, target_names=NOME_CLASSES)
    caminho_report = os.path.join(pasta_saida, 'relatorio_classificacao.txt')
    with open(caminho_report, 'w', encoding='utf-8') as f:
        # <-- MUDANÇA: Título do relatório
        f.write("Relatório de Classificação - Regressão Logística\n")
        f.write("="*50 + "\n")
        f.write(f"Acurácia Global: {accuracy_score(y_test, y_pred):.2%}\n\n")
        f.write(report)
    print(f"Relatório de classificação salvo em: {caminho_report}")

    # 2. Matriz de Confusão
    matriz = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                  xticklabels=NOME_CLASSES, yticklabels=NOME_CLASSES)
    plt.title('Matriz de Confusão - Regressão Logística') # <-- MUDANÇA: Título do gráfico
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    caminho_matriz = os.path.join(pasta_saida, 'matriz_confusao.png')
    plt.savefig(caminho_matriz)
    plt.close()
    print(f"Matriz de confusão salva em: {caminho_matriz}")

    # <-- MUDANÇA CRÍTICA: Importância das features na Regressão Logística (Coeficientes)
    # 3. Coeficientes do Modelo
    # Na Regressão Logística, não temos 'feature_importances_', mas sim 'coef_', que são os coeficientes.
    coeficientes = pd.Series(modelo.coef_[0], index=feature_names).sort_values()
    
    # Seleciona os 10 mais positivos e 10 mais negativos para visualização
    top_coefs = pd.concat([coeficientes.head(10), coeficientes.tail(10)])

    plt.figure(figsize=(12, 10))
    sns.barplot(x=top_coefs, y=top_coefs.index, palette='vlag', hue=top_coefs.index, legend=False)
    plt.title('Coeficientes das Variáveis (Importância)')
    plt.xlabel('Valor do Coeficiente (Impacto na Evasão)')
    plt.ylabel('Variável')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    caminho_importancia = os.path.join(pasta_saida, 'importancia_coeficientes.png')
    plt.savefig(caminho_importancia)
    plt.close()
    print(f"Gráfico de coeficientes salvo em: {caminho_importancia}")

# <-- REMOÇÃO: A função 'visualizar_arvore' foi removida pois não se aplica aqui.

def salvar_grafico_pdp(modelo, X_train, features_para_plotar, pasta_saida):
    """
    Cria e salva Gráficos de Dependência Parcial (PDP).
    (Esta função também é compatível com a Regressão Logística!)
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
    if not os.path.exists(PASTA_RELATORIO_LOGISTICA):
        os.makedirs(PASTA_RELATORIO_LOGISTICA)

    X, y = gerar_dados_para_classificacao(CAMINHO_BASE_ORIGINAL)
    
    modelo, X_train, X_test, y_train, y_test, y_pred = treinar_e_avaliar_regressao_logistica(X, y)
    
    salvar_resultados(y_test, y_pred, modelo, X.columns, PASTA_RELATORIO_LOGISTICA)
    
    # <-- MUDANÇA: Seleciona as features mais importantes com base nos coeficientes para o PDP
    coefs = pd.Series(modelo.coef_[0], index=X.columns)
    features_pdp = coefs.abs().nlargest(2).index.tolist() # Pega as 2 de maior impacto (positivo ou negativo)
    
    salvar_grafico_pdp(modelo, X_train, features_pdp, PASTA_RELATORIO_LOGISTICA)

    print("\nAnálise preditiva com Regressão Logística concluída com sucesso!")