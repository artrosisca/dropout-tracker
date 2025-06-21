"""
comparar_modelos.py

Este script treina múltiplos modelos e compara suas previsões lado a lado
para verificar se eles estão se comportando de forma idêntica ou não.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# --- Bloco para garantir a importação do módulo de pré-processamento ---
try:
    from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
    print("Módulos de 'projeto_am.py' importados com sucesso.")
except ImportError:
    print("Erro: 'projeto_am.py' não encontrado.")
    sys.exit("Certifique-se de que 'projeto_am.py' está no mesmo diretório.")

# --- 1. Carregar e Preparar os Dados ---
# --- 1. Carregar e Preparar os Dados ---
print("\n--- Etapa 1: Carregando e preparando os dados ---")
# Define o caminho para a base de dados de forma robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
caminho_base = os.path.join(project_root, 'dados', 'base.xlsx')

# Agora usa a variável com o caminho correto
df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
df_historico_agg = processar_historico(df_historico)
df_final = preparar_dados_finais(df_alunos, df_historico_agg)

if 'id' in df_final.columns:
    X = df_final.drop(columns=['Evasao', 'id'])
else:
    X = df_final.drop(columns=['Evasao'])
y = df_final['Evasao']
print("Dados prontos.")

# --- 2. Dividir em Treino e Teste (COM O MESMO random_state) ---
# Isto é crucial para uma comparação justa.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Dados divididos. Usando {len(X_test)} amostras para o teste.")

# --- 3. Treinar todos os modelos ---
print("\n--- Etapa 2: Treinando os modelos ---")
# Modelo 1: Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arvore.fit(X_train, y_train)

# Modelo 2: Regressão Logística
modelo_logistica = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
modelo_logistica.fit(X_train, y_train)

# Modelo 3: Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo_rf.fit(X_train, y_train)
print("Modelos treinados.")

# --- 4. Obter as predições de cada modelo ---
print("\n--- Etapa 3: Coletando as predições ---")
pred_arvore = modelo_arvore.predict(X_test)
pred_logistica = modelo_logistica.predict(X_test)
pred_rf = modelo_rf.predict(X_test)

# --- 5. Criar a Tabela de Comparação ---
print("\n--- Etapa 4: Construindo a tabela de comparação ---")
df_comparacao = pd.DataFrame({
    'Verdadeiro': y_test,
    'Pred_Arvore': pred_arvore,
    'Pred_Logistica': pred_logistica,
    'Pred_RandomForest': pred_rf
}, index=X_test.index) # Usamos o índice para identificar os alunos

# Adiciona colunas para ver onde os modelos discordam
df_comparacao['Logistica_vs_RF'] = (df_comparacao['Pred_Logistica'] != df_comparacao['Pred_RandomForest'])
df_comparacao['Arvore_vs_RF'] = (df_comparacao['Pred_Arvore'] != df_comparacao['Pred_RandomForest'])

# --- 6. Mostrar a Prova ---
print("\n\n--- PROVA REAL: COMPARANDO AS PREVISÕES ---")

total_discordancias = df_comparacao['Logistica_vs_RF'].sum()
print(f"\nA Regressão Logística e o Random Forest discordaram em {total_discordancias} dos {len(X_test)} casos.")

if total_discordancias > 0:
    print("\nExemplos de casos onde os modelos DISCORDARAM:")
    # Mostra até 10 exemplos onde as previsões foram diferentes
    print(df_comparacao[df_comparacao['Logistica_vs_RF']].head(10))
else:
    print("\nIncrivelmente, os modelos concordaram em todos os casos. Isso é extremamente raro.")

print("\nAnálise comparativa concluída.")