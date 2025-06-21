"""
comparar_modelos.py

Este script treina múltiplos modelos e compara suas previsões lado a lado.
Versão final que inclui:
- Modelos supervisionados (Árvore, Regressão Logística, Random Forest).
- K-Means com 3 clusters (abordagem validada).
- Normalização dos dados para todos os modelos.
- Gráficos comparativos de métricas e matrizes de confusão.
- Gráfico de visualização dos clusters do K-Means via PCA.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA  # <-- Importação adicionada
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- Bloco para garantir a importação do módulo de pré-processamento ---
try:
    from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
    print("Módulos de 'projeto_am.py' importados com sucesso.")
except ImportError:
    print("Erro: 'projeto_am.py' não encontrado. Tentando buscar no diretório pai...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    try:
        from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
        print("Módulo 'projeto_am.py' encontrado no diretório pai e importado.")
    except ImportError:
        sys.exit("Certifique-se de que 'projeto_am.py' está no diretório do projeto.")


# --- 1. Carregar e Preparar os Dados ---
print("\n--- Etapa 1: Carregando e preparando os dados ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
caminho_base = os.path.join(project_root, 'dados', 'base.xlsx')

df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
df_historico_agg = processar_historico(df_historico)
df_final = preparar_dados_finais(df_alunos, df_historico_agg)

if 'id' in df_final.columns:
    X = df_final.drop(columns=['Evasao', 'id'])
else:
    X = df_final.drop(columns=['Evasao'])
y = df_final['Evasao']
print("Dados prontos.")

# --- 2. Dividir em Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Dados divididos. Usando {len(X_test)} amostras para o teste.")

# --- 3. Normalizar os dados (IMPORTANTE para K-Means e bom para Regressão Logística) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 4. Treinar todos os modelos ---
print("\n--- Etapa 2: Treinando os modelos ---")
# Modelo 1: Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arvore.fit(X_train_scaled, y_train)

# Modelo 2: Regressão Logística
modelo_logistica = LogisticRegression(random_state=101, max_iter=1000)
modelo_logistica.fit(X_train_scaled, y_train)

# Modelo 3: Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=101, n_jobs=-1)
modelo_rf.fit(X_train_scaled, y_train)

# Modelo 4: K-Means (com 3 clusters, justificado pelo método Elbow)
modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
modelo_kmeans.fit(X_train_scaled)
print("Modelos treinados.")

# --- 5. Obter as predições de cada modelo ---
print("\n--- Etapa 3: Coletando as predições ---")
pred_arvore = modelo_arvore.predict(X_test_scaled)
pred_logistica = modelo_logistica.predict(X_test_scaled)
pred_rf = modelo_rf.predict(X_test_scaled)
pred_clusters_kmeans = modelo_kmeans.predict(X_test_scaled)

# --- 5.1 Mapear clusters do K-Means para as classes de Evasão ---
train_clusters = modelo_kmeans.predict(X_train_scaled)
df_map = pd.DataFrame({'cluster': train_clusters, 'verdadeiro': y_train})
mapa_clusters = df_map.groupby('cluster')['verdadeiro'].agg(lambda x: x.mode()[0]).to_dict()
pred_kmeans = np.vectorize(mapa_clusters.get)(pred_clusters_kmeans)
print("Predições do K-Means (3 clusters) mapeadas para as classes de evasão.")
print(f"Mapeamento K-Means aprendido: {mapa_clusters}")


# --- 6. Gerar Gráficos de Avaliação Comparativa ---
print("\n--- Etapa 4: Construindo os gráficos de comparação ---")
modelos = {
    'Árvore de Decisão': pred_arvore,
    'Regressão Logística': pred_logistica,
    'Random Forest': pred_rf,
    'K-Means (3 Clusters)': pred_kmeans
}

# 6.1 Gráfico de Matrizes de Confusão
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Matrizes de Confusão para Cada Modelo', fontsize=16)
axes = axes.flatten()
for i, (nome, pred) in enumerate(modelos.items()):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
    axes[i].set_title(nome)
    axes[i].set_xlabel('Previsto')
    axes[i].set_ylabel('Verdadeiro')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('relatorios_finais/matrizes_confusao_comparativo.png')
print("Gráfico 'matrizes_confusao_comparativo.png' salvo.")
plt.close()

# 6.2 Gráfico de Métricas de Classificação
metricas = {'Acurácia': [], 'Precisão': [], 'Recall': [], 'F1-Score': []}
for nome, pred in modelos.items():
    metricas['Acurácia'].append(accuracy_score(y_test, pred))
    metricas['Precisão'].append(precision_score(y_test, pred, zero_division=0))
    metricas['Recall'].append(recall_score(y_test, pred, zero_division=0))
    metricas['F1-Score'].append(f1_score(y_test, pred, zero_division=0))
df_metricas = pd.DataFrame(metricas, index=modelos.keys())
df_metricas.plot(kind='bar', figsize=(14, 7), rot=0)
plt.title('Comparação de Métricas de Classificação')
plt.ylabel('Pontuação')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('relatorios_finais/comparacao_metricas_final.png')
print("Gráfico 'comparacao_metricas_final.png' salvo.")
plt.close()


# --- 7. Visualização Detalhada do K-Means (PCA) ---
print("\n--- Etapa 5: Gerando visualização dos clusters K-Means ---")
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Visualização dos Clusters K-Means vs. Realidade (Dados de Teste)', fontsize=16)

# Gráfico 1: Colorido pelos clusters do K-Means
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=pred_clusters_kmeans,
                palette='viridis', alpha=0.8, ax=axes[0], legend='full')
axes[0].set_title('Dados de Teste Coloridos por Cluster K-Means')
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')
axes[0].grid(True)

# Gráfico 2: Colorido pela classe real de evasão
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test,
                palette='coolwarm', alpha=0.8, ax=axes[1], legend='full')
axes[1].set_title('Dados de Teste Coloridos por Evasão Real')
axes[1].set_xlabel('Componente Principal 1')
# Ajusta a legenda para ser mais clara
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles, labels=['Não Evadiu (0)', 'Evadiu (1)'], title='Evasão')
axes[1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('relatorios_finais/visualizacao_clusters_kmeans_teste.png')
print("Gráfico 'visualizacao_clusters_kmeans_teste.png' salvo.")
plt.close()


print("\nAnálise comparativa e visualização concluídas.")