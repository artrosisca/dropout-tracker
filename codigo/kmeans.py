import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os
import sys

# Garante que o diretório do script esteja no path para importação
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
    print("Módulos de 'projeto_am.py' importados com sucesso.")
except ImportError as e:
    print(f"Erro ao importar módulos de 'projeto_am.py': {e}")
    sys.exit("Não foi possível carregar as funções de pré-processamento.")

# --- Configurações ---
CAMINHO_BASE_ORIGINAL = 'dados/base.xlsx'
PASTA_RELATORIO_KMEANS = 'resultados_kmeans'

def gerar_df_final_para_kmeans(caminho_base):
    """
    Chama o pipeline de 'projeto_am.py' para gerar o df_final e o prepara para o K-Means.
    """
    print("\nGerando df_final a partir de 'projeto_am.py' para K-Means...")
    df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
    df_historico_agg = processar_historico(df_historico)
    df_final = preparar_dados_finais(df_alunos, df_historico_agg)
    print("df_final gerado com sucesso.")

    features_para_kmeans = df_final.select_dtypes(include=np.number).drop(
        columns=['id', 'Evasao'], errors='ignore'
    )

    if features_para_kmeans.isnull().sum().sum() > 0:
        print("Atenção: Valores NaN encontrados. Preenchendo com a mediana.")
        features_para_kmeans.fillna(features_para_kmeans.median(), inplace=True)
    else:
        print("Não há valores NaN nas features numéricas. Ótimo!")

    print(f"Dados preparados para K-Means. Número de features: {features_para_kmeans.shape[1]}")
    return features_para_kmeans, df_final[['id', 'Evasao']].copy()

def aplicar_kmeans(X, n_clusters):
    """
    Aplica o K-Means e retorna o modelo, labels, dados escalados e o scaler.
    """
    print(f"\nAplicando K-Means com {n_clusters} clusters...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    print("K-Means aplicado com sucesso.")
    return clusters, X_scaled, scaler, kmeans

def encontrar_melhor_k(X_scaled, max_k, pasta_saida):
    """
    Usa o método Elbow para sugerir o número ótimo de clusters (k).
    """
    print("\nEncontrando o número ótimo de clusters (Método Elbow)...")
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia (Soma dos Quadrados Intra-Cluster)', fontsize=12)
    plt.title('Método Elbow para Definição do K Ideal', fontsize=14)
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    caminho_grafico = os.path.join(pasta_saida, 'elbow_method.png')
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico do Método Elbow salvo em: {caminho_grafico}")

def visualizar_clusters(X_scaled, clusters, df_info, pasta_saida):
    """
    Visualiza os clusters em 2D usando PCA e a distribuição de evasão.
    """
    print("\nGerando visualizações principais dos clusters...")
    n_clusters = len(np.unique(clusters))
    
    # --- GRÁFICO PCA COM LABELS MELHORADOS ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    variancia_explicada = pca.explained_variance_ratio_ * 100

    df_viz = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_viz['Cluster'] = clusters
    df_viz['Evasao'] = df_info['Evasao'].values

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_viz, palette='viridis', s=60, alpha=0.8, legend='full')
    
    titulo_pca = f'Visualização dos Clusters via PCA (k={n_clusters})'
    label_pc1 = f'Componente Principal 1 ({variancia_explicada[0]:.1f}% da variância)'
    label_pc2 = f'Componente Principal 2 ({variancia_explicada[1]:.1f}% da variância)'
    
    plt.title(titulo_pca, fontsize=14)
    plt.xlabel(label_pc1, fontsize=12)
    plt.ylabel(label_pc2, fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(pasta_saida, f'clusters_kmeans_pca_{n_clusters}.png'))
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.countplot(x='Cluster', hue='Evasao', data=df_viz, palette='viridis')
    plt.title(f'Distribuição de Evasão por Cluster (k={n_clusters})', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Contagem de Alunos', fontsize=12)
    plt.xticks(ticks=sorted(df_viz['Cluster'].unique()), labels=[f'Cluster {i}' for i in sorted(df_viz['Cluster'].unique())])
    plt.legend(title='Status do Aluno', labels=['Não Evadiu (0)', 'Evadiu (1)'])
    plt.savefig(os.path.join(pasta_saida, f'evasao_por_cluster_kmeans_{n_clusters}.png'))
    plt.close()
    print(f"Visualizações principais salvas em: {pasta_saida}")

def analisar_perfil_clusters(X_original, clusters, pasta_saida):
    """
    Analisa e descreve o perfil de cada cluster, gerando um boxplot para cada feature.
    """
    print("\nAnalisando o perfil de cada cluster com gráficos detalhados...")
    
    # Cria uma pasta para os gráficos de comparação
    pasta_comparacao = os.path.join(pasta_saida, 'comparacao_features')
    os.makedirs(pasta_comparacao, exist_ok=True)
    
    # Adiciona a coluna de cluster ao dataframe original para facilitar a plotagem
    df_analise_features = X_original.copy()
    df_analise_features['Cluster'] = clusters

    # --- NOVO: GRÁFICOS BOXPLOT PARA CADA FEATURE ---
    for feature in X_original.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_analise_features, palette='viridis')
        plt.title(f'Comparação de "{feature}" entre Clusters', fontsize=14)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        caminho_boxplot = os.path.join(pasta_comparacao, f'boxplot_{feature}.png')
        plt.savefig(caminho_boxplot)
        plt.close()
        
    print(f"Gráficos de comparação de features salvos na pasta: {pasta_comparacao}")

if __name__ == '__main__':
    if not os.path.exists(PASTA_RELATORIO_KMEANS):
        os.makedirs(PASTA_RELATORIO_KMEANS)

    features_kmeans_original, df_info_evasao = gerar_df_final_para_kmeans(CAMINHO_BASE_ORIGINAL)

    if features_kmeans_original.empty:
        sys.exit("Nenhuma feature numérica válida encontrada para clusterização.")

    scaler_main = StandardScaler()
    X_scaled_kmeans = scaler_main.fit_transform(features_kmeans_original)

    encontrar_melhor_k(X_scaled_kmeans, max_k=10, pasta_saida=PASTA_RELATORIO_KMEANS)

    try:
        num_clusters = int(input("\nApós analisar o gráfico do 'cotovelo', digite o número de clusters (k) desejado: "))
    except ValueError:
        print("Valor inválido. Usando k=3 como padrão.")
        num_clusters = 3
        
    print(f"\nUtilizando k={num_clusters} para a análise final...")

    clusters_labels, X_scaled_used, scaler_used, kmeans_model = aplicar_kmeans(features_kmeans_original, num_clusters)
    
    visualizar_clusters(X_scaled_used, clusters_labels, df_info_evasao, PASTA_RELATORIO_KMEANS)

    # A análise de perfil agora gera os gráficos de boxplot
    analisar_perfil_clusters(features_kmeans_original, clusters_labels, PASTA_RELATORIO_KMEANS)

    print("\nProcesso K-Means concluído. Verifique a pasta 'resultados_kmeans' e a subpasta 'comparacao_features'.")