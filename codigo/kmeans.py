import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA # Para visualização em 2D
import os
import sys

# Adiciona o diretório atual ao sys.path para que projeto_am.py possa ser importado
# Isso é importante se projeto_am.py não estiver no mesmo diretório ou em um caminho conhecido
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Importa as funções do seu projeto_am.py
# Assume que projeto_am.py está no mesmo diretório ou em um diretório acessível pelo PYTHONPATH
try:
    from projeto_am import carregar_dados_essenciais, processar_historico, preparar_dados_finais
    print("Módulos de 'projeto_am.py' importados com sucesso.")
except ImportError as e:
    print(f"Erro ao importar módulos de 'projeto_am.py': {e}")
    print("Certifique-se de que 'projeto_am.py' está no mesmo diretório ou no PYTHONPATH.")
    sys.exit("Não foi possível carregar as funções de pré-processamento.")

# --- Configurações ---
CAMINHO_BASE_ORIGINAL = 'dados/base.xlsx' # Caminho para o seu arquivo base.xlsx
PASTA_RELATORIO_KMEANS = 'resultados_kmeans' # Pasta para salvar os resultados do K-Means

def gerar_df_final_para_kmeans(caminho_base):
    """
    Chama as funções de pré-processamento do projeto_am.py para gerar o df_final.
    Adapta o df_final para uso com K-Means (seleciona numéricas, trata NaNs).
    """
    print("\nGerando df_final a partir de 'projeto_am.py' para K-Means...")
    try:
        # Carrega e processa os dados usando suas funções existentes
        df_alunos, df_historico = carregar_dados_essenciais(caminho_base)
        df_historico_agg = processar_historico(df_historico)
        df_final = preparar_dados_finais(df_alunos, df_historico_agg)
        
        print("df_final gerado com sucesso pelo pipeline de 'projeto_am.py'.")

        # Selecionar apenas as colunas numéricas que serão features para o K-Means
        # Excluir 'id' e 'EVASAO' (ou 'Situacao' original, que gera EVASAO)
        features_para_kmeans = df_final.select_dtypes(include=np.number).drop(
            columns=['id', 'EVASAO'], errors='ignore'
        )

        # Tratar NaNs nas features (substituir por mediana) - K-Means não lida com NaNs
        # Embora você diga que a base está tratada, esta é uma verificação de segurança
        if features_para_kmeans.isnull().sum().sum() > 0:
            print("Atenção: Valores NaN encontrados nas features numéricas. Preenchendo com a mediana.")
            for col in features_para_kmeans.columns:
                if features_para_kmeans[col].isnull().any():
                    median_val = features_para_kmeans[col].median()
                    features_para_kmeans[col] = features_para_kmeans[col].fillna(median_val)
                    print(f"  Preenchendo NaN em '{col}' com a mediana: {median_val}")
        else:
            print("Não há valores NaN nas features numéricas. Ótimo!")

        print(f"Dados preparados para K-Means. Número de features: {features_para_kmeans.shape[1]}")
        # Retorna as features para K-Means e também id/EVASAO para análise pós-cluster
        return features_para_kmeans, df_final[['id', 'EVASAO']].copy()

    except Exception as e:
        print(f"Erro ao gerar df_final a partir de 'projeto_am.py': {e}")
        sys.exit("Falha na preparação dos dados. Verifique 'projeto_am.py'.")

def aplicar_kmeans(X, n_clusters):
    """
    Aplica o algoritmo K-Means aos dados fornecidos.
    """
    print(f"\nAplicando K-Means com {n_clusters} clusters...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    print("K-Means aplicado com sucesso.")
    return clusters, X_scaled, scaler, kmeans

def visualizar_clusters(X_scaled, clusters, df_info, n_clusters, pasta_saida):
    """
    Visualiza os clusters em 2D usando PCA e salva os gráficos.
    """
    print("\nGerando visualizações dos clusters...")
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_viz = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_viz['Cluster'] = clusters
    df_viz['EVASAO'] = df_info['EVASAO'].values

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_viz, palette='viridis', legend='full', s=50, alpha=0.7)
    plt.title(f'Clusters K-Means ({n_clusters} clusters) via PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.savefig(os.path.join(pasta_saida, f'clusters_kmeans_pca_{n_clusters}.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', hue='EVASAO', data=df_viz, palette='viridis')
    plt.title(f'Distribuição de Evasão por Cluster K-Means ({n_clusters} clusters)')
    plt.xlabel('Cluster')
    plt.ylabel('Contagem de Alunos')
    plt.xticks(ticks=sorted(df_viz['Cluster'].unique()), labels=[f'Cluster {i}' for i in sorted(df_viz['Cluster'].unique())])
    plt.legend(title='Evasão', labels=['Não Evadiu (0)', 'Evadiu (1)'])
    plt.savefig(os.path.join(pasta_saida, f'evasao_por_cluster_kmeans_{n_clusters}.png'))
    plt.show()
    plt.close()

    print(f"Visualizações salvas em: {pasta_saida}")

def analisar_perfil_clusters(X_original, clusters, n_clusters, pasta_saida):
    """
    Analisa o perfil médio de cada cluster em termos das features originais.
    """
    print("\nAnalisando o perfil médio de cada cluster...")

    df_clustered_original = X_original.copy()
    df_clustered_original['Cluster'] = clusters

    cluster_profiles = df_clustered_original.groupby('Cluster').mean()

    print("\nPerfil Médio de Cada Cluster (Valores Originais):")
    print(cluster_profiles)
    cluster_profiles.to_csv(os.path.join(pasta_saida, f'perfil_clusters_{n_clusters}.csv'), sep=';', decimal=',')
    print(f"Perfil médio dos clusters salvo em: {pasta_saida}")

    num_features_to_plot = min(5, X_original.shape[1])
    for i, feature in enumerate(X_original.columns[:num_features_to_plot]):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=cluster_profiles.index, y=feature, data=cluster_profiles, palette='viridis')
        plt.title(f'Média de {feature} por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(f'Média de {feature}')
        plt.savefig(os.path.join(pasta_saida, f'perfil_cluster_feature_{feature}.png'))
        plt.close()

def encontrar_melhor_k(X_scaled, max_k=10, pasta_saida='resultados_kmeans'):
    """
    Usa o método Elbow para encontrar o número ótimo de clusters (k).
    """
    print("\nEncontrando o número ótimo de clusters (Método Elbow)...")
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('SSE (Soma dos Quadrados das Distâncias Intra-Cluster)')
    plt.title('Método Elbow para K-Means')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.savefig(os.path.join(pasta_saida, 'elbow_method.png'))
    plt.show()
    plt.close()
    print(f"Gráfico do Método Elbow salvo em: {pasta_saida}")
    print("Analise o gráfico para identificar o 'cotovelo', que sugere o k ideal.")

if __name__ == '__main__':
    if not os.path.exists(PASTA_RELATORIO_KMEANS):
        os.makedirs(PASTA_RELATORIO_KMEANS)

    # 1. Gerar o df_final chamando as funções do projeto_am.py
    features_kmeans_original, df_info_evasao = gerar_df_final_para_kmeans(CAMINHO_BASE_ORIGINAL)

    if features_kmeans_original.empty:
        print("Nenhuma feature numérica válida encontrada para clusterização.")
        sys.exit()

    # 2. Escalar os dados
    scaler = StandardScaler()
    X_scaled_kmeans = scaler.fit_transform(features_kmeans_original)

    # 3. Encontrar o K ideal
    print("\nIniciando busca pelo K ideal...")
    encontrar_melhor_k(X_scaled_kmeans, max_k=10, pasta_saida=PASTA_RELATORIO_KMEANS)

    # 4. Definir o número de clusters (AJUSTE ESTE VALOR APÓS ANÁLISE DO ELBOW PLOT)
    num_clusters = 3 # Valor inicial, ajuste conforme o Elbow Plot
    print(f"\nUtilizando {num_clusters} clusters para o K-Means...")

    # 5. Aplicar K-Means
    clusters_labels, X_scaled_kmeans_used, scaler_used, kmeans_model = aplicar_kmeans(features_kmeans_original, num_clusters)

    # 6. Visualizar os clusters e analisar a distribuição de evasão
    visualizar_clusters(X_scaled_kmeans_used, clusters_labels, df_info_evasao, num_clusters, PASTA_RELATORIO_KMEANS)

    # 7. Analisar o perfil de cada cluster
    analisar_perfil_clusters(features_kmeans_original, clusters_labels, num_clusters, PASTA_RELATORIO_KMEANS)

    print("\nProcesso K-Means concluído. Verifique a pasta 'resultados_kmeans' para os gráficos e CSVs.")