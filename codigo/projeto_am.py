import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

def carregar_dados_essenciais(caminho_arquivo):
    """
    Carrega apenas as tabelas 'Relacao_Alunos' e 'Historico'.
    """
    print("Carregando Relacao_Alunos e Historico...")
    try:
        df_alunos = pd.read_excel(caminho_arquivo, sheet_name='Relacao_Alunos')
        df_historico = pd.read_excel(caminho_arquivo, sheet_name='Historico')

        df_alunos.columns = df_alunos.columns.str.strip()
        df_historico.columns = df_historico.columns.str.strip()

        print("Tabelas carregadas.")
        return df_alunos, df_historico
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        exit()

def processar_historico(df_historico):
    """
    Agrega os dados de histórico para criar features resumidas por aluno.
    """
    print("Processando dados do histórico...")
    # Padroniza nomes de colunas para minúsculo para consistência
    df_historico.columns = df_historico.columns.str.lower()
    df_historico.rename(columns={'situação disc.': 'situacao_disc', 'freq.(%)':'freq'}, inplace=True, errors='ignore')
    
    df_historico['nota'] = pd.to_numeric(df_historico['nota'], errors='coerce')
    df_historico['freq'] = pd.to_numeric(df_historico['freq'], errors='coerce')

    df_agg = df_historico.groupby('id').agg(
        MEDIA_NOTAS=('nota', 'mean'),
        MEDIA_FREQUENCIA=('freq', 'mean'),
        TAXA_APROVACAO=('situacao_disc', lambda x: np.mean(x == 'Aprovado')),
        QTD_REPROVACOES_NOTA=('situacao_disc', lambda x: (x == 'Reprovado por Nota').sum()),
        QTD_REPROVACOES_FREQ=('situacao_disc', lambda x: (x == 'Reprovado por Frequência').sum())
    ).reset_index()
    print("Histórico processado.")
    return df_agg

def preparar_dados_finais(df_alunos, df_historico_agg):
    """
    Junta as tabelas, limpa os dados, cria a variável alvo e prepara para análise.
    """
    print("Criando tabela master e limpando os dados...")
    df_alunos.columns = df_alunos.columns.str.lower()
    df_alunos.rename(columns={'situação atual do aluno': 'situacao_atual'}, inplace=True, errors='ignore')

    df_master = pd.merge(df_alunos, df_historico_agg, on='id', how='left')
    
    # --- COLUNA RENOMEADA AQUI ---
    df_master['Evasao'] = (df_master['situacao_atual'] == 'Desistente').astype(int)

    df_master.set_index('id', inplace=True)
    
    colunas_para_remover = [
        '#', 'data nascimento', 'ano desistência', 'período desistências',
        'situacao_atual', 'nome_aluno', 'cidade', 'estado'
    ]
    df_master.drop(columns=[col for col in colunas_para_remover if col in df_master.columns], inplace=True, errors='ignore')

    numeric_cols = ['coeficiente', 'escore vest', 'nota enem']
    for col in numeric_cols:
        if col in df_master.columns:
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    df_master = pd.get_dummies(df_master, drop_first=False, dummy_na=False)
    df_master.fillna(df_master.median(), inplace=True)
    df_master.reset_index(inplace=True)
    
    return df_master

def analisar_e_salvar_resultados(df_analise, pasta_relatorio):
    """
    Realiza a análise de correlação, salva os arquivos e gera o gráfico.
    """
    print("Iniciando análise e gerando arquivos de resultado...")
    os.makedirs(pasta_relatorio, exist_ok=True)
    
    # --- NOME DA COLUNA ATUALIZADO AQUI ---
    if 'EVASAO' not in df_analise.columns:
        print("ERRO: Coluna 'EVASAO' não encontrada para análise.")
        return

    correlacoes = df_analise.drop(columns='id').corr(method='spearman')['EVASAO'].sort_values(ascending=False).drop('EVASAO')
    
    caminho_corr = os.path.join(pasta_relatorio, 'analise_correlacao.csv')
    correlacoes.to_csv(caminho_corr, sep=';', header=['Correlacao'], decimal=',')
    print(f"Análise de correlação salva em: {caminho_corr}")

    print("\n--- Resumo da Análise de Relevância ---")
    print("\n--- FATORES QUE AUMENTAM O RISCO DE EVASÃO ---")
    print(correlacoes.head(10))
    print("\n--- FATORES QUE DIMINUEM O RISCO DE EVASÃO ---")
    print(correlacoes.tail(10))

    top_features = correlacoes.dropna().head(15)._append(correlacoes.dropna().tail(15))
    plt.figure(figsize=(10, 10))
    sns.barplot(x=top_features.values, y=top_features.index, palette='coolwarm', hue=top_features.index, legend=False)
    plt.title('Correlação das Variáveis com a Evasão', fontsize=16)
    plt.xlabel('Correlação de Spearman', fontsize=12)
    plt.ylabel('Variável', fontsize=12)
    plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.05)

    caminho_grafico = os.path.join(pasta_relatorio, 'correlacao_variaveis.png')
    plt.savefig(caminho_grafico)
    print(f"\nGráfico de correlação salvo em: {caminho_grafico}")
    plt.close()


if __name__ == '__main__':
    # Constrói os caminhos de forma dinâmica
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Define os caminhos para as pastas de dados e relatório na raiz do projeto
    PASTA_DADOS = os.path.join(project_root, 'dados')
    PASTA_RELATORIO = os.path.join(project_root, 'relatorio') # Salvará o relatório na raiz

    CAMINHO_ARQUIVO_ENTRADA = os.path.join(PASTA_DADOS, 'base.xlsx')
    
    # Cria a pasta de relatório se não existir
    if not os.path.exists(PASTA_RELATORIO):
        os.makedirs(PASTA_RELATORIO)

    df_alunos, df_historico = carregar_dados_essenciais(CAMINHO_ARQUIVO_ENTRADA)
    df_historico_agg = processar_historico(df_historico)
    df_final = preparar_dados_finais(df_alunos, df_historico_agg)
    analisar_e_salvar_resultados(df_final, PASTA_RELATORIO)
    
    caminho_csv_final = os.path.join(PASTA_RELATORIO, 'df_final_com_id_e_evasao.csv')
    try:
        df_final.to_csv(caminho_csv_final, index=False, sep=';', decimal=',')
        print(f"\n[SUCESSO] O DataFrame final para visualização foi salvo em: '{caminho_csv_final}'")
        print("Ele contém as colunas 'id' e 'EVASAO' como solicitado.")
    except Exception as e:
        print(f"\n[ERRO] Não foi possível salvar o arquivo CSV final: {e}")

    print("\nAnálise concluída.")