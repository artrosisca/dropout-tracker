import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Suponha que este seja seu DataFrame final
df = pd.read_excel('alunos_normalizado.xlsx')

# Seleciona as colunas numéricas relevantes
colunas_numericas = [
    'Coeficiente', 'Nota Enem', 'Idade',
]

# Renomeia para melhor visualização no gráfico
df_plot = df[colunas_numericas].rename(columns={
    'Frequencia Media': 'Freq.(%)',
    'Situacao Codificada': 'Situação Atual do Aluno'
})

correlacao = df_plot.corr()

# Cria o heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlação entre Variáveis Numéricas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
