import subprocess
import os

def executar_script(nome_script):
    print(f"\n{'='*50}")
    print(f"Executando {nome_script}...")
    print(f"{'='*50}\n")
    resultado = subprocess.run(['python', nome_script], capture_output=True, text=True)
    print(resultado.stdout)
    if resultado.stderr:
        print(f"ERROS:\n{resultado.stderr}")
    return resultado.returncode == 0

def main():
    # Verifica se os arquivos existem
    scripts = ['criarAlunosUnificado.py', 'criarAlunosNormalizado.py', 'avaliarModelos.py']
    for script in scripts:
        if not os.path.exists(script):
            print(f"Erro: O arquivo {script} não foi encontrado.")
            return
    
    # Executa os scripts em sequência
    if executar_script('criarAlunosUnificado.py'):
        print("\nArquivo alunos_unificado.xlsx criado com sucesso!")
        if executar_script('criarAlunosNormalizado.py'):
            print("\nArquivo alunos_normalizado.xlsx criado com sucesso!")
            if executar_script('avaliarModelos.py'):
                print("\nAvaliação dos modelos concluída com sucesso!")
                print("\nTodos os scripts foram executados com sucesso!")

if __name__ == "__main__":
    main()