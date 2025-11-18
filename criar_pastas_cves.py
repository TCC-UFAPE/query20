import os
import pandas as pd
import re
import requests
import time
import base64
import json

GITHUB_TOKEN = ""
GITHUB_API_BASE = "https://api.github.com"
BASE_URL = "https://vulnerabilityhistory.org/api"

try:
    with open("config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
        GITHUB_TOKEN = config.get("token_github", "")
except:
    pass 

def get_github_headers():
    """Retorna headers para requisições ao GitHub"""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'vuln-history-script/1.0'
    }
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    return headers


def sanitize_folder_name(cve_name):
    """Sanitiza o nome do CVE para criar um nome de pasta válido"""
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', cve_name)
    return sanitized.strip()


def sanitize_file_path(file_path):
    """Sanitiza o caminho do arquivo para criar uma estrutura válida no Windows"""
    sanitized = file_path.replace(':', '_').replace('*', '_').replace('?', '_')
    sanitized = sanitized.replace('"', '_').replace('<', '_').replace('>', '_')
    sanitized = sanitized.replace('|', '_')
    return sanitized


def get_commit_hashes_from_vulnerability(cve, session):
    """Extrai hashes de commits de uma vulnerabilidade específica"""
    try:
        events_response = session.get(f"{BASE_URL}/vulnerabilities/{cve}/events", timeout=30)
        events_response.raise_for_status()
        events = events_response.json()
        
        commit_hashes = []
        seen_hashes = set()  # Para evitar duplicação
        
        for event in events:
            event_type = event.get('event_type', '')
            if event_type in ['fix', 'vcc']:
                description = event.get('description', '')
                commit_match = re.search(r'/commits/([a-f0-9]{40})', description)
                if commit_match:
                    commit_hash = commit_match.group(1)
                    if commit_hash not in seen_hashes:
                        commit_hashes.append(commit_hash)
                        seen_hashes.add(commit_hash)
        
        return commit_hashes
    except:
        return []


def download_file_content_after_commit(repo_full_name, commit_hash, file_path, session):
    """
    Baixa o conteúdo COMPLETO de um arquivo APÓS o commit (versão modificada)
    Retorna o conteúdo do arquivo sem sinais de diff
    """
    try:
        headers = get_github_headers()
        # Buscar o arquivo no estado após o commit
        url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/contents/{file_path}?ref={commit_hash}"
        response = session.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'content' in data and data.get('encoding') == 'base64':
                # Decodificar o conteúdo base64
                content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                return content, data.get('size', 0)
        
        return None, 0
    except Exception as e:
        return None, 0


def process_commit_files(repo_full_name, commit_hash, commit_folder, session):
    """
    Processa e baixa arquivos de um commit específico
    Baixa a versão COMPLETA dos arquivos após o commit (SEM + e -)
    
    Args:
        repo_full_name: Nome do repositório (owner/repo)
        commit_hash: Hash do commit
        commit_folder: Pasta onde salvar os arquivos
        session: Sessão de requests
    
    Returns:
        Número de arquivos baixados
    """
    try:
        headers = get_github_headers()
        url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{commit_hash}"
        response = session.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return 0
        
        commit_data = response.json()
        files = commit_data.get('files', [])
        
        if not files:
            return 0
        
        files_downloaded = 0
        
        # Processar cada arquivo
        for file_info in files:
            filename = file_info.get('filename', '')
            if not filename:
                continue
            
            status = file_info.get('status', 'modified')
            
            # Criar estrutura de diretórios
            sanitized_path = sanitize_file_path(filename)
            file_full_path = os.path.join(commit_folder, sanitized_path)
            os.makedirs(os.path.dirname(file_full_path), exist_ok=True)
            
            # Se o arquivo foi deletado
            if status == 'removed':
                with open(file_full_path + '.DELETED', 'w', encoding='utf-8') as f:
                    f.write(f"Este arquivo foi DELETADO no commit {commit_hash[:8]}\n\n")
                    f.write(f"Arquivo: {filename}\n")
                    f.write(f"Commit: {commit_hash}\n")
                files_downloaded += 1
                print(f"         {filename} (deletado)")
                continue
            
            # Baixar o conteúdo COMPLETO do arquivo após o commit
            file_content, file_size = download_file_content_after_commit(
                repo_full_name, commit_hash, filename, session
            )
            
            if file_content:
                # Salvar o arquivo completo (SEM sinais de diff)
                with open(file_full_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(file_content)
                
                files_downloaded += 1
                print(f"         {filename}")
            else:
                print(f"         {filename} (não foi possível baixar)")
            
            time.sleep(0.1)  # Pausa entre arquivos
        
        return files_downloaded
        
    except Exception as e:
        print(f"         Erro ao processar commit: {e}")
        return 0


def create_cve_folders_from_excel(excel_file="5_analise_completa_tokens.xlsx", base_folder="CVEs"):
    """
    Lê o arquivo Excel e cria uma pasta para cada CVE
    Baixa os arquivos COMPLETOS (sem sinais de diff) de cada commit
    """
    
    if not os.path.exists(excel_file):
        print(f"[ERRO] Arquivo '{excel_file}' não encontrado!")
        return
    
    print(f"\n{'='*70}")
    print(f"  CRIADOR DE PASTAS E DOWNLOAD DE ARQUIVOS DOS CVEs")
    print(f"  (Arquivos completos APÓS os commits - sem + e -)")
    print(f"{'='*70}\n")
    print(f"Analisando arquivo: {excel_file}\n")
    
    try:
        df = pd.read_excel(excel_file)
        print(f"Arquivo carregado com sucesso!")
        print(f"  • Total de linhas: {len(df)}")
        print(f"  • Colunas: {list(df.columns)}\n")
        
        if 'CVE' not in df.columns or 'Repositório GitHub' not in df.columns:
            print("[ERRO] Colunas necessárias não encontradas!")
            return
        
        # Criar pasta base
        os.makedirs(base_folder, exist_ok=True)
        print(f"Pasta base '{base_folder}' pronta\n")
        
        session = requests.Session()
        
        total_cves = len(df)
        total_folders_created = 0
        total_files_downloaded = 0
        
        print(f"{'='*70}")
        print(f"  PROCESSANDO {total_cves} CVE(s)")
        print(f"{'='*70}\n")
        
        for idx, row in df.iterrows():
            cve = row.get('CVE', 'N/A')
            repo = row.get('Repositório GitHub', 'N/A')
            
            if pd.isna(cve) or cve == 'N/A':
                continue
            
            print(f"[{idx + 1}/{total_cves}] Processando {cve}...")
            
            # Criar pasta do CVE
            folder_name = sanitize_folder_name(str(cve))
            cve_folder = os.path.join(base_folder, folder_name)
            
            if not os.path.exists(cve_folder):
                os.makedirs(cve_folder)
                total_folders_created += 1
            
            # Criar README com informações do CVE
            create_readme_for_cve(row, cve_folder)
            
            # Se tiver repositório, baixar arquivos dos commits
            if pd.notna(repo) and repo != 'N/A':
                print(f"   Repositório: {repo}")
                
                # Buscar commits relacionados
                commit_hashes = get_commit_hashes_from_vulnerability(cve, session)
                
                if commit_hashes:
                    print(f"   [{len(commit_hashes)} commit(s) encontrado(s)]")
                    
                    for c_idx, commit_hash in enumerate(commit_hashes, 1):
                        print(f"      [{c_idx}/{len(commit_hashes)}] Commit {commit_hash[:8]}...")
                        
                        # Criar pasta com o hash completo do commit
                        commit_folder = os.path.join(cve_folder, commit_hash)
                        os.makedirs(commit_folder, exist_ok=True)
                        
                        files_count = process_commit_files(
                            repo, commit_hash, commit_folder, session
                        )
                        
                        if files_count > 0:
                            total_files_downloaded += files_count
                            print(f"         [OK] {files_count} arquivo(s) baixado(s)")
                        
                        time.sleep(0.3)  # Pausa entre commits
                else:
                    print(f"   [INFO] Nenhum commit encontrado")
            else:
                print(f"   [INFO] Sem repositório associado")
            
            print()
            time.sleep(0.2)
        
        # Resumo final
        print(f"\n{'='*70}")
        print(f"  RESUMO FINAL")
        print(f"{'='*70}")
        print(f"  • CVEs processados: {total_cves}")
        print(f"  • Pastas criadas: {total_folders_created}")
        print(f"  • Arquivos baixados: {total_files_downloaded}")
        print(f"  • Localização: {os.path.abspath(base_folder)}")
        print(f"{'='*70}\n")
        print("[SUCESSO] Processo concluído!")
        print("\nNOTA: Os arquivos foram baixados na versão APÓS o commit.")
        print("      Não contêm os sinais + e - do diff, são arquivos completos.")
        
    except Exception as e:
        print(f"\n[ERRO FATAL] {e}")


def create_readme_for_cve(row, cve_folder):
    """Cria um arquivo README.md com informações sobre o CVE"""
    try:
        readme_path = os.path.join(cve_folder, "README.md")
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            cve = row.get('CVE', 'N/A')
            f.write(f"# {cve}\n\n")
            f.write(f"## Informações Gerais\n\n")
            
            info_fields = {
                'Projeto': row.get('Projeto', 'N/A'),
                'Repositório GitHub': row.get('Repositório GitHub', 'N/A'),
                'Caracteres Totais (Documentação)': row.get('Caracteres Totais (Documentação)', 0),
                'Tokens Documentação': row.get('Tokens Documentação', 0),
                'Tokens GitHub (Diff)': row.get('Tokens GitHub (Diff)', 0),
                'Total de Tokens': row.get('Total de Tokens', 0),
                'Linhas Adicionadas (GitHub)': row.get('Linhas Adicionadas (GitHub)', 0),
                'Linhas Deletadas (GitHub)': row.get('Linhas Deletadas (GitHub)', 0),
                'Total de Linhas Modificadas': row.get('Total de Linhas Modificadas', 0),
                'Arquivos Modificados (GitHub)': row.get('Arquivos Modificados (GitHub)', 0),
                'Tamanho Total dos Arquivos (bytes)': row.get('Tamanho Total dos Arquivos (bytes)', 0)
            }
            
            for key, value in info_fields.items():
                f.write(f"**{key}:** {value}\n\n")
            
            f.write(f"\n## Estrutura de Arquivos\n\n")
            f.write(f"Esta pasta contém:\n\n")
            f.write(f"- `README.md`: Este arquivo com informações do CVE\n")
            f.write(f"- `[hash do commit]/`: Pastas com o hash completo de cada commit relacionado\n\n")
            f.write(f"### Conteúdo das pastas de commit:\n\n")
            f.write(f"- **Arquivos modificados**: Versão COMPLETA após o commit (sem sinais + e -)\n")
            f.write(f"- `.DELETED`: Marcador para arquivos que foram deletados\n")
            
            f.write(f"\n---\n\n")
            f.write(f"*Dados extraídos de: 5_analise_completa_tokens.xlsx*\n")
            f.write(f"*Os arquivos representam o estado APÓS cada commit*\n")
            
    except Exception as e:
        pass


if __name__ == "__main__":
    create_cve_folders_from_excel()
