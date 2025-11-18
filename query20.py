import os
import requests
import pandas as pd
import time
import re
import json

BASE_URL = "https://vulnerabilityhistory.org/api"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_TOKEN_SOURCE = 'env' if os.environ.get('GITHUB_TOKEN') else None


def load_github_token_from_config():
    """Tenta carregar o token do arquivo `config.json` (somente como fallback)."""
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                # campos comuns possíveis
                return cfg.get('token_github') or cfg.get('github_token') or cfg.get('GITHUB_TOKEN') or ''
    except Exception:
        pass
    return ''


# Se variável de ambiente não definida, tentar carregar do config.json (fallback)
if not GITHUB_TOKEN:
    cfg_token = load_github_token_from_config()
    if cfg_token:
        GITHUB_TOKEN = cfg_token
        GITHUB_TOKEN_SOURCE = 'config'

def get_github_headers():
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'vuln-history-script/1.0'
    }
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    return headers


def request_with_retries(session, method, url, headers=None, timeout=30, max_retries=3, backoff_factor=0.5):
    """Faz uma requisição com retries exponenciais simples em erros de rede e 5xx."""
    attempt = 0
    while True:
        try:
            resp = session.request(method, url, headers=headers, timeout=timeout)
            if 500 <= resp.status_code < 600 and attempt < max_retries:
                attempt += 1
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep_time)
                continue
            return resp
        except requests.exceptions.RequestException:
            if attempt >= max_retries:
                raise
            attempt += 1
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_time)
            continue


def get_all_data():
    print("Iniciando busca de dados da API (isso pode levar um momento)...")
    try:
        session = requests.Session()
        
        print("Buscando todas as vulnerabilidades...")
        vuln_response = session.get(f"{BASE_URL}/vulnerabilities?limit=100000", timeout=180)
        vuln_response.raise_for_status()
        vulnerabilities = vuln_response.json()
        print(f"-> Encontradas {len(vulnerabilities)} vulnerabilidades.")
        
        print("Buscando mapa de tags...")
        tags_response = session.get(f"{BASE_URL}/tags?map=true", timeout=30)
        tags_response.raise_for_status()
        tags_map = tags_response.json()
        print(f"-> Encontradas {len(tags_map)} tags.")
        
        print("Buscando informações dos projetos...")
        projects_response = session.get(f"{BASE_URL}/projects", timeout=30)
        projects_response.raise_for_status()
        projects = projects_response.json()
        print(f"-> Encontrados {len(projects)} projetos.")
        
        # Criar mapeamento de projeto -> repositório GitHub
        project_to_repo = {}
        for project in projects:
            project_name = project.get('name', '')
            git_url_prefix = project.get('git_commit_url_prefix', '')
            
            # Extrair owner/repo da URL do GitHub
            github_match = re.search(r'github\.com/([^/]+/[^/]+)/', git_url_prefix)
            if github_match:
                repo_full_name = github_match.group(1)
                project_to_repo[project_name] = repo_full_name
        
        print(f"-> Mapeados {len(project_to_repo)} projetos para repositórios GitHub.")

        return vulnerabilities, tags_map, project_to_repo

    except requests.exceptions.RequestException as e:
        print(f"\n--- ERRO FATAL AO BUSCAR DADOS DA API: {e} ---")
        return None, None, None


def get_commit_hashes_from_vulnerability(cve, session):
    """Extrai hashes de commits de uma vulnerabilidade específica através dos eventos"""
    try:
        events_response = session.get(f"{BASE_URL}/vulnerabilities/{cve}/events", timeout=30)
        events_response.raise_for_status()
        events = events_response.json()
        
        commit_hashes = []
        for event in events:
            event_type = event.get('event_type', '')
            if event_type in ['fix', 'vcc']:
                description = event.get('description', '')
                commit_match = re.search(r'/commits/([a-f0-9]{40})', description)
                if commit_match:
                    commit_hashes.append(commit_match.group(1))
        
        return commit_hashes
    except requests.exceptions.RequestException:
        return []


def get_github_commit_data(repo_full_name, commit_hash, session):
    """Busca dados completos do commit no GitHub via API REST"""
    try:
        headers = get_github_headers()
        url = f"{GITHUB_API_BASE}/repos/{repo_full_name}/commits/{commit_hash}"
        response = request_with_retries(session, 'GET', url, headers=headers, timeout=30)

        if response.status_code in [401, 403, 404, 422]:
            return None

        response.raise_for_status()
        data = response.json()

        if 'commit' not in data:
            return None

        return data

    except:
        return None


def count_tokens_from_github_commit(commit_data, session):
    """Conta tokens e tamanhos dos arquivos modificados no commit"""
    if not commit_data:
        return 0, 0, 0, 0, 0
    
    stats = commit_data.get('stats', {})
    total_additions = stats.get('additions', 0)
    total_deletions = stats.get('deletions', 0)
    
    total_tokens = 0
    total_file_size = 0
    files = commit_data.get('files', [])
    total_files = len(files)
    
    headers = get_github_headers()
    
    for file_info in files:
        # Buscar tamanho do arquivo via contents_url
        contents_url = file_info.get('contents_url', '')
        if contents_url:
            try:
                contents_response = request_with_retries(session, 'GET', contents_url, headers=headers, timeout=10)
                if contents_response.status_code == 200:
                    contents_data = contents_response.json()
                    total_file_size += contents_data.get('size', 0)
                time.sleep(0.1)
            except:
                pass
        
        # Contar tokens no patch (diff)
        patch = file_info.get('patch', '')
        if patch:
            lines = patch.split('\n')
            for line in lines:
                if line and not line.startswith('@@') and not line.startswith('diff'):
                    clean_line = line[1:] if line and line[0] in ['+', '-', ' '] else line
                    words = clean_line.split()
                    total_tokens += len(words)
    
    return total_additions, total_deletions, total_tokens, total_files, total_file_size


def run_task_1_and_2(vulnerabilities):
    print("\n--- Iniciando Tarefa 1 & 2: Contagem de Vulnerabilidades por Projeto ---")
    print("   [FILTRO ATIVO] Processando apenas projeto systemd")
    
    project_counts = {}
    for vuln in vulnerabilities:
        project_name = vuln.get('project_name', 'N/A')
        
        if project_name not in project_counts:
            project_counts[project_name] = {
                'Projeto': project_name,
                'Vulnerabilidades Totais Documentadas': 0,
                'Vulnerabilidades Curadas': 0
            }
        
        project_counts[project_name]['Vulnerabilidades Totais Documentadas'] += 1
        
        if vuln.get('description'):
            project_counts[project_name]['Vulnerabilidades Curadas'] += 1
            
    df = pd.DataFrame(list(project_counts.values()))
    filename = "1_2_vulnerabilidades_por_projeto.xlsx"
    df.to_excel(filename, index=False)
    print(f"-> Sucesso! Arquivo '{filename}' gerado.")


def run_task_3_and_4(vulnerabilities, tags_map):
    print("\n--- Iniciando Tarefa 3 & 4: Vulnerabilidades por Tipo e por Lição ---")
    print("   [FILTRO ATIVO] Processando apenas projeto systemd")

    types_by_project = {}
    lessons_by_project = {}

    for vuln in vulnerabilities:
        project_name = vuln.get('project_name', 'N/A')
        
        if project_name not in types_by_project:
            types_by_project[project_name] = {}
        if project_name not in lessons_by_project:
            lessons_by_project[project_name] = {}
            
        tag_ids = [str(tag['id']) for tag in vuln.get('tag_json', [])]
        
        for tag_id in tag_ids:
            tag_info = tags_map.get(tag_id)
            if tag_info:
                tag_name = tag_info.get('name', 'Tag Desconhecida')
                
                types_by_project[project_name][tag_name] = types_by_project[project_name].get(tag_name, 0) + 1
                
                if tag_name.startswith('Lesson:'):
                    lesson_name = tag_name.replace('Lesson: ', '').strip()
                    lessons_by_project[project_name][lesson_name] = lessons_by_project[project_name].get(lesson_name, 0) + 1

    df_types = pd.DataFrame.from_dict(types_by_project, orient='index').fillna(0).astype(int)
    df_types = df_types.rename_axis('Projeto').reset_index()
    filename_types = "3_vulnerabilidades_por_tipo.xlsx"
    df_types.to_excel(filename_types, index=False)
    print(f"-> Sucesso! Arquivo '{filename_types}' gerado.")
    
    df_lessons = pd.DataFrame.from_dict(lessons_by_project, orient='index').fillna(0).astype(int)
    df_lessons = df_lessons.rename_axis('Projeto').reset_index()
    filename_lessons = "4_vulnerabilidades_por_licao.xlsx"
    df_lessons.to_excel(filename_lessons, index=False)
    print(f"-> Sucesso! Arquivo '{filename_lessons}' gerado.")


def run_task_5_with_github_tokens(vulnerabilities, project_to_repo):
    print("\n--- Iniciando Tarefa 5: Análise de Texto com Tokens do GitHub ---")
    print("   [FILTRO ATIVO] Processando apenas projeto systemd")
    
    session = requests.Session()
    text_data = []
    
    total_vulns = len(vulnerabilities)
    for idx, vuln in enumerate(vulnerabilities):
        if (idx + 1) % 50 == 0:
            print(f"   Processando vulnerabilidade {idx + 1}/{total_vulns}...")
        
        project_name = vuln.get('project_name', 'N/A')
        cve = vuln.get('cve', 'N/A')
        
        # Análise de texto da documentação (query7)
        description = vuln.get('description', '') or ''
        mistakes = vuln.get('mistakes', '') or ''
        full_text = (description.strip() + " " + mistakes.strip()).strip()
        
        doc_chars = len(full_text)
        doc_tokens = len(full_text.split())
        
        # Dados do GitHub (query10)
        github_tokens = 0
        github_additions = 0
        github_deletions = 0
        github_files = 0
        github_file_size = 0
        
        # Buscar commits relacionados à vulnerabilidade
        repo_full_name = project_to_repo.get(project_name)
        if repo_full_name:
            commit_hashes = get_commit_hashes_from_vulnerability(cve, session)
            
            for commit_hash in commit_hashes:
                github_commit = get_github_commit_data(repo_full_name, commit_hash, session)
                if github_commit:
                    additions, deletions, tokens, files, file_size = count_tokens_from_github_commit(github_commit, session)
                    github_tokens += tokens
                    github_additions += additions
                    github_deletions += deletions
                    github_files += files
                    github_file_size += file_size
                
                # Pausa entre commits
                time.sleep(0.2)
        
        # Criar registro combinado
        if full_text or github_tokens > 0:
            text_data.append({
                'Projeto': project_name,
                'CVE': cve,
                'Repositório GitHub': repo_full_name or 'N/A',
                'Caracteres Totais (Documentação)': doc_chars,
                'Tokens Documentação': doc_tokens,
                'Tokens GitHub (Diff)': github_tokens,
                'Total de Tokens': doc_tokens + github_tokens,
                'Linhas Adicionadas (GitHub)': github_additions,
                'Linhas Deletadas (GitHub)': github_deletions,
                'Total de Linhas Modificadas': github_additions + github_deletions,
                'Arquivos Modificados (GitHub)': github_files,
                'Tamanho Total dos Arquivos (bytes)': github_file_size
            })
        
        # Pausa a cada 20 vulnerabilidades
        if (idx + 1) % 20 == 0:
            time.sleep(0.5)
            
    df = pd.DataFrame(text_data)
    filename = "5_analise_completa_tokens.xlsx"
    df.to_excel(filename, index=False)
    print(f"-> Sucesso! Arquivo '{filename}' gerado com {len(text_data)} vulnerabilidades analisadas.")
    
    # Estatísticas
    if len(text_data) > 0:
        print(f"\n--- Estatísticas Gerais ---")
        print(f"   Total de tokens (documentação): {df['Tokens Documentação'].sum():,}")
        print(f"   Total de tokens (GitHub): {df['Tokens GitHub (Diff)'].sum():,}")
        print(f"   Total combinado: {df['Total de Tokens'].sum():,}")
        print(f"   Total de linhas modificadas: {df['Total de Linhas Modificadas'].sum():,}")
        print(f"   Total de arquivos modificados: {df['Arquivos Modificados (GitHub)'].sum():,}")
        print(f"   Tamanho total dos arquivos: {df['Tamanho Total dos Arquivos (bytes)'].sum():,} bytes")


if __name__ == "__main__":
    if not GITHUB_TOKEN:
        print("\n[AVISO] Nenhum token do GitHub foi encontrado (variável de ambiente `GITHUB_TOKEN` ou `config.json`).")
        print("   O script continuará, mas a análise do GitHub pode estar limitada por rate limits ou falta de acesso a commits privados.")
        print("   Para habilitar a integração completa, defina a variável de ambiente `GITHUB_TOKEN` com um token pessoal, ou adicione 'token_github' em `config.json` (não recomendado para produção).")
    else:
        if GITHUB_TOKEN_SOURCE == 'env':
            print("\n[INFO] Token do GitHub carregado da variável de ambiente `GITHUB_TOKEN`.")
        elif GITHUB_TOKEN_SOURCE == 'config':
            print("\n[AVISO] Token do GitHub carregado a partir de `config.json`. Considere usar variável de ambiente e remover o token do arquivo por segurança.")
    
    all_vulnerabilities, all_tags_map, project_to_repo = get_all_data()
    
    if all_vulnerabilities and all_tags_map:
        # Filtrar apenas vulnerabilidades do systemd
        httpd_vulns = [v for v in all_vulnerabilities if v.get('project_name', '').lower() == 'systemd']
        print(f"\n[FILTRO] {len(httpd_vulns)} vulnerabilidades do systemd de {len(all_vulnerabilities)} totais")
        
        if not httpd_vulns:
            print("\n[ERRO] Nenhuma vulnerabilidade do systemd encontrada.")
        else:
            # Executar tarefas da query7 apenas com systemd
            run_task_1_and_2(httpd_vulns)
            time.sleep(1)
            run_task_3_and_4(httpd_vulns, all_tags_map)
            time.sleep(1)
            
            # Executar análise combinada (query7 + query10) apenas com systemd
            run_task_5_with_github_tokens(httpd_vulns, project_to_repo)
            
            print("\n[SUCESSO] Todas as tarefas foram concluídas!")
    else:
        print("\n[ERRO] Não foi possível obter os dados da API. O script não pode continuar.")
