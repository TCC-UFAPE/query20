import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

EXCEL_FILE = 'Relatorio_Analise_CVEs_LLM.xlsx'
OUTPUT_PNG = 'comparativo_llms_httpd.png'

# Ler todas as abas do Excel usando header=1 (a segunda linha contém os nomes das colunas nas planilhas geradas)
xls = pd.read_excel(EXCEL_FILE, sheet_name=None, header=1, engine='openpyxl')

# Concatenar todas as abas em um único DataFrame, adicionando coluna 'Modelo_Sheet' com o nome da aba
frames = []
for sheet_name, df in xls.items():
    df = df.copy()
    df['Modelo_Sheet'] = sheet_name
    frames.append(df)

if not frames:
    print('Nenhuma aba encontrada no arquivo Excel.')
    raise SystemExit(1)

all_df = pd.concat(frames, ignore_index=True, sort=False)

# Normalizar nomes das colunas (strip)

all_df.columns = [str(c).strip() for c in all_df.columns]

# Função utilitária para encontrar coluna por palavras-chave
def find_column(columns, keywords):
    cols = list(columns)
    low = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(low):
            if kw in c:
                return cols[i]
    return None

# Detectar colunas dinamicamente
cve_col = find_column(all_df.columns, ['cve'])
files_col = find_column(all_df.columns, ['arquivo', 'arquivo', 'arquivos', 'files'])
vuln_col = find_column(all_df.columns, ['vulnerab', 'vulnerability', 'vuln'])

print('Colunas detectadas:')
print(f'  CVE coluna: {cve_col}')
print(f'  Arquivos coluna: {files_col}')
print(f'  Vulnerabilidade coluna: {vuln_col}')

# Construir máscara para systemd: procurar 'systemd' em CVE ou Arquivos Analisados quando existirem
if cve_col is not None:
    mask_cve = all_df[cve_col].astype(str).str.contains('systemd', case=False, na=False)
else:
    mask_cve = pd.Series([False]*len(all_df))

if files_col is not None:
    mask_files = all_df[files_col].astype(str).str.contains('systemd', case=False, na=False)
else:
    mask_files = pd.Series([False]*len(all_df))

mask = mask_cve | mask_files

httpd_df = all_df[mask].copy()

if httpd_df.empty:
    print('Nenhum CVE relacionado a "systemd" encontrado nas abas. Vou usar todos os dados para gerar um gráfico geral.')
    httpd_df = all_df.copy()

# Normalizar valores da coluna de vulnerabilidade
vuln_col = None
for candidate in ['Vulnerabilidade Detectada', 'Vulnerabilidade?', 'Vulnerabilidade']:
    if candidate in httpd_df.columns:
        vuln_col = candidate
        break

if vuln_col is None:
    raise SystemExit('Coluna de vulnerabilidade não encontrada nas abas.')

httpd_df[vuln_col] = httpd_df[vuln_col].astype(str).str.upper().str.strip()

# Classificar respostas simples: YES, NO, ERROR, N/A, OUTROS
def normalize_vuln(x):
    if x in ('YES', 'Y', 'TRUE', 'SIM'):
        return 'YES'
    if x in ('NO', 'N', 'FALSE', 'NAO', 'NÃO'):
        return 'NO'
    if x.startswith('ERROR') or x == 'ERROR':
        return 'ERROR'
    if x in ('N/A', 'NA', 'N A', ''):
        return 'N/A'
    return x

httpd_df['Vuln_Normalizada'] = httpd_df[vuln_col].apply(normalize_vuln)

# Agrupar por modelo e status
group = httpd_df.groupby(['Modelo_Sheet', 'Vuln_Normalizada']).size().reset_index(name='count')

# Pivot para plot
pivot = group.pivot(index='Modelo_Sheet', columns='Vuln_Normalizada', values='count').fillna(0)

# Ordenar modelos por total de análises
pivot['total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('total', ascending=False)

# Plot stacked bar
plt.figure(figsize=(12, max(4, len(pivot)*0.6)))
colors = sns.color_palette('tab10', n_colors=len(pivot.columns.drop('total')))
cols_to_plot = [c for c in pivot.columns if c != 'total']

pivot[cols_to_plot].plot(kind='bar', stacked=True, color=colors, width=0.7)
plt.title('Comparativo de Detecções por LLM (systemd)')
plt.ylabel('Quantidade de análises')
plt.xlabel('Modelo LLM')
plt.legend(title='Status')
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f'Gráfico salvo em: {OUTPUT_PNG}')
