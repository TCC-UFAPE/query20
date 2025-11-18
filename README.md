# Query20 - Análise Completa de Vulnerabilidades do systemd

## Descrição

A **Query20** é um script Python que combina as funcionalidades das queries 7 e 10 para realizar uma análise abrangente das vulnerabilidades do projeto **systemd**, integrando dados de documentação com informações detalhadas dos commits do GitHub.

## Funcionalidades

### Tarefas Executadas

#### 1. Contagem de Vulnerabilidades por Projeto
- Conta o total de vulnerabilidades documentadas
- Identifica vulnerabilidades curadas (com descrição completa)
- **Filtro**: Apenas projeto systemd
- **Arquivo gerado**: `1_2_vulnerabilidades_por_projeto.xlsx`

#### 2. Análise de Vulnerabilidades por Tipo
- Classifica vulnerabilidades por tags/tipos
- Agrupa por projeto
- **Filtro**: Apenas projeto systemd
- **Arquivo gerado**: `3_vulnerabilidades_por_tipo.xlsx`

#### 3. Análise de Vulnerabilidades por Lição
- Identifica lições aprendidas (tags começando com "Lesson:")
- Conta ocorrências por projeto
- **Filtro**: Apenas projeto systemd
- **Arquivo gerado**: `4_vulnerabilidades_por_licao.xlsx`

#### 4. Análise Completa de Tokens (Documentação + GitHub)
Esta é a **funcionalidade principal** que integra dados de múltiplas fontes:

**Dados da Documentação (Query 7):**
- Caracteres totais da documentação
- Tokens (palavras) da descrição e erros

**Dados do GitHub (Query 10):**
- Tokens extraídos dos diffs dos commits
- Linhas adicionadas e deletadas
- Número de arquivos modificados
- Tamanho total dos arquivos modificados (em bytes)

**Arquivo gerado**: `5_analise_completa_tokens.xlsx`

### Colunas do Arquivo Principal (Tarefa 4)

O arquivo `5_analise_completa_tokens.xlsx` contém:

| Coluna | Descrição |
|--------|-----------|
| Projeto | Nome do projeto (systemd) |
| CVE | Identificador da vulnerabilidade |
| Repositório GitHub | URL do repositório no GitHub |
| Caracteres Totais (Documentação) | Total de caracteres na descrição |
| Tokens Documentação | Palavras na documentação |
| Tokens GitHub (Diff) | Palavras extraídas dos diffs |
| Total de Tokens | Soma de tokens da documentação e GitHub |
| Linhas Adicionadas (GitHub) | Linhas de código adicionadas |
| Linhas Deletadas (GitHub) | Linhas de código removidas |
| Total de Linhas Modificadas | Soma de adições e deleções |
| Arquivos Modificados (GitHub) | Quantidade de arquivos alterados |
| Tamanho Total dos Arquivos (bytes) | Tamanho real dos arquivos via API |

## Requisitos

### Dependências Python
```bash
pip install requests pandas openpyxl
```

### Token do GitHub
É necessário um token de acesso pessoal do GitHub para fazer requisições à API:

```python
GITHUB_TOKEN = "seu_token_aqui"
```

### Aviso sobre API paga

- **Importante**: este projeto agora utiliza a API paga do Grop para funcionalidades de processamento/integração LLM. Você precisa fornecer sua chave de API do Grop antes de executar o script. A chave pode ser colocada em `config.json` ou definida como variável de ambiente `GROP_API_KEY`.
- **Custo**: chamadas à API paga podem gerar custos — verifique seu plano e limites com o provedor antes de executar operações em massa.


**Limites da API:**
- Sem token: 60 requisições/hora
- Com token: 5000 requisições/hora

## Como Usar

```bash
python Query20.py
```

## Fluxo de Execução

1. **Busca dados da API** do Vulnerability History
   - Vulnerabilidades
   - Tags/Tipos
   - Informações dos projetos

2. **Filtra vulnerabilidades** do systemd

3. **Executa análises**:
   - Tarefas 1-2: Contagem de vulnerabilidades
   - Tarefas 3-4: Classificação por tipo e lição
   - Tarefa 5: Análise completa com dados do GitHub

4. **Para cada vulnerabilidade**:
   - Extrai commits relacionados via eventos (fix/vcc)
   - Busca dados completos de cada commit na API do GitHub
   - Calcula tokens dos diffs
   - Obtém tamanho real dos arquivos via `contents_url`

5. **Gera arquivos Excel** com resultados consolidados

## Estatísticas Geradas

Ao final da execução, o script exibe:
- Total de tokens (documentação)
- Total de tokens (GitHub)
- Total combinado de tokens
- Total de linhas modificadas
- Total de arquivos modificados
- Tamanho total dos arquivos processados

## Diferencial da Query20

A Query20 se destaca por:
- ✅ **Integração completa** entre documentação textual e código fonte
- ✅ **Análise focada** no projeto systemd
- ✅ **Métricas detalhadas** de tamanho e complexidade
- ✅ **Dados reais** via API do GitHub (não aproximações)
- ✅ **Visão 360°** de cada vulnerabilidade

## Arquivos Gerados

- `1_2_vulnerabilidades_por_projeto.xlsx` - Contagens gerais
- `3_vulnerabilidades_por_tipo.xlsx` - Classificação por tipo
- `4_vulnerabilidades_por_licao.xlsx` - Lições aprendidas
- `5_analise_completa_tokens.xlsx` - **Análise completa integrada**
