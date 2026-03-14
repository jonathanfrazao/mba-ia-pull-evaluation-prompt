# Desafio: Pull, Otimização e Avaliação de Prompts com LangChain e LangSmith

## Sumário

- [Visão Geral](#visão-geral)
- [Técnicas Aplicadas (Fase 2)](#técnicas-aplicadas-fase-2)
- [Resultados Finais](#resultados-finais)
- [Como Executar](#como-executar)
- [Estrutura do Projeto](#estrutura-do-projeto)

---

## Visão Geral

Este projeto implementa um pipeline completo de otimização de prompts usando **LangChain** e **LangSmith**. O objetivo é transformar um prompt de baixa qualidade (`bug_to_user_story_v1`) que converte Bug Reports em User Stories em uma versão otimizada (`bug_to_user_story_v2`) capaz de atingir score ≥ 0.9 em todas as métricas de aprovação.

**Stack utilizada:**
- Linguagem: Python 3.9+
- Framework: LangChain
- Plataforma de avaliação: LangSmith
- LLM principal: `gpt-4o-mini`
- LLM de avaliação: `gpt-4o`

---

## Técnicas Aplicadas (Fase 2)

### 1. Role Prompting

**O que é:** Definir uma persona clara e detalhada para o modelo, estabelecendo autoridade, contexto profissional e estilo de raciocínio esperado.

**Por que escolhi:** O prompt v1 definia apenas uma persona genérica ("assistente que transforma bugs em tarefas para desenvolvedores"), sem autoridade de Product Manager, sem empatia pelo usuário afetado e sem o vocabulário de negócio esperado. A persona afeta diretamente o **Tone Score**, pois força uma linguagem centrada no usuário em vez de linguagem técnica.

**Como apliquei:**

```
Você é um Product Manager sênior com 10 anos de experiência. Sua especialidade
é transformar bugs técnicos em User Stories claras, empáticas e acionáveis,
sempre priorizando a perspectiva e a frustração do usuário afetado.
```

---

### 2. Few-Shot Learning

**O que é:** Fornecer exemplos completos de entrada/saída dentro do prompt para que o modelo aprenda o padrão esperado por demonstração.

**Por que escolhi:** É a técnica com maior impacto para tarefas de formatação estruturada. O v1 gerava User Stories inconsistentes — às vezes adicionava seções extras em bugs simples, às vezes omitia contexto técnico em bugs complexos. Com exemplos cobrindo todos os níveis de complexidade (simples, médio, complexo), o modelo aprende a calibrar o output pelo padrão correto de cada tipo.

**Como apliquei:** Incluí 18 pares Bug → User Story no `system_prompt`, cobrindo:
- **5 bugs simples** (1-2 frases): apenas User Story + critérios BDD, sem seções extras
- **12 bugs médios** (com steps/logs/dados técnicos): User Story + BDD + seção de contexto técnico
- **1 bug complexo** (múltiplos problemas + impacto crítico): formato `=== SEÇÕES ===` com tasks técnicas

Exemplo de par few-shot para bug simples:
```
Bug: Campo de email aceita texto sem @, permitindo cadastros inválidos.

User Story:
Como um usuário criando uma conta, eu quero que o sistema valide meu email
corretamente, para que eu não insira um endereço inválido por engano.

Critérios de Aceitação:
- Dado que estou no formulário de cadastro
- Quando digito um email sem o caractere @
- Então devo ver uma mensagem de erro
- E não devo conseguir prosseguir com o cadastro
- E a mensagem deve explicar o formato correto
```

---

### 3. Chain of Thought (CoT)

**O que é:** Instruir o modelo a realizar etapas de raciocínio antes de gerar a resposta final.

**Por que escolhi:** A principal falha do v1 era escolher o formato errado dependendo da complexidade do bug. Um bug de 1 linha e um bug com 10 problemas identificados exigem outputs completamente diferentes. O CoT força o modelo a classificar a complexidade antes de responder, reduzindo erros de formato.

**Como apliquei:** No `user_prompt`, instruo o modelo a identificar o tipo de bug antes de escrever:

```
Para bugs simples (1-2 frases sem steps, logs ou dados técnicos):
  SOMENTE user story + critérios BDD. Nada mais.
Para bugs com detalhes técnicos:
  inclua TODOS os dados do bug (valores, endpoints, códigos HTTP, percentuais,
  volumes, timings, mensagens de erro) nas seções apropriadas.
Para bugs com PROBLEMAS IDENTIFICADOS numerados ou impacto crítico:
  use o formato === SEÇÕES === com tasks técnicas.
```

---

### 4. Structured Output

**O que é:** Definir explicitamente o esquema de saída — nomes de seções, hierarquia, formatação e regras de quando cada seção deve ou não aparecer.

**Por que escolhi:** User Stories têm um formato de mercado bem definido (`Como um... eu quero... para que...` + BDD `Dado/Quando/Então/E`). O v1 não especificava isso, gerando variações que o avaliador penalizava. Com templates explícitos por nível de complexidade, o modelo segue o formato correto consistentemente.

**Como apliquei:** Defini 3 templates no `system_prompt`:
- **Simples:** `User Story` + `Critérios de Aceitação` apenas
- **Médio:** `User Story` + `Critérios de Aceitação` + seção técnica contextual
- **Complexo:** `=== USER STORY PRINCIPAL ===` + `=== CRITÉRIOS DE ACEITAÇÃO ===` + `=== CRITÉRIOS TÉCNICOS ===` + `=== CONTEXTO DO BUG ===` + `=== TASKS TÉCNICAS SUGERIDAS ===`

---

### 5. Instruction Layering

**O que é:** Organizar as instruções em camadas hierárquicas — regras gerais primeiro, regras específicas depois — para que o modelo processe as diretrizes do mais importante ao mais específico.

**Por que escolhi:** Prompts com instruções misturadas causam conflitos internos no modelo. Ao separar em camadas (persona → regras absolutas → classificação de complexidade → templates → exemplos), cada camada reforça a anterior sem contradição.

**Como apliquei:** O `system_prompt` segue esta ordem:
1. Persona (quem o modelo é)
2. Regras absolutas (o que jamais pode falhar)
3. Classificação de complexidade (como decidir o formato)
4. Templates por nível
5. Exemplos few-shot concretos

---

### 6. Constraint Prompting

**O que é:** Definir restrições explícitas de comportamento — o que o modelo **não deve** fazer — para evitar outputs fora do padrão.

**Por que escolhi:** O v1 não tinha restrições, então o modelo inventava dados, adicionava seções desnecessárias e usava personas genéricas. Restrições explícitas eliminam esses comportamentos sem precisar de exemplos para cada caso negativo.

**Como apliquei:**

```
- Não invente dados que não estão no bug report
- Não omita nenhum dado técnico presente no bug (valores, endpoints, códigos HTTP)
- Nunca use apenas "Como um usuário" sem contexto específico
- Bug simples: zero seções extras além de User Story e Critérios de Aceitação
```

---

## Resultados Finais

### Comparativo v1 vs v2

| Métrica de Aprovação | v1 (reprovado) | v2 (aprovado) | Evolução |
|---|---|---|---|
| **Tone Score** | 0.83 | **0.94** ✓ | +0.11 |
| **Acceptance Criteria Score** | 0.75 | **0.92** ✓ | +0.17 |
| **User Story Format Score** | 0.79 | **0.94** ✓ | +0.15 |
| **Completeness Score** | 0.77 | **0.93** ✓ | +0.16 |
| **Média das 4 métricas** | 0.7834 | **0.9300** ✓ | +0.1466 |

| Métricas Gerais | v1 | v2 |
|---|---|---|
| Helpfulness | 0.85 | **0.95** ✓ |
| Correctness | 0.78 | **0.94** ✓ |
| F1-Score | 0.72 | **0.94** ✓ |
| Clarity | 0.87 | **0.96** ✓ |
| Precision | 0.84 | **0.93** ✓ |

### Evidências no LangSmith

#### Prompt publicado (público)
- https://smith.langchain.com/hub/jonathanfreire/bug_to_user_story_v2

#### Dataset de avaliação
- **Nome:** `prompt-optimization-challenge-resolved-eval`
- **Exemplos:** 20 bugs (complexidade simples, média e complexa)
- Criado automaticamente pelo `evaluate.py` na primeira execução

#### Screenshots das avaliações
Ver pasta `docs/screenshots/` no repositório:
- `evaluate_v1_reprovado.png` — prompt v1 com todas as 4 métricas abaixo de 0.9 (média: 0.7834)
- `evaluate_v2_aprovado.png` — prompt v2 com todas as 4 métricas ≥ 0.9 (média: 0.9300)

#### Tracing detalhado — 3 exemplos

**Trace 1 — Bug médio: Webhook de pagamento** (`run-17d7bb7f-c3c9-4e6e-bc9c-b68367f76c8c`)

- **Métrica avaliada:** Clarity
- **Bug:** Webhook de pagamento aprovado não está sendo chamado (HTTP 500 em `POST /api/webhooks/payment`)
- **User Story gerada:** "Como o sistema de e-commerce, eu quero receber notificações de pagamento aprovado via webhook, para que o status dos pedidos seja atualizado automaticamente após confirmação do pagamento."
- **Score:** 0.90
- **Reasoning do avaliador (gpt-4o):** Resposta bem organizada com seções claras para User Story, Critérios de Aceitação, Contexto Técnico e Tasks. Linguagem simples e direta, sem ambiguidades. Pequena omissão: nome do gateway não especificado, mas não compromete a compreensão geral.

---

**Trace 2 — Bug médio: Webhook de pagamento** (`run-e1ca72cd-d5c6-4302-a3c3-27da557b5135`)

- **Métrica avaliada:** Precision
- **Bug:** Webhook de pagamento aprovado não está sendo chamado (HTTP 500 em `POST /api/webhooks/payment`)
- **User Story gerada:** mesma do Trace 1
- **Score:** 0.90
- **Reasoning do avaliador (gpt-4o):** Sem alucinações — todas as informações são baseadas nos dados fornecidos pelo usuário. Resposta focada no problema. Informações factualmente corretas, exceto pela ausência do nome do gateway, que não foi fornecido no bug report original.

---

**Trace 3 — Bug complexo: Sistema de checkout** (`run-88cbd5ec-8043-4c64-9568-1e2f1344e6b1`)

- **Métrica avaliada:** User Story Format Score
- **Bug:** Sistema de checkout com múltiplas falhas críticas (XSS no cupom, timeout 504 em 30% dos casos, race condition em cupons, loading infinito após timeout). Impacto: 150+ clientes, R$ 15.000 em perdas, rating caiu de 4.5 → 3.2.
- **User Story gerada:** "Como um cliente finalizando minha compra, eu quero um processo de checkout seguro, confiável e com feedback claro..." com seções `=== CRITÉRIOS DE ACEITAÇÃO ===`, `=== CONTEXTO DO BUG ===` e `=== TASKS TÉCNICAS SUGERIDAS ===`
- **Score:** 1.00
- **Reasoning do avaliador (gpt-4o):** Template padrão correto com todas as três partes presentes. Persona "cliente do e-commerce" clara e específica. Ação e benefício bem articulados. Seções claramente separadas facilitando leitura e navegação. Estrutura bem organizada, permitindo fácil compreensão dos requisitos.

### Iterações realizadas

O desafio foi concluído em **1 iteração**, aplicando as 6 técnicas simultaneamente na construção do v2 após análise detalhada dos pontos fracos do v1.

**Análise do v1 que guiou a otimização:**

| Problema identificado no v1 | Técnica aplicada para resolver |
|---|---|
| Tom genérico, sem empatia (Tone: 0.83) | Role Prompting |
| Formato BDD inconsistente (AccCriteria: 0.75) | Structured Output + Few-Shot |
| Seções erradas para cada tipo de bug (Format: 0.79) | Chain of Thought + Constraint Prompting |
| Dados técnicos omitidos (Completeness: 0.77) | Instruction Layering + Few-Shot |

---

## Como Executar

### Pré-requisitos

- Python 3.9+
- Conta no [LangSmith](https://smith.langchain.com) (gratuita)
- API Key da [OpenAI](https://platform.openai.com/api-keys)

### 1. Clonar e configurar o ambiente

```bash
git clone https://github.com/jonathanfreire/desafio-prompt-engineer
cd desafio-prompt-engineer

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env`:

```env
# LangSmith
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=prompt-optimization-challenge

# Seu username no LangSmith Hub
USERNAME_LANGSMITH_HUB=jonathanfreire

# OpenAI
OPENAI_API_KEY=sk-xxxxxxxxxxxx
LLM_PROVIDER=openai
MAIN_MODEL=gpt-4o-mini
EVAL_MODEL=gpt-4o
```

### 3. Pull do prompt v1

```bash
python src/pull_prompts.py
```

Baixa `leonanluppi/bug_to_user_story_v1` para `prompts/raw_prompts.yml`.

### 4. Push do prompt v2 otimizado

```bash
python src/push_prompts.py
```

Publica `prompts/bug_to_user_story_v2.yml` no LangSmith Hub como `bug_to_user_story_v2`.

### 5. Avaliação

```bash
python src/evaluate.py
```

Avalia o prompt contra os 20 exemplos do dataset e exibe os scores das 4 métricas de aprovação.

### 6. Testes de validação

```bash
pytest tests/test_prompts.py -v
```

Saída esperada:
```
tests/test_prompts.py::test_prompt_has_system_prompt       PASSED
tests/test_prompts.py::test_prompt_has_role_definition     PASSED
tests/test_prompts.py::test_prompt_mentions_format         PASSED
tests/test_prompts.py::test_prompt_has_few_shot_examples   PASSED
tests/test_prompts.py::test_prompt_no_todos                PASSED
tests/test_prompts.py::test_minimum_techniques             PASSED

6 passed
```

---

## Estrutura do Projeto

```
desafio-prompt-engineer/
├── .env.example
├── requirements.txt
├── README.md
├── prompts/
│   ├── bug_to_user_story_v1.yml       # Prompt original (após pull)
│   └── bug_to_user_story_v2.yml       # Prompt otimizado
├── datasets/
│   └── bug_to_user_story.jsonl        # 20 exemplos de bugs para avaliação
├── src/
│   ├── pull_prompts.py
│   ├── push_prompts.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── dataset.py
│   └── utils.py
└── tests/
    └── test_prompts.py
```
