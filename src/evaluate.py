"""
Script para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula métricas:
   - Métricas gerais: F1-Score, Clarity, Precision
   - Métricas específicas (critério de aprovação): Tone, Acceptance Criteria,
     User Story Format, Completeness
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Critério de Aprovação (conforme enunciado do desafio):
- Tone Score >= 0.9
- Acceptance Criteria Score >= 0.9
- User Story Format Score >= 0.9
- Completeness Score >= 0.9
- MÉDIA das 4 métricas >= 0.9
- TODAS as 4 métricas devem estar >= 0.9, não apenas a média

Métricas adicionais (objetivo do desafio):
- F1-Score >= 0.9
- Clarity >= 0.9
- Precision >= 0.9

Suporta múltiplos providers de LLM:
- OpenAI (gpt-4o, gpt-4o-mini)
- Google Gemini (gemini-2.5-flash)

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import (
    evaluate_f1_score,
    evaluate_clarity,
    evaluate_precision,
    evaluate_tone_score,
    evaluate_acceptance_criteria_score,
    evaluate_user_story_format_score,
    evaluate_completeness_score,
)

load_dotenv()

THRESHOLD = 0.9


def get_llm():
    return get_configured_llm(temperature=0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Criando dataset de avaliação: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("Nenhum exemplo carregado do arquivo .jsonl")
        return dataset_name

    print(f"Carregados {len(examples)} exemplos do arquivo {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"Dataset '{dataset_name}' já existe, usando existente")
            return dataset_name
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Dataset de avaliação Bug to User Story",
            )

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            print(f"Dataset criado com {len(examples)} exemplos")
            return dataset_name

    except Exception as e:
        print(f"Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"Puxando prompt do LangSmith Hub: {prompt_name}")
        prompt = hub.pull(prompt_name)
        print(f"Prompt carregado com sucesso")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def _safe_bug_report(example: Any) -> str:
    inputs = getattr(example, "inputs", None) or {}
    return (inputs.get("bug_report") or "").strip()


def _extract_dataset_label(example: Any) -> str:
    """Extrai um identificador curto do bug report para exibição."""
    bug = _safe_bug_report(example)
    # Pega a primeira linha, trunca em 80 caracteres
    first_line = bug.split("\n")[0].strip()
    if len(first_line) > 80:
        first_line = first_line[:77] + "..."
    return first_line


def _sorted_examples(examples: List[Any]) -> List[Any]:
    """Ordenação determinística para reduzir variação entre execuções."""
    return sorted(examples, key=lambda e: _safe_bug_report(e))


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content if hasattr(response, "content") else str(response)

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get("bug_report", inputs.get("question", "N/A"))
        else:
            question = "N/A"

        return {
            "answer": answer,
            "reference": reference,
            "question": question
        }

    except Exception as e:
        print(f"Erro ao avaliar exemplo: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "answer": "",
            "reference": "",
            "question": ""
        }


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\nAvaliando: {prompt_name}")

    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        examples = list(client.list_examples(dataset_name=dataset_name))
        examples = _sorted_examples(examples)
        total = len(examples)
        print(f"   Dataset: {total} exemplos")

        llm = get_llm()

        # Métricas gerais
        f1_scores: List[float] = []
        clarity_scores: List[float] = []
        precision_scores: List[float] = []

        # Métricas específicas (critério de aprovação)
        tone_scores: List[float] = []
        acceptance_scores: List[float] = []
        format_scores: List[float] = []
        completeness_scores: List[float] = []

        print("   Avaliando exemplos...")

        for i, example in enumerate(examples, 1):
            # Exibe identificador do dataset antes da avaliação
            #label = _extract_dataset_label(example)
            #print(f"      --- [{i}/{total}] {label}")

            result = evaluate_prompt_on_example(prompt_template, example, llm)

            if not result["answer"]:
                print(f"      [{i}/{total}] Sem resposta gerada")
                continue

            q = result["question"]
            a = result["answer"]
            r = result["reference"]

            # Métricas gerais
            f1 = evaluate_f1_score(q, a, r)["score"]
            clarity = evaluate_clarity(q, a, r)["score"]
            precision = evaluate_precision(q, a, r)["score"]

            # Métricas específicas para Bug to User Story
            tone = evaluate_tone_score(q, a, r)["score"]
            acc = evaluate_acceptance_criteria_score(q, a, r)["score"]
            fmt = evaluate_user_story_format_score(q, a, r)["score"]
            comp = evaluate_completeness_score(q, a, r)["score"]

            f1_scores.append(f1)
            clarity_scores.append(clarity)
            precision_scores.append(precision)

            tone_scores.append(tone)
            acceptance_scores.append(acc)
            format_scores.append(fmt)
            completeness_scores.append(comp)

            print(
                f"      [{i}/{total}] "
                f"F1:{f1:.2f} Clarity:{clarity:.2f} Precision:{precision:.2f} | "
                f"Tone:{tone:.2f} Acc:{acc:.2f} Format:{fmt:.2f} Comp:{comp:.2f}"
            )

        avg_f1 = _avg(f1_scores)
        avg_clarity = _avg(clarity_scores)
        avg_precision = _avg(precision_scores)

        avg_tone = _avg(tone_scores)
        avg_acceptance = _avg(acceptance_scores)
        avg_format = _avg(format_scores)
        avg_completeness = _avg(completeness_scores)

        avg_helpfulness = (avg_clarity + avg_precision) / 2
        avg_correctness = (avg_f1 + avg_precision) / 2

        avg_approval = _avg([avg_tone, avg_acceptance, avg_format, avg_completeness])

        return {
            "helpfulness": round(avg_helpfulness, 4),
            "correctness": round(avg_correctness, 4),
            "f1_score": round(avg_f1, 4),
            "clarity": round(avg_clarity, 4),
            "precision": round(avg_precision, 4),
            "tone_score": round(avg_tone, 4),
            "acceptance_criteria_score": round(avg_acceptance, 4),
            "user_story_format_score": round(avg_format, 4),
            "completeness_score": round(avg_completeness, 4),
            "approval_average": round(avg_approval, 4),
        }

    except Exception as e:
        print(f"Erro na avaliação: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "helpfulness": 0.0,
            "correctness": 0.0,
            "f1_score": 0.0,
            "clarity": 0.0,
            "precision": 0.0,
            "tone_score": 0.0,
            "acceptance_criteria_score": 0.0,
            "user_story_format_score": 0.0,
            "completeness_score": 0.0,
            "approval_average": 0.0,
        }


def _passes_requirements(scores: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Verifica se TODAS as métricas atendem ao critério do desafio.
    """
    failures: List[str] = []

    if scores["f1_score"] < THRESHOLD:
        failures.append("F1-Score")
    if scores["clarity"] < THRESHOLD:
        failures.append("Clarity")
    if scores["precision"] < THRESHOLD:
        failures.append("Precision")

    if scores["tone_score"] < THRESHOLD:
        failures.append("Tone Score")
    if scores["acceptance_criteria_score"] < THRESHOLD:
        failures.append("Acceptance Criteria Score")
    if scores["user_story_format_score"] < THRESHOLD:
        failures.append("User Story Format Score")
    if scores["completeness_score"] < THRESHOLD:
        failures.append("Completeness Score")
    if scores["approval_average"] < THRESHOLD:
        failures.append("Approval Average (4 métricas)")

    return (len(failures) == 0), failures


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nMétricas Gerais:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=THRESHOLD)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=THRESHOLD)}")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=THRESHOLD)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=THRESHOLD)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=THRESHOLD)}")

    print("\nMétricas Específicas (Critério de Aprovação):")
    print(f"  - Tone Score: {format_score(scores['tone_score'], threshold=THRESHOLD)}")
    print(f"  - Acceptance Criteria Score: {format_score(scores['acceptance_criteria_score'], threshold=THRESHOLD)}")
    print(f"  - User Story Format Score: {format_score(scores['user_story_format_score'], threshold=THRESHOLD)}")
    print(f"  - Completeness Score: {format_score(scores['completeness_score'], threshold=THRESHOLD)}")

    print("\n" + "-" * 50)
    print(f"Média das 4 métricas do critério de aprovação: {scores['approval_average']:.4f}")
    print("-" * 50)

    passed, failures = _passes_requirements(scores)

    if passed:
        print(f"\nSTATUS: APROVADO - Todas as métricas atingiram o mínimo de {THRESHOLD}")
    else:
        print(f"\nSTATUS: REPROVADO - Métricas abaixo do mínimo de {THRESHOLD}")
        print("   Métricas com falha:")
        for f in failures:
            print(f"     - {f}")

    return passed


def main():
    print_section_header("AVALIAÇÃO DE PROMPTS OTIMIZADOS")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")

    print(f"Provider: {provider}")
    print(f"Modelo Principal: {llm_model}")
    print(f"Modelo de Avaliação: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGCHAIN_PROJECT", "prompt-optimization-challenge-resolved")

    jsonl_path = os.getenv("DATASET_JSONL_PATH", "datasets/bug_to_user_story.jsonl")

    if not Path(jsonl_path).exists():
        print(f"Arquivo de dataset não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo existe antes de continuar.")
        return 1

    dataset_name = os.getenv("DATASET_NAME", f"{project_name}-eval")
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS PARA AVALIAR")
    print("=" * 70)
    print("\nEste script irá puxar prompts do LangSmith Hub.")
    print("Certifique-se de ter feito push dos prompts antes de avaliar:")
    print("  python src/push_prompts.py\n")

    prompts_to_evaluate = [
        "bug_to_user_story_v2",
    ]

    approved = 0
    rejected = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)
            passed = display_results(prompt_name, scores)

            if passed:
                approved += 1
            else:
                rejected += 1

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            print(f"\nFalha ao avaliar '{prompt_name}': {e}")
            rejected += 1

            results_summary.append({
                "prompt": prompt_name,
                "scores": {},
                "passed": False
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50 + "\n")

    print(f"Prompts avaliados: {len(prompts_to_evaluate)}")
    print(f"Aprovados: {approved}")
    print(f"Reprovados: {rejected}\n")

    if rejected == 0:
        print("Todos os prompts atingiram os requisitos!")
        print(f"\nConfira os resultados em:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nPróximos passos:")
        print("1. Documente o processo no README.md")
        print("2. Capture screenshots das avaliações")
        print("3. Faça commit e push para o GitHub")
        return 0
    else:
        print("Alguns prompts não atingiram os requisitos.")
        print("\nPróximos passos:")
        print("1. Refatore os prompts com score baixo")
        print("2. Faça push novamente: python src/push_prompts.py")
        print("3. Execute: python src/evaluate.py novamente")
        return 1


if __name__ == "__main__":
    sys.exit(main())