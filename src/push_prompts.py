from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

from utils import load_yaml, print_section_header

load_dotenv()

PROMPTS_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def _has_any_env(*keys: str) -> bool:
    return any(os.getenv(k) for k in keys)


def _require_any_env(keys: List[str]) -> None:
    if not _has_any_env(*keys):
        raise RuntimeError(
            "❌ Variáveis de ambiente ausentes. Configure pelo menos uma:\n"
            f"  - {', '.join(keys)}\n"
            "Dica: copie .env.example para .env e preencha as chaves."
        )


def _get_prompt_root(data: Dict[str, Any], prompt_key: str) -> Dict[str, Any]:
    if prompt_key in data:
        root = data[prompt_key]
    else:
        # fallback: primeiro nó
        first_key = next(iter(data.keys()))
        root = data[first_key]
    if not isinstance(root, dict):
        raise ValueError("Estrutura do YAML inválida: esperado objeto no nó do prompt.")
    return root


def _detect_few_shot(system_prompt: str, user_prompt: str, root: Dict[str, Any]) -> bool:
    """
    Detecta few-shot de forma flexível:
    - campo explícito: few_shot_examples / examples / shots
    - ou presença de marcadores "Exemplo", "Input:", "Output:" etc. no texto
    """
    for key in ("few_shot_examples", "examples", "shots", "few_shots"):
        val = root.get(key)
        if isinstance(val, list) and len(val) >= 1:
            first = val[0]
            if isinstance(first, dict) and any(k in first for k in ("input", "output", "in", "out")):
                return True
            return True

    text = f"{system_prompt}\n{user_prompt}".lower()
    markers = [
        "exemplo",
        "exemplos",
        "input:",
        "output:",
        "entrada:",
        "saída:",
        "saida:",
        "before:",
        "after:",
        "bug:",
        "user story:",
        "acceptance criteria:",
        "given",
        "when",
        "then",
    ]
    score = sum(1 for m in markers if m in text)
    return score >= 2


def validate_prompt(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida o YAML do prompt otimizado.
    Regras:
    - Deve existir system_prompt e user_prompt não vazios
    - metadata.techniques deve ter >= 2
    - não pode conter TODO/[TODO]
    - deve conter indício de few-shot
    """
    errors: List[str] = []

    if not isinstance(data, dict) or not data:
        return False, ["YAML vazio ou inválido"]

    try:
        root = _get_prompt_root(data, PROMPT_KEY)
    except Exception as ex:
        return False, [str(ex)]

    system_prompt = str(root.get("system_prompt", "") or "")
    user_prompt = str(root.get("user_prompt", "") or "")
    description = str(root.get("description", "") or "")

    if not system_prompt.strip():
        errors.append("Campo 'system_prompt' ausente ou vazio")

    if not user_prompt.strip():
        errors.append("Campo 'user_prompt' ausente ou vazio")

    metadata = root.get("metadata", {}) or {}
    techniques = metadata.get("techniques", [])
    if not isinstance(techniques, list) or len(techniques) < 2:
        errors.append("metadata.techniques deve ser uma lista com pelo menos 2 técnicas")

    combined = f"{system_prompt}\n{user_prompt}\n{description}"
    if "[TODO]" in combined or "TODO" in combined:
        errors.append("Encontrado TODO/[TODO] no prompt")

    if system_prompt.strip() and user_prompt.strip():
        if not _detect_few_shot(system_prompt, user_prompt, root):
            errors.append(
                "Few-shot não detectado. Inclua exemplos de entrada/saída "
                "(ex: seção 'Exemplos' com Input/Output) ou um campo few_shot_examples."
            )

    return (len(errors) == 0), errors


def build_prompt(root: Dict[str, Any]) -> ChatPromptTemplate:
    system_prompt = str(root["system_prompt"])
    user_prompt = str(root["user_prompt"])
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )


def push_prompt_names(prompt: ChatPromptTemplate) -> List[str]:
    """
    Publica em:
      1) bug_to_user_story_v2 (compatível com evaluate.py do boilerplate)
      2) {USERNAME_LANGSMITH_HUB}/bug_to_user_story_v2 (aderente ao enunciado, se existir)
    """
    names: List[str] = [PROMPT_KEY]

    username = (os.getenv("USERNAME_LANGSMITH_HUB") or "").strip()
    if username:
        names.append(f"{username}/{PROMPT_KEY}")

    pushed: List[str] = []
    for name in names:
        hub.push(name, prompt)
        pushed.append(name)

    return pushed


def main() -> int:
    print_section_header("PUSH DE PROMPTS OTIMIZADOS")

    _require_any_env(["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"])

    if not os.path.exists(PROMPTS_FILE):
        print(f"❌ Arquivo não encontrado: {PROMPTS_FILE}")
        return 1

    data = load_yaml(PROMPTS_FILE)

    is_valid, errors = validate_prompt(data)
    if not is_valid:
        print("❌ Prompt inválido. Corrija antes de fazer push:")
        for e in errors:
            print(f" - {e}")
        return 1

    root = _get_prompt_root(data, PROMPT_KEY)
    prompt = build_prompt(root)

    try:
        pushed = push_prompt_names(prompt)
        print("✅ Push realizado com sucesso!")
        for name in pushed:
            print(f"   - Publicado: {name}")
        print("\n👉 Agora execute: python src/evaluate.py")
        print("⚠️  Se o Hub não marcar como público automaticamente, abra o dashboard e altere a visibilidade para Public.")
        return 0
    except Exception as ex:
        print("❌ Falha no push:", ex)
        return 1


if __name__ == "__main__":
    sys.exit(main())