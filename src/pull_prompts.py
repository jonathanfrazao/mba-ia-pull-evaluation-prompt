from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from langchain import hub


PROMPT_ID = "leonanluppi/bug_to_user_story_v1"
OUT_FILE = Path("prompts/raw_prompts.yml")


def _require_env(keys: List[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Variáveis de ambiente ausentes: "
            + ", ".join(missing)
            + "\nDica: copie .env.example para .env e preencha as chaves."
        )


def _serialize_prompt(prompt_obj: Any) -> Dict[str, Any]:
    """
    Tenta transformar o objeto retornado pelo hub.pull em um YAML legível.
    """
    data: Dict[str, Any] = {
        "source": "langsmith_prompt_hub",
        "prompt_id": PROMPT_ID,
        "object_type": prompt_obj.__class__.__name__,
    }

    # input variables
    input_vars = getattr(prompt_obj, "input_variables", None)
    if isinstance(input_vars, (list, tuple)):
        data["input_variables"] = list(input_vars)

    # mensagens
    msgs = getattr(prompt_obj, "messages", None)
    if isinstance(msgs, list):
        serialized_msgs = []
        for m in msgs:
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
            template = getattr(m, "prompt", None)
            if template is not None:
                # m.prompt.template existe
                template_text = getattr(template, "template", None) or str(template)
            else:
                template_text = getattr(m, "template", None) or str(m)

            serialized_msgs.append(
                {"role": str(role), "template": str(template_text)}
            )
        data["messages"] = serialized_msgs

    # fallback: string
    if "messages" not in data:
        data["raw"] = str(prompt_obj)

    return data


def main() -> None:
    load_dotenv()

    _require_env(["LANGCHAIN_API_KEY"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    prompt = hub.pull(PROMPT_ID)
    payload = _serialize_prompt(prompt)

    OUT_FILE.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"OK. Prompt puxado: {PROMPT_ID}")
    print(f"Salvo em: {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()