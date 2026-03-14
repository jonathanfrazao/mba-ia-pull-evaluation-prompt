"""
Testes automatizados para validação de prompts.
"""
import re
import sys
from pathlib import Path

import pytest
import yaml

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure


PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def load_prompts(file_path: Path):
    """Carrega prompts do arquivo YAML."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Arquivo de prompt não encontrado: {file_path}. "
            "Verifique se o caminho está correto e se o arquivo existe."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("O YAML de prompts deve ser um dicionário não vazio.")

    return data


@pytest.fixture(scope="session")
def prompts_dict():
    return load_prompts(PROMPT_FILE)


@pytest.fixture(scope="session")
def all_prompts(prompts_dict):
    """
    Retorna uma lista de (prompt_name, prompt_data) para todos os prompts do YAML.
    """
    items = []
    for name, prompt_data in prompts_dict.items():
        if not isinstance(prompt_data, dict):
            raise ValueError(f"O prompt '{name}' deve ser um objeto/dicionário no YAML.")
        items.append((name, prompt_data))
    return items


def _get_text_fields(prompt_data: dict) -> str:
    """
    Junta system_prompt + user_prompt em um único texto para buscas.
    """
    system_prompt = prompt_data.get("system_prompt") or ""
    user_prompt = prompt_data.get("user_prompt") or ""
    return f"{system_prompt}\n{user_prompt}"


class TestPrompts:
    def test_prompt_has_system_prompt(self, all_prompts):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        for name, prompt_data in all_prompts:
            assert "system_prompt" in prompt_data, f"'{name}' não possui 'system_prompt'."
            system_prompt = prompt_data.get("system_prompt")
            assert isinstance(system_prompt, str), f"'{name}.system_prompt' deve ser string."
            assert system_prompt.strip(), f"'{name}.system_prompt' está vazio."

    def test_prompt_has_role_definition(self, all_prompts):
        """Verifica se o prompt define uma persona (ex: 'Você é um Product Manager')."""
        role_regex = re.compile(r"Você\s+é\s+um[a]?\s+.*Product\s+Manager", re.IGNORECASE)
        for name, prompt_data in all_prompts:
            system_prompt = (prompt_data.get("system_prompt") or "").strip()
            assert system_prompt, f"'{name}.system_prompt' está vazio."
            assert role_regex.search(system_prompt), (
                f"'{name}' não define claramente uma persona no system_prompt. "
                "Esperado algo como 'Você é um Product Manager'."
            )

    def test_prompt_mentions_format(self, all_prompts):
        """
        Verifica se o prompt exige formato Markdown ou User Story padrão.
        (No seu desafio, o mais importante é exigir User Story + critérios.)
        """
        # Aceita as duas formas: mencionar "User Story" OU forçar o padrão "Como um ..." + "Critérios de Aceitação"
        for name, prompt_data in all_prompts:
            text = _get_text_fields(prompt_data)

            mentions_user_story = "User Story" in text or "user story" in text.lower()
            mentions_como_um = "Como um" in text or "Como uma" in text
            mentions_criteria = "Critérios de Aceitação" in text

            assert (mentions_user_story or (mentions_como_um and mentions_criteria)), (
                f"'{name}' não exige claramente um formato de saída. "
                "Esperado mencionar 'User Story' ou instruir 'Como um ...' e 'Critérios de Aceitação'."
            )

    def test_prompt_has_few_shot_examples(self, all_prompts):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        for name, prompt_data in all_prompts:
            system_prompt = (prompt_data.get("system_prompt") or "")

            # Aceita os padrões: Bug/User Story (formato do v2), Bug/Output ou INPUT/OUTPUT
            has_bug_marker = "Bug:" in system_prompt
            has_output_marker = (
                "User Story:" in system_prompt or
                "Output:" in system_prompt or
                "OUTPUT:" in system_prompt
            )

            # Pelo menos 2 exemplos (2 ocorrências de "Bug:")
            bug_count = system_prompt.count("Bug:")
            has_multiple_examples = bug_count >= 2

            assert has_bug_marker and has_output_marker and has_multiple_examples, (
                f"'{name}' parece não conter few-shot de entrada/saída no system_prompt. "
                f"Encontrados {bug_count} exemplo(s) com 'Bug:'. "
                "Esperado pelo menos 2 exemplos com pares Bug:/User Story: ou Bug:/Output:."
            )

    def test_prompt_no_todos(self, all_prompts):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        for name, prompt_data in all_prompts:
            text = _get_text_fields(prompt_data)
            assert "[TODO]" not in text, f"'{name}' contém '[TODO]'."
            # Também previne TODO sem colchetes
            assert "TODO" not in text, f"'{name}' contém 'TODO' não resolvido."

    def test_minimum_techniques(self, all_prompts):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        for name, prompt_data in all_prompts:
            metadata = prompt_data.get("metadata")
            assert isinstance(metadata, dict), f"'{name}.metadata' deve existir e ser um objeto/dict."

            techniques = metadata.get("techniques")
            assert isinstance(techniques, list), f"'{name}.metadata.techniques' deve ser uma lista."
            assert len(techniques) >= 2, (
                f"'{name}' possui menos de 2 técnicas listadas em metadata.techniques."
            )

            # Extra (não pedido, mas útil): valida estrutura usando a função do projeto
            # (garante consistência com push_prompts.py)
            validate_prompt_structure(prompt_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])