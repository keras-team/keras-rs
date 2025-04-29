import re
import textwrap
from typing import Any


def format_docstring(template: str, width: int = 80, **kwargs: Any) -> str:
    """Formats and wraps a docstring using dedent and fill."""
    base_indent_str = " " * 4

    # Initial format
    formatted = template.format(**kwargs)

    # Dedent the whole block
    dedented_all = textwrap.dedent(formatted).strip()

    # Split into logical paragraphs/blocks.
    blocks = re.split(r"(\n\s*\n)", dedented_all)

    processed_output = []

    for block in blocks:
        stripped_block = block.strip()
        if not stripped_block:
            processed_output.append(block)
            continue

        if "```" in stripped_block:
            formula_dedented = textwrap.dedent(stripped_block)
            processed_output.append(
                textwrap.indent(formula_dedented, base_indent_str)
            )
        elif "where:" in stripped_block:
            # Expect this to be already indented.
            splitted_block = stripped_block.split("\n")
            processed_output.append(
                textwrap.indent(
                    splitted_block[0] + "\n\n" + "\n".join(splitted_block[1:]),
                    base_indent_str,
                )
            )
        else:
            processed_output.append(
                textwrap.fill(
                    stripped_block,
                    width=width - len(base_indent_str),
                    initial_indent=base_indent_str,
                    subsequent_indent=base_indent_str,
                )
            )

    final_string = "".join(processed_output).strip()
    final_string = base_indent_str + final_string
    return final_string
