"""
Plain-text formatting for arXiv titles and abstracts that contain LaTeX markup
"""

from __future__ import annotations

import re

_LATEX_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

_TEXT_COMMANDS = frozenset(
    {
        "text",
        "mathrm",
        "mathbf",
        "mathit",
        "textrm",
        "textbf",
        "emph",
        "operatorname",
    }
)

_BRACE_GROUP_RE = re.compile(r"\{([^{}]*)\}")
_INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
_TEXT_COMMAND_RE = re.compile(
    r"\\(" + "|".join(sorted(_TEXT_COMMANDS, key=len, reverse=True)) + r")\{([^{}]*)\}"
)
_LATEX_COMMAND_RE = re.compile(r"\\([a-zA-Z]+)")
_GREEK_COMMAND_RE = re.compile(
    r"\\(" + "|".join(sorted(_LATEX_GREEK, key=len, reverse=True)) + r")\b"
)
_WHITESPACE_RE = re.compile(r"\s+")


def _replace_latex_command(match: re.Match[str]) -> str:
    command = match.group(1)
    if command in _TEXT_COMMANDS:
        return ""
    return _LATEX_GREEK.get(command, "")


def _simplify_math_content(content: str) -> str:
    simplified = content.strip()
    for _ in range(8):
        updated = _BRACE_GROUP_RE.sub(r"\1", simplified)
        if updated == simplified:
            break
        simplified = updated
    simplified = _LATEX_COMMAND_RE.sub(_replace_latex_command, simplified)
    simplified = simplified.replace("^", "").replace("_", "")
    simplified = simplified.replace("{", "").replace("}", "")
    return simplified.strip()


def format_arxiv_display_text(text: str) -> str:
    if not text:
        return text

    formatted = text
    for _ in range(8):
        updated = _TEXT_COMMAND_RE.sub(r"\2", formatted)
        if updated == formatted:
            break
        formatted = updated

    for _ in range(8):
        updated = _INLINE_MATH_RE.sub(
            lambda match: _simplify_math_content(match.group(1)),
            formatted,
        )
        if updated == formatted:
            break
        formatted = updated

    formatted = _GREEK_COMMAND_RE.sub(
        lambda match: _LATEX_GREEK[match.group(1)],
        formatted,
    )

    for _ in range(8):
        updated = _BRACE_GROUP_RE.sub(r"\1", formatted)
        if updated == formatted:
            break
        formatted = updated

    formatted = _LATEX_COMMAND_RE.sub(_replace_latex_command, formatted)
    formatted = formatted.replace("{", "").replace("}", "")
    formatted = _WHITESPACE_RE.sub(" ", formatted).strip()
    return formatted
