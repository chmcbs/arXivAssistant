from core.arxiv_text import format_arxiv_display_text


def test_format_arxiv_display_text_unwraps_latex_alpha():
    assert format_arxiv_display_text(r"$\alpha$-Net") == "α-Net"


def test_format_arxiv_display_text_unwraps_unicode_alpha_in_math_delimiters():
    assert format_arxiv_display_text("$α$") == "α"


def test_format_arxiv_display_text_handles_embedded_math():
    assert (
        format_arxiv_display_text(r"Scaling laws for $\beta$-divergence models")
        == "Scaling laws for β-divergence models"
    )


def test_format_arxiv_display_text_strips_text_commands():
    assert format_arxiv_display_text(r"\textbf{Attention} is all you need") == (
        "Attention is all you need"
    )


def test_format_arxiv_display_text_leaves_plain_titles_unchanged():
    title = "Efficient LLM inference with speculative decoding"
    assert format_arxiv_display_text(title) == title
