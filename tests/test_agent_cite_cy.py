"""Ported from LegalTech apps/api/tests/test_cite_cy.py (b3a0d49): the desktop finds norms as the server does.

Кипрские цитаты: извлечение и свёртка омоглифов.

Почему этот файл понадобился. Извлекатель не знал формата Cap. 113 вовсе и возвращал по
кипрским текстам пустой список. Следствие было не только в мерилах: тот же извлекатель стоит в
боевом страже ``app/draft/verify.py``, то есть кипрские ссылки в черновиках НЕ проверялись —
страж видел ноль цитат и молча пропускал. Выдуманная кипрская норма прошла бы насквозь.

Вторая беда — алфавит. Статьи, вставленные поправками, имеют суффикс греческой буквой (Α, Β, Γ,
Δ … — греческие порядковые), но шесть из 523 в источнике набраны ЛАТИНСКОЙ «A». Α и A
неразличимы на экране и набираются одна вместо другой вслепую.
"""
from __future__ import annotations

from deidkit.agents.cite import extract_citations, fold_citation


def test_cap_first_and_section_first_both_normalize():
    assert extract_citations("See Cap. 113 s. 105.") == ["Cap. 113 s. 105"]
    assert extract_citations("Under Cap 113, section 108") == ["Cap. 113 s. 108"]
    assert extract_citations("section 202 of Cap. 113") == ["Cap. 113 s. 202"]


def test_the_suffix_is_kept_exactly_as_written():
    """Хранится то, что опубликовал источник. Молча переписать написание цитаты — значит
    разойтись с тем, на что ссылается юрист."""
    assert extract_citations("Cap. 113 s. 4Α") == ["Cap. 113 s. 4Α"]     # греческая альфа
    assert extract_citations("Cap. 113 s. 31A") == ["Cap. 113 s. 31A"]   # латинская A


def test_greek_and_latin_lookalikes_fold_to_one_key():
    assert fold_citation("Cap. 113 s. 4Α") == fold_citation("Cap. 113 s. 4A")
    assert fold_citation("Cap. 113 s. 47Β") == fold_citation("Cap. 113 s. 47B")


def test_greek_letters_without_a_lookalike_are_left_alone():
    """Γ, Δ, Θ, Λ латинских двойников не имеют. Сворачивать их не к чему, а попытка «на всякий
    случай» отобразить их на что-то латинское склеила бы разные статьи в один ключ."""
    for ch in "ΓΔΘΛ":
        assert fold_citation(f"Cap. 113 s. 5{ch}") == f"Cap. 113 s. 5{ch}"


def test_other_jurisdictions_are_untouched():
    """Кипрский образец требует «Cap. 113» рядом. Без этого условия «s. 25» из британского
    текста стал бы кипрской цитатой, и чистота юрисдикций посыпалась бы тихо."""
    got = extract_citations("See 8 CFR § 214.2, MCA 1973 s. 25 and 8 U.S.C. § 1182.")
    assert "Cap. 113 s. 25" not in got
    assert set(got) == {"8 CFR § 214.2", "MCA 1973 s. 25", "8 U.S.C. § 1182"}


def test_a_bare_section_number_is_not_a_cyprus_citation():
    assert extract_citations("The register in section 105 must be kept.") == []


def test_fold_is_safe_on_empty_and_none_like_input():
    assert fold_citation("") == ""
    assert fold_citation(None) == ""
