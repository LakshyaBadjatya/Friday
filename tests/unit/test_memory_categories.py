"""Unit tests for ``friday.memory.categories``.

Pin the category/tier taxonomy, the :class:`CategorizedFact` defaults, and the
soft-delete contract — in particular that ``DECISION`` records are never
deletable.
"""

from __future__ import annotations

import pytest

from friday.memory.categories import (
    CategorizedFact,
    MemoryCategory,
    MemoryTier,
    UndeletableDecisionError,
    is_deletable,
    soft_delete,
)


def test_memory_category_members() -> None:
    assert {c.name for c in MemoryCategory} == {
        "FACT",
        "PREFERENCE",
        "DECISION",
        "TASK",
        "PERSON",
        "PROJECT",
        "EVENT",
    }


def test_memory_tier_members() -> None:
    assert {t.name for t in MemoryTier} == {"HOT", "WARM", "COLD"}


def test_categories_are_str_enum() -> None:
    assert MemoryCategory.FACT.value == "fact"
    assert MemoryTier.COLD.value == "cold"
    # StrEnum members are usable as plain strings.
    assert isinstance(MemoryCategory.FACT, str)
    assert f"{MemoryTier.HOT}" == "hot"


def test_categorized_fact_defaults() -> None:
    fact = CategorizedFact(text="the sky is blue", category=MemoryCategory.FACT)
    assert fact.tier is MemoryTier.HOT
    assert fact.namespace == "default"
    assert fact.deleted is False
    assert fact.links == []


def test_categorized_fact_links_are_independent_per_instance() -> None:
    a = CategorizedFact(text="a", category=MemoryCategory.FACT)
    b = CategorizedFact(text="b", category=MemoryCategory.FACT)
    a.links.append("x")
    assert b.links == []


def test_categorized_fact_accepts_explicit_fields() -> None:
    fact = CategorizedFact(
        text="prefers dark mode",
        category=MemoryCategory.PREFERENCE,
        tier=MemoryTier.WARM,
        namespace="user-42",
        links=["fact-1"],
    )
    assert fact.category is MemoryCategory.PREFERENCE
    assert fact.tier is MemoryTier.WARM
    assert fact.namespace == "user-42"
    assert fact.links == ["fact-1"]


def test_categorized_fact_validates_category() -> None:
    with pytest.raises(ValueError):
        CategorizedFact(text="x", category="not-a-category")


def test_soft_delete_sets_deleted_flag() -> None:
    fact = CategorizedFact(text="todo: buy milk", category=MemoryCategory.TASK)
    returned = soft_delete(fact)
    assert fact.deleted is True
    assert returned is fact


def test_soft_delete_is_idempotent() -> None:
    fact = CategorizedFact(text="met Alice", category=MemoryCategory.PERSON)
    soft_delete(fact)
    soft_delete(fact)
    assert fact.deleted is True


@pytest.mark.parametrize(
    "category",
    [
        MemoryCategory.FACT,
        MemoryCategory.PREFERENCE,
        MemoryCategory.TASK,
        MemoryCategory.PERSON,
        MemoryCategory.PROJECT,
        MemoryCategory.EVENT,
    ],
)
def test_soft_delete_allows_non_decision_categories(
    category: MemoryCategory,
) -> None:
    fact = CategorizedFact(text="x", category=category)
    soft_delete(fact)
    assert fact.deleted is True


def test_soft_delete_protects_decision() -> None:
    decision = CategorizedFact(
        text="chose Postgres over MySQL", category=MemoryCategory.DECISION
    )
    with pytest.raises(UndeletableDecisionError):
        soft_delete(decision)
    # Record must remain undeleted after the failed attempt.
    assert decision.deleted is False


def test_undeletable_decision_error_is_friday_error() -> None:
    from friday.errors import FridayError

    assert issubclass(UndeletableDecisionError, FridayError)


def test_is_deletable_reports_decision_as_protected() -> None:
    decision = CategorizedFact(text="x", category=MemoryCategory.DECISION)
    fact = CategorizedFact(text="y", category=MemoryCategory.FACT)
    assert is_deletable(decision) is False
    assert is_deletable(fact) is True
