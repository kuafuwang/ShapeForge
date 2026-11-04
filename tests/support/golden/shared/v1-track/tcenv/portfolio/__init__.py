"""Shape portfolio contracts (P0): semantic domain, candidate projection, exact dispatch.

The package is deliberately offline and backend-neutral.  It parses and validates the closed
documents a candidate planner emits, projects every variant/fallback into an ordinary
``CandidateArtifact``, and resolves one request to exactly one provider — or refuses with a
typed, non-guessing outcome.  It builds nothing, runs nothing and activates nothing: a
``PortfolioDispatchRecord`` is evidence of a decision, never an active manifest or a performance
claim.
"""

from __future__ import annotations

from .candidate import (
    DOMAIN_FALLBACK_ROLE,
    PORTFOLIO_ENTRY_PATH,
    PORTFOLIO_SCHEMA_ID,
    TIMING_BASELINE_ROLE,
    CandidateProjection,
    DomainFallback,
    EvaluationBinding,
    PortfolioCandidate,
    PortfolioVariant,
    RegistrationReport,
    parse_portfolio,
    project_provider,
)
from .dispatch import (
    DISPATCH_RECORD_SCHEMA_ID,
    DISPATCH_RULES_VERSION,
    ExactDispatchResolver,
    GeometryPreflight,
    PortfolioDispatchRecord,
    preflight_geometry,
)
from .domain import (
    SEMANTIC_DOMAIN_PLAN_SCHEMA_ID,
    DomainAuthority,
    Guard,
    SemanticDomainInstance,
    SemanticDomainPlan,
    canonical_ast,
    canonical_guard,
    guard_digest,
    normalize_semantic_domain_plan,
    parse_guard,
    parse_semantic_domain_plan,
)

__all__ = [
    "DOMAIN_FALLBACK_ROLE",
    "DISPATCH_RECORD_SCHEMA_ID",
    "DISPATCH_RULES_VERSION",
    "CandidateProjection",
    "DomainAuthority",
    "DomainFallback",
    "EvaluationBinding",
    "ExactDispatchResolver",
    "GeometryPreflight",
    "Guard",
    "PORTFOLIO_ENTRY_PATH",
    "PORTFOLIO_SCHEMA_ID",
    "PortfolioCandidate",
    "PortfolioDispatchRecord",
    "PortfolioVariant",
    "RegistrationReport",
    "SEMANTIC_DOMAIN_PLAN_SCHEMA_ID",
    "SemanticDomainInstance",
    "SemanticDomainPlan",
    "TIMING_BASELINE_ROLE",
    "canonical_ast",
    "canonical_guard",
    "guard_digest",
    "normalize_semantic_domain_plan",
    "parse_guard",
    "parse_portfolio",
    "parse_semantic_domain_plan",
    "preflight_geometry",
    "project_provider",
]
