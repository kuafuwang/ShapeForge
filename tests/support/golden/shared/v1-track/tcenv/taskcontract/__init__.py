"""The v2 task-contract plane: standard entry, reference oracle, display and baselines.

This package is deliberately *not* a second semantic authority.  ``tasks/<op>/opspec.json``
(v2) owns the semantics, ``reference.py`` owns the independent oracle, and the optional
``reference.json``/``baselines.*``/``display.json``/``known_good.*`` files own only the
representation, sampling, performance and self-check layers.  ``display.json`` can never
change cases, tolerances, ABI or launch, and ``baselines`` can never serve as the
correctness oracle.
"""

from __future__ import annotations

from .baselines import (
    BASELINES_FILE,
    BASELINES_RUNNER_FILE,
    BaselinePackage,
    BaselineRegistration,
    BaselineRegistry,
    load_baselines,
)
from .display import (
    G0_SEM_SPEC_TEXT_ALLOWLIST,
    SEM_INT_SEED_DIGITS_MAX,
    DisplayOverrides,
    load_display,
    spec_text_override_allowed,
)
from .errors import (
    AssemblyError,
    ReferenceBoundaryError,
    SemanticUnresolved,
    TaskContractError,
    UnsupportedContractCapability,
)
from .locator import (
    StagedTaskPack,
    TaskDiscoveryError,
    TaskLocator,
    discover_standard_tasks,
)
from .pack import PackClosureError, TaskPackClosure, code_import_closure, task_pack_closure, task_pack_files
from .reference import ReferenceModule, ReferenceRegistry, load_reference
from .runner import (
    CandidateBuildRequest,
    CandidateBuilder,
    CandidateEvaluationRequest,
    CandidateEvaluator,
    GeometryResolver,
    TaskEvaluationResult,
    TrackSelection,
    default_geometry_resolver,
    evaluate_task,
    select_track,
)
from .sampling import SamplingPlan, SamplingProfile, generate_inputs, sampling_plan_from_document
from .loader import (
    LegacyStandardTask,
    StandardEntry,
    TaskContract,
    inspect_standard_entry,
    load_standard_task,
    load_task_contract,
)

__all__ = [
    "AssemblyError",
    "BASELINES_FILE",
    "BASELINES_RUNNER_FILE",
    "BaselinePackage",
    "BaselineRegistration",
    "BaselineRegistry",
    "CandidateBuildRequest",
    "CandidateBuilder",
    "CandidateEvaluationRequest",
    "CandidateEvaluator",
    "DisplayOverrides",
    "G0_SEM_SPEC_TEXT_ALLOWLIST",
    "GeometryResolver",
    "LegacyStandardTask",
    "PackClosureError",
    "ReferenceBoundaryError",
    "ReferenceModule",
    "ReferenceRegistry",
    "SamplingPlan",
    "SamplingProfile",
    "SemanticUnresolved",
    "SEM_INT_SEED_DIGITS_MAX",
    "StagedTaskPack",
    "StandardEntry",
    "TaskContract",
    "TaskDiscoveryError",
    "TaskLocator",
    "TaskPackClosure",
    "TaskContractError",
    "UnsupportedContractCapability",
    "code_import_closure",
    "default_geometry_resolver",
    "discover_standard_tasks",
    "evaluate_task",
    "generate_inputs",
    "inspect_standard_entry",
    "load_baselines",
    "load_display",
    "load_reference",
    "load_standard_task",
    "load_task_contract",
    "sampling_plan_from_document",
    "select_track",
    "spec_text_override_allowed",
    "task_pack_closure",
    "task_pack_files",
]
