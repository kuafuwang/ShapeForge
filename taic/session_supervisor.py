"""Session-level policy: attempt labels, provider retry state, and outcome classification.

This module deliberately knows nothing about evaluation, prompts, budgets or the orchestrator's
candidate list. It turns one attempt's ``AgentResult``-shaped object into an outcome, and runs the
bounded provider-retry loop with callbacks for state publication and scheduling. Keeping it pure
makes the retry crash windows testable without GPU, hmz or a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

NO_KERNEL_OUTCOMES = frozenset({"agent_failed", "provider_failed", "agent_no_submission"})
PROVIDER_RETRY_BASE_DELAY_S = 5.0


@dataclass(frozen=True)
class ProviderAttempt:
    """One attempt of one candidate: a stable candidate label plus a distinct session tag."""

    candidate_label: str
    attempt: int

    @property
    def session_tag(self) -> str:
        return self.candidate_label if self.attempt == 0 else f"{self.candidate_label}.retry{self.attempt}"


def session_origin(*, provider: str, model: str, committee: str, session_label: str, candidate_label: str, attempt: int) -> dict:
    """Provenance for a candidate written by one attempt, kept JSON-serializable."""
    return {
        "origin": "session", "provider": provider, "model": model, "committee": committee,
        "session_label": session_label, "candidate_label": candidate_label, "attempt": attempt,
    }


def classify_result(result, seed_source: str | None) -> str:
    """Classify one session without deciding what to do with it.

    ``provider_failed`` is only a structured provider turn failure with no submission.
    A timeout/process failure is ``agent_failed``; a normal answer without a submission is
    ``agent_no_submission``. An identical seed is ``unchanged`` and a new source is evaluated.
    """
    if result.status == "timeout":
        return "agent_failed"
    if result.source is None:
        failure_class = str((getattr(result, "diagnostics", {}) or {}).get("failure_class") or "")
        if failure_class == "candidate_delivery":
            # The session finished normally and wrote a submission whose bytes cannot become a
            # candidate.  It is a typed delivery rejection, not an agent failure, and it must not be
            # charged against the agent/provider failure budget.
            return "delivery_rejected"
        if failure_class == "turn_failed":
            return "provider_failed"
        if failure_class == "no_submission_after_result":
            return "agent_no_submission"
        return "agent_failed"
    if result.status == "failed" and result.source == seed_source:
        return "agent_failed"
    if seed_source is not None and result.source == seed_source:
        return "unchanged"
    return "evaluated"


class ProviderRetryController:
    """Bounded provider retries and their durable attempt markers.

    ``attempts`` maps a candidate label to the next attempt to run. A value of
    ``retries + 1`` is a terminal marker: resume returns ``provider_failed`` without
    opening another session. The round caller clears the marker after consuming it.
    """

    def __init__(self, retries: int, attempts: dict[str, int] | None = None) -> None:
        self.retries = retries
        self.attempts: dict[str, int] = dict(attempts or {})

    @staticmethod
    def delay(attempt: int) -> float:
        return min(30.0, PROVIDER_RETRY_BASE_DELAY_S * (2 ** max(0, attempt - 1)))

    def clear(self, label: str) -> None:
        self.attempts.pop(label, None)

    async def run(
        self,
        candidate_label: str,
        *,
        attempt_fn: Callable[[str, int], Awaitable[str]],
        publish: Callable[[str | None], None],
        on_scheduled: Callable[[str, str, int, float], None],
        sleep: Callable[[float], Awaitable[None]],
    ) -> str:
        """Run one attempt and retry only ``provider_failed`` outcomes.

        @param candidate_label stable candidate label.
        @param attempt_fn callback ``(session_tag, attempt) -> outcome``.
        @param publish callback that persists state with phase ``None``/``"llm"``.
        @param on_scheduled callback ``(failed_session_tag, candidate_label, next_attempt, delay)``.
        @param sleep injected backoff sleep.
        @return the final attempt outcome, including ``provider_failed`` for a terminal marker.
        """
        attempt = int(self.attempts.get(candidate_label, 0) or 0)
        if attempt > self.retries:
            # A terminal marker from a crashed final attempt. Do not open another paid session;
            # the round caller consumes this provider_failed outcome and increments its counter.
            return "provider_failed"
        while True:
            session_tag = ProviderAttempt(candidate_label, attempt).session_tag
            outcome = await attempt_fn(session_tag, attempt)
            if outcome != "provider_failed":
                self.clear(candidate_label)
                publish(None)
                return outcome
            next_attempt = attempt + 1
            self.attempts[candidate_label] = next_attempt
            publish("llm")
            if next_attempt > self.retries:
                # Keep the terminal marker on disk until the round consumes this outcome. If the
                # process dies here, resume returns provider_failed without replaying the final
                # attempt or resetting the retry budget.
                return "provider_failed"
            delay = self.delay(next_attempt)
            on_scheduled(session_tag, candidate_label, next_attempt, delay)
            attempt = next_attempt
            await sleep(delay)
