"""Lazy, backend-injected Track services.

This module is deliberately limited to construction mechanics.  It does not define a
new candidate role, delivery, execution kind, media type, or registry schema.  A backend
factory receives the already-selected :class:`TrackSpec` and returns a
:class:`LazyTrackAdapter`; services are imported/constructed only when a caller resolves
them.  Unknown service identities fail closed instead of falling back to an MSL adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..contract_errors import UnsupportedContractCapability
from .registry import TrackSpec


@dataclass
class LazyTrackAdapter:
    """Resolve backend-owned services for one frozen TrackSpec.

    Factories are closures supplied by the backend.  No default registry is read here,
    which is important for worker reconstruction: the caller must pass the same frozen
    TrackSpec it audited, not a possibly drifted default.
    """

    spec: TrackSpec
    _factories: Mapping[str, Callable[[], Any]] = field(default_factory=dict)
    _resolved: dict[str, Any] = field(default_factory=dict, init=False)

    def resolve(self, service: str) -> Any:
        """Construct and cache one named service, or refuse an unknown binding."""
        if service in self._resolved:
            return self._resolved[service]
        factory = self._factories.get(service)
        if factory is None or not callable(factory):
            raise UnsupportedContractCapability(
                f"track {self.spec.track_id}/{self.spec.track_version} has no registered "
                f"{service!r} adapter binding"
            )
        value = factory()
        self._resolved[service] = value
        return value


__all__ = ["LazyTrackAdapter"]
