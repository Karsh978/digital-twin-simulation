"""Async event bus for cross-slice shock propagation (meta-architecture).

Design rules from the architecture doc (§1.1):
- Slices never share state or agents. They emit and consume *shock events*
  (rate changes, demand shifts, regulatory actions, major failures).
- The bus is the loosest-coupled part of the system: pub/sub, async delivery,
  no slice may reach into another slice.
- Cross-slice output is a LOWER-CONFIDENCE layer downstream of validated
  single-slice output. Every event therefore carries a `confidence` label and
  the bus tags anything it propagates as cross-slice (lower confidence).
"""

from __future__ import annotations

import heapq
import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class Confidence(str, Enum):
    """Confidence label attached to simulation output (§1.1)."""

    SLICE_VALIDATED = "slice_validated"      # single-slice, validated vs. dense historical data
    CROSS_SLICE = "cross_slice"              # propagated between slices — thinner validation
    SPECULATIVE = "speculative"              # no historical analog; treat as scenario exploration


class ShockType(str, Enum):
    RATE_CHANGE = "rate_change"
    DEMAND_SHIFT = "demand_shift"
    REGULATORY_ACTION = "regulatory_action"
    MAJOR_FAILURE = "major_failure"          # e.g. a down-round wave, a large collapse
    LIQUIDITY_SHOCK = "liquidity_shock"
    CUSTOM = "custom"


@dataclass(order=True)
class ShockEvent:
    """A cross-slice shock. Delivered to subscribers after `delay_weeks`.

    `magnitude` is a signed intensity in roughly [-1, 1]; sign convention is
    defined per shock type (e.g. for RATE_CHANGE, +0.5 = rates up sharply).
    """

    deliver_at_week: int
    shock_type: ShockType = field(compare=False)
    origin_slice: str = field(compare=False)
    magnitude: float = field(compare=False)
    description: str = field(compare=False, default="")
    confidence: Confidence = field(compare=False, default=Confidence.CROSS_SLICE)
    payload: dict = field(compare=False, default_factory=dict)
    _seq: int = field(compare=True, default=0, repr=False)


Subscriber = Callable[[ShockEvent], None]


class EventBus:
    """Priority-queue event bus with delayed (async) delivery.

    Slices subscribe with a callback; `publish` schedules an event for
    delivery at the slice's current week + delay. The environment server
    calls `dispatch_due(current_week)` each tick.
    """

    def __init__(self) -> None:
        self._queue: List[ShockEvent] = []
        self._subscribers: Dict[str, List[Subscriber]] = {}
        self._counter = itertools.count()
        self.delivered: List[ShockEvent] = []  # full audit log

    def subscribe(self, slice_name: str, callback: Subscriber) -> None:
        self._subscribers.setdefault(slice_name, []).append(callback)

    def publish(
        self,
        shock_type: ShockType,
        origin_slice: str,
        magnitude: float,
        current_week: int,
        delay_weeks: int = 0,
        description: str = "",
        confidence: Confidence = Confidence.CROSS_SLICE,
        payload: Optional[dict] = None,
    ) -> ShockEvent:
        if confidence == Confidence.SLICE_VALIDATED:
            # Anything crossing the bus is, by definition, cross-slice output.
            confidence = Confidence.CROSS_SLICE
        ev = ShockEvent(
            deliver_at_week=current_week + max(0, delay_weeks),
            shock_type=shock_type,
            origin_slice=origin_slice,
            magnitude=magnitude,
            description=description,
            confidence=confidence,
            payload=payload or {},
            _seq=next(self._counter),
        )
        heapq.heappush(self._queue, ev)
        log.debug("queued %s from %s at week %d", shock_type, origin_slice, ev.deliver_at_week)
        return ev

    def dispatch_due(self, current_week: int) -> List[ShockEvent]:
        """Deliver all events due at or before `current_week`. Returns them."""
        due: List[ShockEvent] = []
        while self._queue and self._queue[0].deliver_at_week <= current_week:
            ev = heapq.heappop(self._queue)
            self.delivered.append(ev)
            due.append(ev)
            for slice_name, subs in self._subscribers.items():
                if slice_name == ev.origin_slice:
                    continue  # a slice does not receive its own shocks back
                for cb in subs:
                    cb(ev)
        return due

    def pending(self) -> int:
        return len(self._queue)
