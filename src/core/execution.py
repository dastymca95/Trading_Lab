from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionDecision:
    allowed: bool
    reason:  str

    # ── Constructores semánticos ───────────────────────────────────────────
    @classmethod
    def permit(cls) -> ExecutionDecision:
        """Decisión de ejecución permitida. reason='ok' para compatibilidad
        con el flujo actual que espera el string 'ok'."""
        return cls(allowed=True, reason="ok")

    @classmethod
    def deny(cls, reason: str) -> ExecutionDecision:
        """Decisión de ejecución denegada con motivo explícito."""
        return cls(allowed=False, reason=reason)

    # ── Compatibilidad temporal con código que espera tuple[bool, str] ────
    def as_tuple(self) -> tuple[bool, str]:
        """Devuelve (allowed, reason) para consumidores que aún hacen
        unpacking: `can_eval, eval_reason = decision.as_tuple()`"""
        return (self.allowed, self.reason)
