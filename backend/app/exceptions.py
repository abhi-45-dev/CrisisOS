from typing import Any


class CrisisOSError(Exception):
    status_code = 400
    error_code = "crisisos_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "error": self.error_code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class NotFoundError(CrisisOSError):
    status_code = 404
    error_code = "not_found"


class InvalidDisasterError(CrisisOSError):
    status_code = 400
    error_code = "invalid_disaster_type"


class InvalidEventError(CrisisOSError):
    status_code = 422
    error_code = "invalid_event"


class InvalidScenarioError(CrisisOSError):
    status_code = 422
    error_code = "invalid_scenario"


class SimulationError(CrisisOSError):
    status_code = 400
    error_code = "simulation_error"


class CapacityError(CrisisOSError):
    status_code = 409
    error_code = "insufficient_capacity"


class ResourceUnavailableError(CrisisOSError):
    status_code = 409
    error_code = "resource_unavailable"


class BlockedRouteError(CrisisOSError):
    status_code = 409
    error_code = "blocked_route"


class LLMError(CrisisOSError):
    status_code = 503
    error_code = "llm_unavailable"
