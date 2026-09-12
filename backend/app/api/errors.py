from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.exceptions import CrisisOSError
from app.logging_config import logger


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(CrisisOSError)
    async def crisisos_handler(_request: Request, exc: CrisisOSError) -> JSONResponse:
        logger.error("API error %s: %s", exc.error_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.error("Malformed request: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "malformed_request",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(ValidationError)
    async def pydantic_handler(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "malformed_request",
                "message": "Data validation failed",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def generic_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "An unexpected error occurred"},
        )
