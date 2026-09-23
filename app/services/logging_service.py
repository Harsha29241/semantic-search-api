import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | "
    "%(name)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)

logger = logging.getLogger("semantic-search-api")


async def log_requests(request: Request, call_next):
    """
    Log every API request with method, path,
    response status and execution time.
    """

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        execution_time = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "%s %s | status=%s | time=%.2fms",
            request.method,
            request.url.path,
            response.status_code,
            execution_time
        )

        return response

    except Exception:
        execution_time = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "%s %s | status=500 | time=%.2fms",
            request.method,
            request.url.path,
            execution_time
        )

        raise


async def handle_unexpected_error(
    request: Request,
    exc: Exception
):
    """
    Centralized handler for unexpected application errors.
    """

    logger.exception(
        "Unhandled exception | %s %s",
        request.method,
        request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred."
        }
    )


def configure_logging(app: FastAPI):
    """
    Configure request logging and centralized
    unexpected-error handling.
    """

    app.middleware("http")(log_requests)

    app.add_exception_handler(
        Exception,
        handle_unexpected_error
    )

    logger.info(
        "Logging and centralized error handling configured"
    )