"""
OpenTelemetry / Jaeger Trace Propagation for AsterScope.

Provides cross-service trace_id propagation to fulfill the
"Explainability" claim. Integrates with FastAPI middleware to
automatically create and propagate spans across the retrieval pipeline.
"""

import os
import logging
import uuid
import time
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  OpenTelemetry Integration (conditional import)
# --------------------------------------------------------------------------- #

_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _OTEL_AVAILABLE = True
except ImportError:
    trace = None
    logger.info("OpenTelemetry SDK not installed. Using fallback trace propagation.")


def init_tracing(app=None, service_name: str = "asterscope") -> None:
    """
    Initialize OpenTelemetry tracing with Jaeger exporter.

    Call this during application startup. If the OTel SDK is not installed,
    falls back to the lightweight NovaTracer.

    Args:
        app: FastAPI application instance (for auto-instrumentation).
        service_name: Service name for the tracer resource.
    """
    global _tracer, _OTEL_AVAILABLE

    jaeger_endpoint = os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT")
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not _OTEL_AVAILABLE:
        logger.info("OTel SDK not available. Using NovaTracer fallback.")
        _tracer = NovaTracer(service_name=service_name)
        return

    try:
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
        })
        provider = TracerProvider(resource=resource)

        # Jaeger exporter (if endpoint configured)
        if jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                jaeger_exporter = JaegerExporter(
                    agent_host_name=jaeger_endpoint.split(":")[0],
                    agent_port=int(jaeger_endpoint.split(":")[-1]) if ":" in jaeger_endpoint else 6831,
                )
                provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
                logger.info("OTel: Jaeger exporter configured at %s", jaeger_endpoint)
            except Exception as exc:
                logger.warning("OTel: Jaeger exporter init failed: %s", exc)

        # OTLP exporter (if endpoint configured)
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info("OTel: OTLP exporter configured at %s", otlp_endpoint)
            except Exception as exc:
                logger.warning("OTel: OTLP exporter init failed: %s", exc)

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)

        # Auto-instrument FastAPI
        if app is not None:
            try:
                FastAPIInstrumentor.instrument_app(app)
                logger.info("OTel: FastAPI auto-instrumentation enabled.")
            except Exception as exc:
                logger.warning("OTel: FastAPI instrumentation failed: %s", exc)

        logger.info("OTel tracing initialized for service '%s'.", service_name)

    except Exception as exc:
        logger.error("OTel initialization failed, using fallback: %s", exc)
        _tracer = NovaTracer(service_name=service_name)


# --------------------------------------------------------------------------- #
#  Lightweight Fallback Tracer
# --------------------------------------------------------------------------- #

class NovaTracer:
    """
    Lightweight fallback tracer when OpenTelemetry is not installed.
    Generates trace_id / span_id for log correlation and request tracking.
    """

    def __init__(self, service_name: str = "asterscope"):
        self.service_name = service_name

    @contextmanager
    def start_as_current_span(self, name: str, attributes: Dict[str, Any] = None):
        """Context manager that mimics OTel span interface."""
        span = NovaSpan(name=name, service=self.service_name, attributes=attributes)
        logger.debug("TRACE [%s] span '%s' started", span.trace_id[:8], name)
        try:
            yield span
        finally:
            span.end()
            logger.debug(
                "TRACE [%s] span '%s' ended (%.2fms)",
                span.trace_id[:8],
                name,
                span.duration_ms,
            )


class NovaSpan:
    """Lightweight span object for the fallback tracer."""

    def __init__(self, name: str, service: str, attributes: Dict[str, Any] = None):
        self.name = name
        self.service = service
        self.trace_id = uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.attributes = attributes or {}
        self._start_time = time.monotonic()
        self.duration_ms: float = 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def end(self) -> None:
        self.duration_ms = (time.monotonic() - self._start_time) * 1000


# --------------------------------------------------------------------------- #
#  Tracing Decorator
# --------------------------------------------------------------------------- #

def traced(span_name: str = None):
    """
    Decorator to wrap a function in a trace span.

    Usage:
        @traced("retrieval.dense_search")
        def search_dense(query: str): ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = span_name or f"{func.__module__}.{func.__qualname__}"
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                try:
                    result = func(*args, **kwargs)
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("status", "ok")
                    return result
                except Exception as exc:
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("status", "error")
                        span.set_attribute("error.message", str(exc))
                    raise
        return wrapper
    return decorator


def get_tracer():
    """Get the current tracer (OTel or fallback)."""
    global _tracer
    if _tracer is None:
        _tracer = NovaTracer()
    return _tracer
