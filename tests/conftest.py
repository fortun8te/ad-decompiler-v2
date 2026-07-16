import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _isolate_vlm_endpoint_state():
    """Reset vlm_client's process-global endpoint state between tests.

    The result cache and the circuit breaker are module-level singletons keyed by
    (base_url, model), which every test shares. Without this, a test that simulates a dead
    endpoint (3+ consecutive timeouts) leaves the breaker OPEN, and unrelated tests that
    run later in the same process get their VLM calls refused — order-dependent failures
    that have nothing to do with the code under test.
    """
    try:
        from src import vlm_client
    except Exception:            # vlm_client not importable in this env: nothing to isolate
        yield
        return
    vlm_client.reset_breaker()
    vlm_client.reset_cache()
    yield
    vlm_client.reset_breaker()
    vlm_client.reset_cache()
