from typing import Literal
from e2b import Sandbox
import time

from .constants import E2B_API_KEY, SANDBOX_TEMPLATE_ID, SANDBOX_NGINX_PORT, SANDBOX_RETRY_COUNT

def create_sandbox(template: str=SANDBOX_TEMPLATE_ID) -> Sandbox:
    for attempt in range(1, SANDBOX_RETRY_COUNT + 1):
        try:
            return Sandbox(
                api_key=E2B_API_KEY,
                template=template
            )
        except Exception as e:
            if attempt < SANDBOX_RETRY_COUNT:
                time.sleep(1 * attempt)
            else:
                raise e
    raise RuntimeError("Failed to create sandbox after maximum attempts")

def get_sandbox_app_url(
        sandbox: Sandbox,
        app_type: Literal["react", "vue"]
    ) -> str:
    return f"https://{sandbox.get_host(port=SANDBOX_NGINX_PORT)}/container/?app={app_type}"
