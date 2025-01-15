'''
Constants for sandbox.
'''

import os

E2B_API_KEY = os.environ.get("E2B_API_KEY")
'''
API key for the e2b API.
'''

SANDBOX_TEMPLATE_ID: str = "bxq9sha9l55ytsyfturr"
'''
Template ID for the sandbox.
'''

SANDBOX_NGINX_PORT: int = 8000
'''
Nginx port for the sandbox.
'''

SANDBOX_RETRY_COUNT: int = 3
'''
Number of times to retry the sandbox creation.
'''