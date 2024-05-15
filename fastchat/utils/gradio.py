"""
Gradio utilities.
"""


"""A javascript function to get url parameters for the gradio web server."""
get_window_url_params_js = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);
    return url_params;
    }
"""


get_window_url_params_with_tos_js = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log("url_params", url_params);

    msg = "Users of this website are required to agree to the following terms:\\n\\nThe service is a research preview. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.\\nPlease do not upload any private information.\\nThe service collects user dialogue data, including both text and images, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license."
    alert(msg);
    return url_params;
    }
"""


def parse_gradio_auth_creds(filename: str):
    """Parse a username:password file for gradio authorization."""
    gradio_auth_creds = []
    with open(filename, "r", encoding="utf8") as file:
        for line in file.readlines():
            gradio_auth_creds += [x.strip() for x in line.split(",") if x.strip()]
    if gradio_auth_creds:
        auth = [tuple(cred.split(":")) for cred in gradio_auth_creds]
    else:
        auth = None
    return auth
