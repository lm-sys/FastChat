# import all the tools
from . import *


# Load the tools from the tool_config_file and convert them to the format required by the model API
def api_tools_loading(tools, api_type):
    if tools is None:
        if api_type in ['openai', 'nvidia_llama31']:
            return None
        elif api_type == 'anthropic_message':
            return []
        elif api_type == 'gemini':
            return None
    else:
        # We use OpenAI's tools format as the default format
        if api_type in ['openai', 'nvidia_llama31']:
            return tools
        elif api_type == 'anthropic_message':
            return_tools = []
            for tool in tools:
                if tool.get('type') == 'function':
                    function_instance = tool.get('function')
                    new_tool = {
                        'name': function_instance.get('name'),
                        'description': function_instance.get('description'),
                        'input_schema': {}
                    }
                    for key, value in function_instance.get('parameters').items():
                        if key in ['type', 'properties', 'required']:
                            new_tool['input_schema'][key] = value
                    return_tools.append(new_tool)
            return return_tools
        elif api_type == 'gemini':
            import google.generativeai as genai  # pip install google-generativeai
            return_tools = []
            for tool in tools:
                if tool.get('type') == 'function':
                    function_instance = tool.get('function')
                    function_name = function_instance.get('name')
                    description=function_instance.get('description')
                    parameters = function_instance.get('parameters')

                    parameters['type'] = genai.protos.Type[parameters['type'].upper()]
                    for prop in parameters['properties'].values():
                        prop['type'] = genai.protos.Type[prop['type'].upper()]
                    new_tool = genai.protos.FunctionDeclaration(
                        name=function_name,
                        description=description,
                        parameters=parameters
                    )
                    return_tools.append(new_tool)
            return return_tools
        else:
            raise ValueError(f"model_type {api_type} not supported")