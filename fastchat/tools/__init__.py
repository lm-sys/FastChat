# import all the tools
import json
from . import *


# Load the tools from the tool_config_file and convert them to the format required by the model API
def general_tools_loading(tool_config_file, model_api_dict):
    tools = json.load(open(tool_config_file))

    if model_api_dict['api_type'] == 'openai':
        return tools
    elif model_api_dict['api_type'] == 'anthropic_message':
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
    else:
        raise ValueError(f"model_type {model_api_dict['model_type']} not supported")