import os, sys

sys.path.append(os.getcwd())

from fastchat.model.model_adapter import get_model_adapter, model_adapters, MODEL_IDS

if __name__ == "__main__":
    print(MODEL_IDS)
    adapter = get_model_adapter("lmsys/vicuna-7b-v1.5", "vicuna")
    print(f"adapter type: {type(adapter)}, and adapter_id: {adapter.adapter_model_id}")
    adapter = get_model_adapter("lmsys/vicuna-7b-v1.5")
    print(
        f"adapter type: {type(adapter)}, and adapter_id: {adapter.adapter_model_id}\n\n"
    )

    for adapter in model_adapters:
        print(
            f"adapter type: {type(adapter)}, and adapter id: {adapter.adapter_model_id}"
        )
