import torch


def tokensByDevice(device, token, isBase, debug=False, showTokens=False):
    if debug and showTokens:
        print(
            f'{"Tokenising prompt" if isBase else "Inferencing"} using ({device})... Tokens:', token)
    elif debug:
        print(
            f'{"Tokenising prompt" if isBase else "Inferencing"} using ({device})...')
    if device == 'cuda':
        if isBase:
            return torch.as_tensor([token]).cuda()
        else:
            return torch.as_tensor([[token]], device="cuda")
    elif device == 'cpu-gptq':
        if isBase:
            return token
        else:
            return torch.as_tensor([[token]])
    elif device == 'cpu':
        if isBase:
            return torch.as_tensor([token])
        else:
            return torch.as_tensor([[token]])
