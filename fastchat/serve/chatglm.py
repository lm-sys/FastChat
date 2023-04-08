import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper
)


def prepare_inputs_chatglm(input_ids, output_ids, past_key_values, device, model_config):
    MASK, gMASK = model_config.mask_token_id, model_config.gmask_token_id
    mask_token = gMASK if gMASK in input_ids else MASK
    use_gmask = True if gMASK in input_ids else False
    seq = input_ids + output_ids
    mask_position = seq.index(mask_token)

    if past_key_values == None:
        input_ids = torch.as_tensor([input_ids], device=device)
        attention_mask, position_ids = get_masks_and_position_ids(
            seq=seq,
            mask_position=mask_position,
            context_length=len(seq),
            device=input_ids.device,
            gmask=use_gmask,
            model_config=model_config
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids
        }
    else:
        last_token = output_ids[-1]
        input_ids= torch.as_tensor([[last_token]], device=device)
        bos_token_id = model_config.bos_token_id
        context_length = seq.index(bos_token_id)
        position_ids = torch.tensor([[[mask_position], [len(seq) - context_length]]], 
                                    dtype=torch.long, device=input_ids.device)
        return {
            "input_ids": input_ids,
            "attention_mask": None,
            "past_key_values": past_key_values,
            "position_ids": position_ids
        }


def process_logits_chatglm(last_token_logits):
    # Invalid Score
    if torch.isnan(last_token_logits).any() or torch.isinf(last_token_logits).any():
        last_token_logits.zeros_()
        last_token_logits[..., 20005] = 5e4

    # topk and topp
    warpers = LogitsProcessorList()
    warpers.append(TopKLogitsWarper(50))
    warpers.append(TopPLogitsWarper(0.7))

    last_token_logits = warpers(None, last_token_logits[None, :])
    return last_token_logits


def get_masks_and_position_ids(seq, mask_position, context_length, device, gmask=False, model_config=None):
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., :context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    seq_length = seq.index(model_config.bos_token_id)
    position_ids = torch.arange(context_length, dtype=torch.long, device=device)
    if not gmask:
        position_ids[seq_length:] = mask_position
    block_position_ids = torch.cat((
        torch.zeros(seq_length, dtype=torch.long, device=device),
        torch.arange(context_length - seq_length, dtype=torch.long, device=device) + 1
    ))
    position_ids = torch.stack((position_ids, block_position_ids), dim=0)

    position_ids = position_ids.unsqueeze(0)

    return attention_mask, position_ids

