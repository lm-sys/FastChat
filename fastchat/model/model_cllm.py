import torch
import gc

import os
import random
from typing import Dict, Optional, Sequence, List, Tuple
from transformers.cache_utils import Cache, DynamicCache
from transformers import LlamaModel,LlamaForCausalLM

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

DynamicCache.delete_false_key_value = delete_false_key_value

@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    tokenizer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001,  dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0

        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0

        prev_len = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1)

            next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)

            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            
            jacobian_trajectory.append(next_point)

            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()

                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()

                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length + fast_forward_cnt]:
                    eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                    eos_position = eos_positions[0]
                    generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                else:
                    generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length + fast_forward_cnt], skip_special_tokens=True)

                print(generated_str[prev_len:], flush=True, end="")
                prev_len = len(generated_str)
                break 

            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated


            if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length]:
                eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                eos_position = eos_positions[0]

                generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
            else:

                generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length], skip_special_tokens=True)

            print(generated_str[prev_len:], flush=True, end="")
            prev_len = len(generated_str)
            

            iter_counter += 1

        return accurate_n_gram, first_correct_token, iter_counter
        

LlamaForCausalLM.jacobi_forward = jacobi_forward

def get_jacobian_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens
):

    bsz = input_ids.shape[0] 
    prompt_len = [torch.sum(t) for t in attention_mask]
    max_prompt_len = max(prompt_len)
    total_len = max_prompt_len + max_new_tokens

    # initialize the first point of jacobian trajectory
    tokens = torch.full((bsz, total_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    for i in range(bsz):
        tokens[i, :] = torch.tensor(random.choices(input_ids[i][attention_mask[i]==1], k=total_len), dtype=torch.long, device="cuda")
        tokens[i, : prompt_len[i]] = input_ids[i][: prompt_len[i]].clone().detach()
    itr = 0
    next_generation = tokens
    generate_attention_mask = torch.full_like(next_generation, 1).to(tokens.device)
    while True:
        
        current_generation = next_generation
        with torch.no_grad():
            logits = model(current_generation, generate_attention_mask).logits
        next_generation = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)

        # hold prompt unchanged and update generated tokens
        for i in range(bsz):
            next_generation[i, :] = torch.cat((tokens[i, :prompt_len[i]], next_generation[i, prompt_len[i]-1:total_len-1]), dim=0)
        if torch.all(torch.eq(next_generation, current_generation)).item():
            return next_generation, itr # right generation is saved twice so we delete the last element of trajectory list
        itr+=1

def generate_stream_cllm(model, tokenizer, params, device, context_len, stream_interval = 2, judge_sent_end = False):
    prompt = params["prompt"]
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_new_tokens = int(params.get("max_new_tokens", 32))
    max_new_seq_len = int(params.get("max_new_seq_len", 2048))
    forward_times = 0

    #all_jacobian_trajectory = []

    itr=0
    time_speed = []
    converge_step = []
    inference_time = 0
    input_echo_len = None
    finish_reason = "stop"
    eos_positions = None
    ### the following is use jacobian generate per max_new_tokens, max_seq_len is initialized
    while True:
        if itr == 0:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs['input_ids']
            input_echo_len = len(input_ids)
            input_masks = inputs['attention_mask']
        else:
            input_masks = torch.ones_like(input_ids).to(input_ids.device)
            for j in range(bsz):
                input_masks[j][torch.sum(inputs["attention_mask"], dim=-1)[j] + itr*max_new_tokens:] = 0
            
        bsz = input_ids.shape[0]
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        manual_eos_check = torch.tensor([False] * bsz, device="cuda")

        generation, itr_steps = get_jacobian_trajectory(model, tokenizer, input_ids, input_masks, max_new_tokens)
        inference_time += itr_steps
        ### tokens generated after <eos> are set to <pad>
        for j in range(bsz):
            prompt_len = torch.sum(input_masks, dim=-1)
            eos_positions = torch.where(generation[j]==tokenizer.eos_token_id)[0]
            
            # hard-coded manual check: if </s> is reached.
            # TODO: optimize this by taking <EOT> token into account during training
            manual_eos_positions = torch.where(generation[j]==27)[0]

            if len(eos_positions)==0 and len(manual_eos_positions)==0:
                # no EOS, continue to the next item in the batch
                generation[j][prompt_len[j]+ max_new_tokens:] = tokenizer.pad_token_id
                continue
            # otherwise, set tokens coming after EOS as pad 
            else:
                if len(eos_positions)!=0:
                    eos_reached[j] = True
                    generation[j, int(eos_positions[0])+1:] = tokenizer.pad_token_id
                else:
                    manual_eos_check[j] = True
                    generation[j, int(manual_eos_positions[0])+4:] = tokenizer.pad_token_id
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_ids
        itr+=1      
        
        if all(eos_reached) or all(manual_eos_check) or itr*max_new_tokens >= max_new_seq_len:
            finish_reason = "length"
            break
        input_ids = generation[torch.where(eos_reached==False)[0].tolist(), ...] # delete samples with <eos> generated
    prompt_token_len = torch.sum(inputs['attention_mask'], dim=-1)
    total_token_len = torch.sum(generation != tokenizer.pad_token_id, dim=-1)
    decoded_generation = tokenizer.decode(generation[0][prompt_token_len:eos_positions[0]])
    # print(decoded_generation)
    
    yield {
        "text": decoded_generation,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": itr*max_new_tokens,
            "total_tokens": input_echo_len + itr*max_new_tokens,
        },
        "finish_reason": finish_reason,
    }
    


def generate_stream_cllm_test(model, tokenizer, params, device, context_len, stream_interval = 2, judge_sent_end = False):
    #converge_step = []
    prompt = params["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    max_new_tokens = int(params.get("max_new_tokens", 32))
    max_new_seq_len = int(params.get("max_new_seq_len", 2048))
    forward_times = 0

    #all_jacobian_trajectory = []

    prompt_len = torch.sum(inputs['attention_mask'], dim=-1)
    generation = inputs['input_ids']
    input_echo_len = len(generation)
    ### prefill the kv-cache

    past_key_values, first_correct_token = model.jacobi_forward(input_ids=inputs['input_ids'], tokenizer=tokenizer, max_new_tokens=max_new_tokens, past_key_values=None, use_cache = True, prefill_phase = True)
    ### generation phase
    itr = 0
    eos_reached = False
    while True:
        itr+=1
        bsz = 1 # only support batch_size = 1 now
        # randomly initialize the first point of jacobian trajectory
        random_point = torch.tensor(random.choices(generation[0], k=(max_new_tokens-1)), device="cuda").view(1,-1)
        input_ids = torch.cat((first_correct_token.view(1,-1), random_point),dim=-1)
        n_gram_generation, first_correct_token, iter_steps = model.jacobi_forward(input_ids=input_ids, tokenizer=tokenizer, max_new_tokens=max_new_tokens, past_key_values=past_key_values, use_cache = True, prefill_phase = False)
        forward_times += iter_steps
        #all_jacobian_trajectory.append(jacobian_trajectory)
        
        eos_positions = torch.where(n_gram_generation[0]==tokenizer.eos_token_id)[0]

        if len(eos_positions)>0:
            eos_reached = True
        
        ### see if next max_new_tokens should be generated & if True, update weights and prepare new input_id 
        generation = torch.cat((generation, n_gram_generation), dim=-1)

        if eos_reached or itr*max_new_tokens > max_new_seq_len:
            break
    

    if eos_reached or itr*max_new_tokens > max_new_seq_len:
        finish_reason = "length"
    else:
        finish_reason = "stop"
        
    output = tokenizer.decode(generation[0], skip_special_tokens=False)

    yield {
        "text": "",
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": itr*max_new_tokens,
            "total_tokens": input_echo_len + itr*max_new_tokens,
        },
        "finish_reason": finish_reason,
    }

    # clean
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()
