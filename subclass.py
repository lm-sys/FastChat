"""sampling code pulled from Transformers & slightly modified to stream tokens"""
import warnings
from typing import List, Optional,  Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import  LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.generation.utils import SampleOutput, SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from transformers import LlamaForCausalLM

class YieldingLlama(LlamaForCausalLM):
    """Overriding sample to yield tokens"""
    def sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: Optional[bool] = False,
            **model_kwargs,
        ) -> Union[SampleOutput, torch.LongTensor]:
            r"""
            Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
            can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

            <Tip warning={true}>

            In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
            For an overview of generation strategies and code examples, check the [following
            guide](./generation_strategies).

            </Tip>

            Parameters:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation.
                logits_processor (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                    used to modify the prediction scores of the language modeling head applied at each generation step.
                stopping_criteria (`StoppingCriteriaList`, *optional*):
                    An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                    used to tell if the generation loop should stop.
                logits_warper (`LogitsProcessorList`, *optional*):
                    An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                    to warp the prediction score distribution of the language modeling head applied before multinomial
                    sampling at each generation step.
                max_length (`int`, *optional*, defaults to 20):
                    **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                    tokens. The maximum length of the sequence to be generated.
                pad_token_id (`int`, *optional*):
                    The id of the *padding* token.
                eos_token_id (`int`, *optional*):
                    The id of the *end-of-sequence* token.
                output_attentions (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                    returned tensors for more details.
                output_hidden_states (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                    for more details.
                output_scores (`bool`, *optional*, defaults to `False`):
                    Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
                return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                    Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                synced_gpus (`bool`, *optional*, defaults to `False`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                model_kwargs:
                    Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                    an encoder-decoder model the kwargs should include `encoder_outputs`.

            Return:
                [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
                A `torch.LongTensor` containing the generated tokens (default behaviour) or a
                [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
                `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
                `model.config.is_encoder_decoder=True`.

            Examples:

            ```python
            >>> from transformers import (
            ...     AutoTokenizer,
            ...     AutoModelForCausalLM,
            ...     LogitsProcessorList,
            ...     MinLengthLogitsProcessor,
            ...     TopKLogitsWarper,
            ...     TemperatureLogitsWarper,
            ...     StoppingCriteriaList,
            ...     MaxLengthCriteria,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id
            >>> model.generation_config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList(
            ...     [
            ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
            ...     ]
            ... )
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList(
            ...     [
            ...         TopKLogitsWarper(50),
            ...         TemperatureLogitsWarper(0.7),
            ...     ]
            ... )

            >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

            >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
            >>> outputs = model.sample(
            ...     input_ids,
            ...     logits_processor=logits_processor,
            ...     logits_warper=logits_warper,
            ...     stopping_criteria=stopping_criteria,
            ... )

            >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
            ```"""
            # init values
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
            if max_length is not None:
                warnings.warn(
                    "`max_length` is deprecated in this function, use"
                    " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                    UserWarning,
                )
                stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
            logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
            pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
            eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
            output_attentions = (
                output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )
            return_dict_in_generate = (
                return_dict_in_generate
                if return_dict_in_generate is not None
                else self.generation_config.return_dict_in_generate
            )

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.config.is_encoder_decoder:
                encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_hidden_states = (
                    model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

            this_peer_finished = False  # used by synced_gpus only
            # auto-regressive generation
            while True:
                if synced_gpus:
                    # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                    # The following logic allows an early break if all peers finished generating their sequence
                    this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                    # send 0.0 if we finished, 1.0 otherwise
                    dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                    # did all peers finish? the reduced sum will be 0.0 then
                    if this_peer_finished_flag.item() == 0.0:
                        break

                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

                if synced_gpus and this_peer_finished:
                    continue  # don't waste resources running the code we don't need

                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

                # stop when each sentence is finished, or if we exceed the maximum length
                if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                    if not synced_gpus:
                        break
                    else:
                        this_peer_finished = True
                else:
                    yield next_tokens

            if return_dict_in_generate:
                if self.config.is_encoder_decoder:
                    yield SampleEncoderDecoderOutput(
                        sequences=input_ids,
                        scores=scores,
                        encoder_attentions=encoder_attentions,
                        encoder_hidden_states=encoder_hidden_states,
                        decoder_attentions=decoder_attentions,
                        cross_attentions=cross_attentions,
                        decoder_hidden_states=decoder_hidden_states,
                    )
                else:
                    yield SampleDecoderOnlyOutput(
                        sequences=input_ids,
                        scores=scores,
                        attentions=decoder_attentions,
                        hidden_states=decoder_hidden_states,
                    )
            else:
                yield next_tokens