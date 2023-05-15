from typing import Optional

import transformers
from transformers.trainer import *


def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    """
    Will save the model, so you can reload it using `from_pretrained()`.

    Will only save from the main process.
    """

    if output_dir is None:
        output_dir = self.args.output_dir

    if is_torch_tpu_available():
        self._save_tpu(output_dir)
    elif is_sagemaker_mp_enabled():
        # Calling the state_dict needs to be done on the wrapped model and on all processes.
        os.makedirs(output_dir, exist_ok=True)
        state_dict = self.model_wrapped.state_dict()
        if self.args.should_save:
            self._save(output_dir, state_dict=state_dict)
        if IS_SAGEMAKER_MP_POST_1_10:
            # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
            Path(os.path.join(output_dir, "user_content.pt")).touch()
    elif (
            ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
            or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            or self.fsdp is not None
    ):
        if self.fsdp:
            from torch.distributed.fsdp.api import (
                FullOptimStateDictConfig, FullStateDictConfig, StateDictType)
            import torch.distributed.fsdp.fully_sharded_data_parallel as FSDP
            FSDP.FullyShardedDataParallel.\
                set_state_dict_type(self.model, StateDictType.FULL_STATE_DICT,
                                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                    FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True))
        state_dict = self.model.state_dict()

        if self.args.should_save:
            self._save(output_dir, state_dict=state_dict)
    elif self.deepspeed:
        # this takes care of everything as long as we aren't under zero3
        if self.args.should_save:
            self._save(output_dir)

        if is_deepspeed_zero3_enabled():
            # It's too complicated to try to override different places where the weights dump gets
            # saved, so since under zero3 the file is bogus, simply delete it. The user should
            # either user deepspeed checkpoint to resume or to recover full weights use
            # zero_to_fp32.py stored in the checkpoint.
            if self.args.should_save:
                file = os.path.join(output_dir, WEIGHTS_NAME)
                if os.path.isfile(file):
                    # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                    os.remove(file)

            # now save the real model if stage3_gather_16bit_weights_on_model_save=True
            # if false it will not be saved.
            # This must be called on all ranks
            if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                logger.warning(
                    "deepspeed.save_16bit_model didn't save the model, since"
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                self.deepspeed.save_checkpoint(output_dir)

    elif self.args.should_save:
        self._save(output_dir)

    # Push to the Hub when `save_model` is called by the user.
    if self.args.push_to_hub and not _internal_call:
        self.push_to_hub(commit_message="Model save")


def replace_hf_save_model():
    transformers.Trainer.save_model = save_model
