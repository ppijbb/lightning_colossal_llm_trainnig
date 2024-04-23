import gc
import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, BartModel, BartConfig, BartPretrainedModel
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import colo_set_process_memory_fraction

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.accelerator import get_accelerator
from torch.optim import Adam, Optimizer
from functools import partial
from typing import Callable, Iterable
from contextlib import contextmanager
__all__ = ['S2SLitModule', 'get_optimizer']


@contextmanager
def no_init_weights():
    def dummy_fn(*args):
        return
    try:
        old_init_weights = BartPretrainedModel._init_weights
        BartPretrainedModel._init_weights = dummy_fn
        yield
    finally:
        BartPretrainedModel._init_weights = old_init_weights


class S2SLMModel(nn.Module):
    def __init__(self, 
                 hidden_size:int=768, 
                 num_layers:int=12, 
                 num_attention_heads:int=12,
                 prompt_layers:int=3, 
                 max_seq_len:int=1024, 
                 vocab_size:int=64512, # 50257
                 checkpoint:bool=False):
        super().__init__()
        self.checkpoint = checkpoint
        with no_init_weights():
            self.model = BartForConditionalGeneration(
                BartConfig(
                    d_model=hidden_size, 
                    encoder_layers=prompt_layers,
                    decoder_layers=num_layers,
                    encoder_head=num_attention_heads, 
                    decoder_head=num_attention_heads, 
                    max_position_embeddings=max_seq_len, 
                    n_ctx=max_seq_len, 
                    vocab_size=vocab_size))
        if checkpoint:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # Only return lm_logits
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask, 
            use_cache=not self.checkpoint)[0]


def s2s_tiny(checkpoint=True):
    return S2SLMModel(hidden_size=128, num_layers=4, num_attention_heads=4, checkpoint=checkpoint)


def s2s_small(checkpoint=True):
    return S2SLMModel(hidden_size=768, num_layers=12, num_attention_heads=12, checkpoint=checkpoint)


def s2s_medium(checkpoint=True):
    return S2SLMModel(hidden_size=1024, num_layers=24, num_attention_heads=16, checkpoint=checkpoint)


def s2s_large(checkpoint=True):
    return S2SLMModel(hidden_size=1280, num_layers=36, num_attention_heads=20, checkpoint=checkpoint)


def s2s_xl(checkpoint=True):
    return S2SLMModel(hidden_size=1600, num_layers=48, num_attention_heads=25, checkpoint=checkpoint)


def s2s_2B(checkpoint=True):
    return S2SLMModel(hidden_size=2048, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def s2s_9B(checkpoint=True):
    return S2SLMModel(hidden_size=2048, num_layers=178, num_attention_heads=16, checkpoint=checkpoint)


def s2s_3B(checkpoint=True):
    return S2SLMModel(hidden_size=2304, num_layers=48, num_attention_heads=16, checkpoint=checkpoint)


def s2s_4B(checkpoint=True):
    return S2SLMModel(hidden_size=2304, num_layers=64, num_attention_heads=16, checkpoint=checkpoint)


def s2s_6B(checkpoint=True):
    return S2SLMModel(hidden_size=4096, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def s2s_8B(checkpoint=True):
    return S2SLMModel(hidden_size=3072, num_layers=72, num_attention_heads=24, checkpoint=checkpoint)


def s2s_12B(checkpoint=True):
    return S2SLMModel(hidden_size=4096, num_layers=60, num_attention_heads=16, checkpoint=checkpoint)


def s2s_15B(checkpoint=True):
    return S2SLMModel(hidden_size=4096, num_layers=78, num_attention_heads=16, checkpoint=checkpoint)


def s2s_18B(checkpoint=True):
    return S2SLMModel(hidden_size=4096, num_layers=90, num_attention_heads=16, checkpoint=checkpoint)


def s2s_20B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=25, num_attention_heads=16, checkpoint=checkpoint)


def s2s_24B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=30, num_attention_heads=16, checkpoint=checkpoint)


def s2s_28B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=35, num_attention_heads=16, checkpoint=checkpoint)


def s2s_32B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=40, num_attention_heads=16, checkpoint=checkpoint)


def s2s_36B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=45, num_attention_heads=16, checkpoint=checkpoint)


def s2s_40B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=50, num_attention_heads=16, checkpoint=checkpoint)


def s2s_45B(checkpoint=True):
    return S2SLMModel(hidden_size=8192, num_layers=56, num_attention_heads=16, checkpoint=checkpoint)


def s2s_LLM(checkpoint=True):
    return S2SLMModel(max_seq_len=2048, hidden_size=12288, num_layers=96, num_attention_heads=96, checkpoint=checkpoint)


def get_s2s_model(model_name: str, checkpoint: bool = True) -> nn.Module:
    model_map = {
        's2s_tiny': s2s_tiny,
        's2s_small': s2s_small,
        's2s_medium': s2s_medium,
        's2s_large': s2s_large,
        's2s_xl': s2s_xl,
        's2s_2B': s2s_2B,
        's2s_9.2B': s2s_9B,
        's2s_3B': s2s_3B,
        's2s_4B': s2s_4B,
        's2s_6B': s2s_6B,
        's2s_8B': s2s_8B,
        's2s_12B': s2s_12B,
        's2s_15B': s2s_15B,
        's2s_18B': s2s_18B,
        's2s_20B': s2s_20B,
        's2s_24B': s2s_24B,
        's2s_28B': s2s_28B,
        's2s_32B': s2s_32B,
        's2s_36B': s2s_36B,
        's2s_40B': s2s_40B,
        's2s_45B': s2s_45B,
        's2s_LLM': s2s_LLM,
    }
    assert model_name in model_map
    # print(f"Training model is {model_name}")
    return model_map[model_name](checkpoint)


class GPTLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_optimizer(strategy: str, **kwargs) -> Callable[[Iterable], Optimizer]:
    assert strategy in ('ddp', 'deepspeed', 'colossal')
    if strategy == 'ddp':
        opt_cls = Adam
    elif strategy == 'deepspeed':
        offload = kwargs.pop('offload')
        if offload:
            opt_cls = DeepSpeedCPUAdam
        else:
            opt_cls = FusedAdam
    else:
        opt_cls = HybridAdam
    return partial(opt_cls, **kwargs)


class S2SLitModule(pl.LightningModule):
    def __init__(self, 
                 model_name: str, 
                 optimizer_init_fn: Callable[[Iterable], Optimizer],
                 checkpoint: bool = True, 
                 cuda_mem_fraction: float = 1.0,
                 model_checkpoint_dir: str = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.optimizer_init_fn = optimizer_init_fn
        self.checkpoint = checkpoint
        self.criterion = GPTLMLoss()
        self.cuda_mem_fraction = cuda_mem_fraction
        self.model_checkpoint = model_checkpoint_dir

    def _save_in_hub_(self)->None:
        # print(self.model.model)
        self.model.model.save_pretrained(
            save_directory="/data/llm_checkpoint/checkpoint/",#
            # use_temp_dir=False,
            push_to_hub=True,
            max_shard_size="124MB",
            # safe_serialization=True,
            repo_id=os.getenv("MODEL_SAVE_REPO"),
            use_auth_token=os.getenv("HUGGINGFACE_AUTO_TOKEN"))

    def __memory_clean__(self)->None:
        get_accelerator().empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

    def configure_sharded_model(self) -> None:
        self.model = get_s2s_model(
            model_name=self.model_name,
            checkpoint=self.checkpoint)

    # def on_save_checkpoint(self, checkpoint)->None:
    #     self._save_in_hub_()

    def on_load_checkpoint(self, checkpoint) -> None:
        if not hasattr(self, 'model'):
            self.configure_sharded_model()
        if self.model_checkpoint != None:
            self.model.model.load_state_dict(
                get_fp32_state_dict_from_zero_checkpoint(self.model_checkpoint))

    def configure_optimizers(self):
        return self.optimizer_init_fn(self.model.parameters())

    def training_step(self, batch, batch_idx):
        if type(batch) is dict:
            loss_list= torch.empty(0,device=torch.cuda.current_device())
            for loader in batch:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch[loader].values()
                logits = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids, 
                        decoder_attention_mask=decoder_attention_mask)
                loss = self.criterion(logits, labels)
                loss_list = torch.cat((loss_list, loss.view(-1)))    
            loss = loss_list.sum()
            return loss
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
            logits = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask)
            loss = self.criterion(logits, input_ids)
            return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) is dict:
            loss_list= torch.empty(0,device=torch.cuda.current_device())
            for loader in batch:
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch[loader].values()
                logits = self.model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids, 
                        decoder_attention_mask=decoder_attention_mask)
                loss = self.criterion(logits, labels)
                loss_list = torch.cat((loss_list, loss.view(-1)))    
            loss = loss_list.sum()
        else:
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = batch
            logits = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask)
            loss = self.criterion(logits, labels)

    def on_training_batch_end(self, outputs, batch, batch_idx):
        self.__memory_clean__()

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self.__memory_clean__()

    def on_fit_start(self) -> None:
        if self.cuda_mem_fraction < 1.0:
            colo_set_process_memory_fraction(self.cuda_mem_fraction)
            
