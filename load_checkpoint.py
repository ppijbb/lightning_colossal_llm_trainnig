
import argparse
import warnings
import logging
from data import RandomDataloader, RandomS2SDataloader
from dataloader import LanguageDataModule
from model import GPTLitModule, get_optimizer
from s2s_model import S2SLitModule as lit_module
from callback import MemoryMonitor
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy

# from pytorch_lightning.plugins.deepspeed import Deepspeed
import warnings
warnings.filterwarnings('ignore')


# lit_module = S2SLitModule
train_dataloader = None # RandomS2SDataloader(args.steps_per_epoch, args.batch_size, args.seq_len)
data_module = LanguageDataModule(
            train_file="/data/kevin.jung/Train.csv",
            val_file="/data/kevin.jung/Dev_s.csv",
            test_file="/data/kevin.jung/Test.csv", 
            tokenizer_path="Gunulhona/tb_tokenizer_big",
            max_seq_len=1024,
            batch_size=1)
optimizer_cfg = {
        'lr': 5e-5
        }

trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': DeepSpeedStrategy(
                stage=3,
                offload_parameters=True,
                offload_optimizer=True,
                initial_scale_power=5,
                load_full_weights=True,
                logging_batch_size_per_gpu=1,
                logging_level=logging.ERROR) # 로그에 warning 너무 많이 쌓여서 추가
            }
        
optimizer_cfg['offload'] = True

opt_init_fn = get_optimizer("deepspeed", **optimizer_cfg)

model = lit_module(
        model_name="s2s_9.2B", 
        optimizer_init_fn=opt_init_fn,
        checkpoint=not True,
        cuda_mem_fraction=1.0,
        model_checkpoint_dir="/data/llm_checkpoint/last.ckpt/")
model.configure_sharded_model()

trainer = pl.Trainer(
        max_epochs=1,
        devices=-1,
        enable_checkpointing=True,
        callbacks=[
            MemoryMonitor(),
            TQDMProgressBar(
                refresh_rate=5000),
            ModelCheckpoint(
                dirpath='/data/llm_checkpoint/',
                mode="min",
                monitor="loss",
                filename="llm-{epoch:02d}-{val_loss:.4f}.ckpt",
                every_n_train_steps=1,
                save_last=True)],
        fast_dev_run=True,
        profiler="advanced",
        **trainer_cfg)

trainer.fit(
    model=model, 
    train_dataloaders=train_dataloader,
    datamodule=data_module)
 

print("\nDONE\n")
