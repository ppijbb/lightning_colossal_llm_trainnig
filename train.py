import pytorch_lightning as pl
import argparse
import warnings
import logging
from data import RandomDataloader, RandomS2SDataloader
from dataloader import LanguageDataModule
from model import GPTLitModule, get_optimizer
from s2s_model import S2SLitModule
from callback import MemoryMonitor
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.strategies.colossalai import ColossalAIStrategy
# from pytorch_lightning.plugins.deepspeed import Deepspeed
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tqdm_rate', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--steps_per_epoch', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model', default='gpt2_xl')
    parser.add_argument('--np', type=int, default=1)
    parser.add_argument('--no_activation_ckpt', action='store_true', default=False)
    parser.add_argument('--opt_nvme_offload_frac', type=float, default=0.0)
    parser.add_argument('--opt_nvme_offload_dir', default='./offload')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--placement_policy', default='cuda')
    parser.add_argument('--opt_gpu_margin_rat', type=float, default=0.0)
    parser.add_argument('--cuda_mem_frac', type=float, default=1.0)
    parser.add_argument('--strategy', default='ddp', choices=['ddp', 'colossal', 'deepspeed'])
    parser.add_argument('--offload', action='store_true', default=False)
    parser.add_argument('--model_checkpoint_dir', default='/data/llm_checkpoint/')
    parser.add_argument('--model_zero_ckpt_dir', default=None)
    args = parser.parse_args()
    
    if "gpt" in args.model:
        lit_module = GPTLitModule
        train_dataloader = RandomDataloader(args.steps_per_epoch, args.batch_size, args.seq_len)
        data_module = None
    else:
        lit_module = S2SLitModule
        train_dataloader = None # RandomS2SDataloader(args.steps_per_epoch, args.batch_size, args.seq_len)
        data_module = LanguageDataModule(
            train_file="/data/kevin.jung/Train.csv",
            val_file="/data/kevin.jung/Dev_s.csv",
            test_file="/data/kevin.jung/Test.csv", 
            tokenizer_path="Gunulhona/tb_tokenizer_big",
            max_seq_len=args.seq_len,
            batch_size=args.batch_size)

    optimizer_cfg = {
        'lr': args.lr
        }

    if args.strategy == 'ddp':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': DDPStrategy(static_graph=True)
        }

    elif args.strategy == 'colossal':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': ColossalAIStrategy(
                placement_policy=args.placement_policy,
                gpu_margin_mem_ratio=args.opt_gpu_margin_rat,
                initial_scale=32,
                chunk_search_range= 64 * 1024**2,
                chunk_search_n_grids= 4096,
                min_chunk_size= 32 * 1024**2)
            }
        
        optimizer_cfg['nvme_offload_dir'] = args.opt_nvme_offload_dir
        optimizer_cfg['nvme_offload_fraction'] = args.opt_nvme_offload_frac

    elif args.strategy == 'deepspeed':
        trainer_cfg = {
            'accelerator': 'gpu',
            'precision': 16,
            'strategy': DeepSpeedStrategy(
                stage=3,
                offload_parameters=args.offload,
                offload_optimizer=args.offload,
                initial_scale_power=5,
                load_full_weights=True,
                logging_batch_size_per_gpu=args.batch_size,
                logging_level=logging.ERROR) # 로그에 warning 너무 많이 쌓여서 추가
            }
        
        optimizer_cfg['offload'] = args.offload

    opt_init_fn = get_optimizer(args.strategy, **optimizer_cfg)

    model = lit_module(
        model_name=args.model, 
        optimizer_init_fn=opt_init_fn,
        checkpoint=not args.no_activation_ckpt,
        cuda_mem_fraction=args.cuda_mem_frac,
        model_checkpoint_dir=args.model_zero_ckpt_dir)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.np,
        enable_checkpointing=True,
        callbacks=[
            MemoryMonitor(),
            TQDMProgressBar(
                refresh_rate=args.tqdm_rate),
            ModelCheckpoint(
                dirpath=args.model_checkpoint_dir,
                mode="min",
                monitor="loss",
                filename="llm-{epoch:02d}-{val_loss:.4f}.ckpt",
                every_n_train_steps=1,
                save_last=True)],
        fast_dev_run=False,
        profiler="advanced",
        **trainer_cfg)
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        datamodule=data_module)
