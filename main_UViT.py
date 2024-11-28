import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset import LitLOLDataModule
from diffusion import LitDiffusion, EnlightDiffusion
from pytorch_lightning.strategies import DDPStrategy
import yaml
from utils.gpuoption import gpuoption
from encoder import Unet_encoder
from model_UViT import UViT

def train(config):
    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)
    
    # DDP server debug
    if config.devices[0] != 0:
        config.results_folder = config.results_folder + str(config.devices[0])

    # debug setting
    if config.fast_dev_run:
        if not config.accelerator == 'cpu':
            config.devices = 1
        # config.strategy = 'auto'
        config.use_wandb = False
        # if config.batch_size >= 2:
        #     config.batch_size //= 2

    if config.use_dataset == 'LOL':
        train_folders = [config.train_folders_v1]
    elif config.use_dataset == "LOLv2":
        train_folders = [config.train_folders_v2]
    elif config.use_dataset == 'LOL+LOLv2':
        train_folders = [config.train_folders_v1, config.train_folders_v2]
    elif config.use_dataset == 'LOL+LOLv2+VELOL':
        train_folders = [config.train_folders_v1, config.train_folders_v2, config.train_folders_VE]
    else:
        NotImplementedError("dataset not supported")
    test_folder = config.test_folder

    # seed
    pl.seed_everything(seed=config.seed, workers=True)

    # data
    litdataModule = LitLOLDataModule(config, train_folders, [test_folder])
    litdataModule.setup()

    # model        
    encoder_model = Unet_encoder(
        unet_dim=config.unet_dim, 
        in_dim=config.cond_in_dim, 
        unet_outdim=config.unet_outdim, 
        dim_mults=config.dim_mults, 
        use_attn=config.use_attn, 
        use_wn=config.use_wn, 
        use_in=config.use_in, 
        weight_init=config.weight_init,
        on_res=config.cond_on_res, 
        get_feats=False, 
        use_cond=False)

    noise_model = UViT(
        unet_dim = config.unet_dim, 
        unet_outdim = config.unet_outdim, 
        dim_mults = config.dim_mults, 
        use_ViT = config.use_ViT, 
        use_wn = config.use_wn, 
        use_instance_norm = config.use_in, 
        weight_init = config.weight_init, 
        stronger_cond = config.stronger_cond, 
        in_dim = config.in_dim,    
        dim_adjust_factor = config.dim_adjust_factor, 
        num_blocks = config.num_blocks, 
        heads = config.heads,
        ffn_expansion_factor = config.ffn_expansion_factor, 
        bias = config.bias, 
        LayerNorm_type= config.LayerNorm_type,
        skip = config.skip,
        flash_attn_valid_switch = config.flash_attn_valid_switch,
        lambda_num= config.lambda_num

)

    diffusion = EnlightDiffusion(noise_model, config)

    if config.diffusion_path != '':
        litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=encoder_model, config=config, strict=False)
    else:
        litmodel = LitDiffusion(diffusion, encoder=encoder_model, config=config)

    callbacks = [
        ModelSummary(max_depth=3),
        LearningRateMonitor(),
        ModelCheckpoint(monitor='valid/combined',
                        save_last=False, mode='max', auto_insert_metric_name=False,
                        filename='epoch={epoch:02d}-monitor={valid/combined:.2f}'),
    ]

    # strategy and trainer
    if config.strategy is not None:
        if config.strategy == 'ddp':
            config.strategy = DDPStrategy(static_graph=False, find_unused_parameters=True)
        
        trainer = pl.Trainer(
            benchmark=config.benchmark,
            enable_checkpointing=config.enable_checkpointing,
            gradient_clip_algorithm=config.gradient_clip_algorithm,
            gradient_clip_val=config.gradient_clip_val, 
            accumulate_grad_batches=config.accumulate_grad_batches,
            accelerator=config.accelerator,
            precision=config.precision,
            log_every_n_steps=config.log_every_n_steps,
            detect_anomaly=config.detect_anomaly,
            deterministic=config.deterministic,
            num_sanity_val_steps=config.num_sanity_val_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs,
            callbacks=callbacks,
            devices=config.devices,

            limit_train_batches=config.limit_train_batches,
            fast_dev_run=config.fast_dev_run,
            logger=True,
            strategy=config.strategy,   
            profiler=config.profiler,
        )
    else:
                trainer = pl.Trainer(
            benchmark=config.benchmark,
            enable_checkpointing=config.enable_checkpointing,
            gradient_clip_algorithm=config.gradient_clip_algorithm,
            gradient_clip_val=config.gradient_clip_val, 
            accumulate_grad_batches=config.accumulate_grad_batches,
            accelerator=config.accelerator,
            precision=config.precision,
            log_every_n_steps=config.log_every_n_steps,
            detect_anomaly=config.detect_anomaly,
            deterministic=config.deterministic,
            num_sanity_val_steps=config.num_sanity_val_steps,
            check_val_every_n_epoch=config.check_val_every_n_epoch,
            max_epochs=config.max_epochs,
            min_epochs=config.min_epochs,
            fast_dev_run=config.fast_dev_run,
            devices=config.devices,

            limit_train_batches=config.limit_train_batches,

            callbacks=callbacks,
            logger=True,
            # strategy=config.strategy,   
            profiler=config.profiler 
        )

    trainer.fit(model=litmodel, datamodule=litdataModule)

    # after train, test it
    if not config.fast_dev_run and config.max_epochs > 1:
        ckpt_path = trainer.checkpoint_callback.best_model_path

        litmodel = LitDiffusion.load_from_checkpoint(
            ckpt_path, diffusion_model=diffusion, encoder=encoder_model, config=config, strict=False)
    
        # new trainer
        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=[config.devices[0]] if isinstance(config.devices, list) else config.devices,
            logger=True,
            callbacks=callbacks,
            
            precision=config.precision,
            log_every_n_steps=config.log_every_n_steps,
        )
        trainer.test(litmodel, datamodule=litdataModule)

if __name__ == '__main__':
    # to make calculation faster
    torch.set_float32_matmul_precision('high')
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')

    # select config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default='cfg/train/train.yaml')
    config = parser.parse_args()

    train(config)