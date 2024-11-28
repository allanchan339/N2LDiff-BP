# for evaluate LOL dataset on model
import argparse
import yaml
from diffusion import LitDiffusion, EnlightDiffusion
from encoder import Unet_encoder
from model_UViT import UViT
from dataset import LitLOLDataModule
import pytorch_lightning as pl
from utils.gpuoption import gpuoption

def main(config):
    with open(config.cfg, "r") as infile:
        cfg = yaml.full_load(infile)

    config = argparse.Namespace(**cfg)

    # seed
    pl.seed_everything(seed=config.seed, workers=True)
    
    litdataModule = LitLOLDataModule(config, [''], [config.test_folder])
    litdataModule.setup(stage='test')

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
        use_cond=False
        )

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
)
    diffusion = EnlightDiffusion(noise_model, config)

    assert config.diffusion_path != '', "diffusion.path must be a valid path"
    litmodel = LitDiffusion.load_from_checkpoint(
            config.diffusion_path, diffusion_model=diffusion, encoder=encoder_model, config=config, strict=False, map_location='cpu')

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        precision=config.precision,
        fast_dev_run=config.fast_dev_run,
        devices=config.devices,
        logger=True,
    )

    trainer.test(model=litmodel,
                        datamodule=litdataModule)

if __name__ == "__main__":
    # to fix 4090 NCCL P2P bug in driver
    if gpuoption():
        print('NCCL P2P is configured to disabled, new driver should fix this bug')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", default='cfg/test/test.yaml')
    config = parser.parse_args()

    main(config)