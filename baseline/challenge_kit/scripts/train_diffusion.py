import torch
from lvdm.ema import LitEma
from lvdm.utils.train import get_env_vars, get_model, get_parser, get_trainer, prepare_logger, set_model_lr
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

if __name__ == "__main__":
    now, local_rank, global_rank, num_rank = get_env_vars()

    # Extends existing argparse by default Trainer attributes
    parser = get_parser()
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank (0 = disabled)")
    parser.add_argument("--lora-alpha", type=float, default=4.0, help="LoRA alpha scaling")
    parser.add_argument("--unet-checkpoint", type=str, default=None,
                        help="Path to a full model checkpoint to load UNet weights before LoRA injection")
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    seed_everything(args.seed)

    # yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # setup workspace directories and logger
    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(lightning_config, config, global_rank, now)

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    model = get_model(config.model, workdir)
    model = set_model_lr(model, config.model, num_rank, config.data.params.batch_size)

    # Load full model weights (UNet included) from a separate checkpoint before LoRA injection
    if args.unet_checkpoint:
        pl_sd = torch.load(args.unet_checkpoint, map_location="cpu")
        state_dict = pl_sd.get("state_dict", pl_sd)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[unet-checkpoint] Loaded {args.unet_checkpoint} | missing={len(missing)} unexpected={len(unexpected)}")

    # Inject LoRA adapters and freeze non-LoRA params
    if args.lora_rank > 0:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from lora_utils import apply_lora_to_model
        apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # ensure ema is initialised to reloaded checkpoint if reloaded
    if model.use_ema:
        model.model_ema = LitEma(model.model)

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = instantiate_from_config(config.data)
    data.setup()

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trainer = get_trainer(
        lightning_config=lightning_config,
        trainer_config=trainer_config,
        config=config,
        args=args,
        workdir=workdir,
        ckptdir=ckptdir,
        logger=logger,
    )

    ## TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trainer.fit(model, data)
