from pytorch_lightning.loggers import CSVLogger


def get_run_logger(config, run_kwargs):

    # add the special kwargs to the config
    for key, value in run_kwargs.items():
        setattr(config, key, value)

    # add the special kwargs to the run name
    run_name = getattr(config, "run_name", "run")
    if run_kwargs:
        run_name += "_" + "_".join([f"{k}={v}" for k, v in run_kwargs.items()])

    trainer_logger = getattr(getattr(config, "trainer", None), "logger", None)
    init_args = getattr(trainer_logger, "init_args", None)
    save_dir = getattr(init_args, "save_dir", "outputs/logs")
    return CSVLogger(
        name=run_name,
        save_dir=save_dir,
    )
