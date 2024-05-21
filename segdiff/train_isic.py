"""
Train a diffusion model on images.
"""

import argparse
import datetime
import json
import os
from pathlib import Path

import git
from mpi4py import MPI

from improved_diffusion import dist_util, logger
from datasets.vaih import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.utils import set_random_seed, set_random_seed_for_iterations
from dataset import ISICDataset
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def main():

    ## All the arguments
    args = create_argparser().parse_args()
    args.use_fp16 = True
    args.clip_denoised = False
    args.learn_sigma = False
    args.sigma_small = False
    args.num_channels = 128
    args.image_size = 256
    args.num_res_blocks = 3
    args.noise_schedule = "linear"
    args.rescale_learned_sigmas = False
    args.rescale_timesteps = False
    args.use_scale_shift_norm = False
    args.deeper_net = True

    exp_name = f"vaih_256_{args.rrdb_blocks}_{args.lr}_{args.batch_size}_{args.diffusion_steps}_{str(args.dropout)}_{MPI.COMM_WORLD.Get_rank()}"

    ## Logs of the run
    logs_root = Path(__file__).absolute().parent.parent / "logs"
    log_path = logs_root / f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{exp_name}"
    os.environ["OPENAI_LOGDIR"] = str(log_path)
    set_random_seed(MPI.COMM_WORLD.Get_rank(), deterministic=True)
    set_random_seed_for_iterations(MPI.COMM_WORLD.Get_rank())
    dist_util.setup_dist()
    logger.configure(dir=str(log_path))

    if args.resume_checkpoint:
        resumed_checkpoint_arg = args.resume_checkpoint
        args.__dict__.update(json.loads((Path(args.resume_checkpoint) / 'args.json').read_text()))
        args.resume_checkpoint = resumed_checkpoint_arg

    logger.info(args.__dict__)

    (Path(log_path) / 'args.json').write_text(json.dumps(args.__dict__, indent=4))
    logger.info(f"log folder path: {Path(log_path).resolve()}")

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    logger.log(f"git commit hash {sha}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    transform_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(), ]
    transform_train = transforms.Compose(transform_list)

    train_dataset = ISICDataset(
        training=True,
        data_path="/users/xwang259/CSCI1430-Final-Project-MedImage-Segmentation/CSCI1430-Final-Project-MedImage-Segmentation/data/ISIC",
        csv_file="ISBI2016_ISIC_Part3B_Training_GroundTruth.csv",
        img_folder="ISBI2016_ISIC_Part3B_Training_Data",
        transform=transform_train,
        flip_p=0.5
    )

    val_dataset = ISICDataset(
        training=False,
        data_path="/users/xwang259/CSCI1430-Final-Project-MedImage-Segmentation/CSCI1430-Final-Project-MedImage-Segmentation/data/ISIC",
        csv_file="ISBI2016_ISIC_Part3B_Test_GroundTruth.csv",
        img_folder="ISBI2016_ISIC_Part3B_Test_Data",
        flip_p=0.5
    )

    ## Loading the data and the validation data. The data is arleady batched.
    data = load_data(
        dataset=train_dataset,
        batch_size=args.batch_size
    )

    logger.log(f"gpu {MPI.COMM_WORLD.Get_rank()} / {MPI.COMM_WORLD.Get_size()} val length {len(val_dataset)}")

    ## Training the model
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip_denoised=args.clip_denoised,
        logger=logger,
        image_size=args.image_size,
        val_dataset=val_dataset,
        run_without_test=args.run_without_test,
        args=args
        # dist_util=dist_util,
    ).run_loop(max_iter=300000, start_print_iter=args.start_print_iter)


def load_data(
    *, dataset, batch_size, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
    while True:
        yield from loader


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.00002,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip_denoised=False,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        save_interval=1000,
        start_print_iter=75000,
        log_interval=200,
        run_without_test=False,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()