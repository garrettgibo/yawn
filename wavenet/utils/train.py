"""Training script.

This loosely follows this example:
https://github.com/IgorSusmelj/pytorch-styleguide
"""
import time

import torch
import torch.nn as nn
import wavenet.utils as utils
from tqdm import tqdm

logger = utils.new_logger(__name__)


def train(
    model,
    model_name: str,
    train_dataloader,
    num_epochs: int,
    learning_rate: float,
    resume: bool = False,
    checkpoint_path: str = None,
    save_every: int = 10,
):
    """Train model.

    Args:
        model: Desired model to train
        train_dataloader: PyTorch dataloader with custom dataset
        num_epochs: Additional number of epochs to train
        learning_rate: Learning rate for optimizer
        resume: Whether to resume training
        checkpoint_path: Path to model state_dict to load from if resuming train
        save_every: Number of epochs inbetween saves
    """
    current_time = time.strftime("%m_%d_%y_%H_%M_%S", time.localtime())
    model_name = model_name + current_time

    criterion = nn.CrossEntropyLoss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # create optimizers
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # load checkpoint if needed/ wanted
    start_epoch = 0
    if resume:
        model, optim, start_epoch = utils.load_model(model, optim, checkpoint_path)

    # Main training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # set model to train mode
        model.train()

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        start_time = time.time()

        for _, (data, target) in pbar:
            # send data to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            prepare_time = start_time - time.time()

            # forward and backward pass
            out = model(data)
            loss = criterion(out, target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f"Compute efficiency: {compute_efficiency:.2f}, "
                f"loss: {loss.item():.2f},  epoch: {epoch}/{num_epochs}"
            )
            start_time = time.time()

            # maybe do a test pass every N=1 epochs
            if epoch % save_every == save_every - 1:
                ckpt_name = f"{model_name}_epoch_{epoch}.pt"
                utils.save_model(model, optim, epoch, ckpt_name)

    ckpt_name = f"{model_name}_fin.pt"
    utils.save_model(model, optim, epoch, ckpt_name)

    return ckpt_name
