import sys
import os
import argparse
import numpy as np
import re
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Protein_dataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = self.tokenizer.encode(line)
        return torch.tensor(line.ids)


def load_data(file: str) -> tuple[list[str], list[str]]:
    lines = []
    prefixes = set()
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            prefix = re.match(r"<\|.*\|>", line).group(0)
            prefixes.add(prefix)
            lines.append(line)
    prefixes = sorted(list(prefixes))
    return lines, prefixes


def init_new_embeddings(model: ProGenForCausalLM, prefixes: list[str]):
    if len(prefixes) <= 2:
        logger.info("No new embeddings to initialize.")
        return
    new_embs = torch.zeros((len(prefixes) - 2, model.config.embed_dim)).to(model.device)

    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()

    # initialize new embeddings with normal distribution same as untrained embeddings
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    logger.debug(f"New embeddings shape: {new_embs.shape}")
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.config.vocab_size_emb = new_embs.shape[0]


def get_lr_schedule(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace, train_steps: int
):
    if args.decay == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=train_steps,
        )
    elif args.decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.9, last_epoch=-1
        )
    elif args.decay == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    else:
        raise ValueError(
            f"Invalid learning rate decay type. Must be 'cosine', 'linear', 'exponential', or 'constant'. Got: {args.decay}"
        )
    return scheduler


def train_epoch(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    args: argparse.Namespace,
):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(total=len(dataloader) // args.accumulation_steps)
    batch: torch.Tensor
    for i, batch in enumerate(dataloader):
        batch = batch.to(model.device)
        loss: torch.Tensor = model(batch, labels=batch).loss
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss = total_loss + loss.item()
        # using gradient accumulation to save memory
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update()
    pbar.close()
    logger.info(f"TRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}")
    logger.debug(f"Last learning rate: {scheduler.get_last_lr()}")
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: ProGenForCausalLM,
    dataset: Protein_dataset,
    args: argparse.Namespace,
    before_train: bool = False,
):
    model.eval()
    total_loss = 0
    if before_train:
        # batch_size needs to be 1 so that we dont have different lengths of rows in the tensor
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=True)
    total_length = len(dataloader)
    pbar = tqdm(total=total_length)
    batch: torch.Tensor
    for batch in dataloader:
        if before_train:
            # remove padding, because the base model wasn't trained with padding
            non_zero_length = (batch != 0).sum().item()
            batch = batch[:, :non_zero_length]
        batch = batch.to(model.device)
        loss: torch.Tensor = model(batch, labels=batch).loss
        total_loss += loss.item()
        pbar.update()
    pbar.close()
    logger.info(f"EVAL loss: {total_loss / total_length}")
    return total_loss / total_length


def train(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    train_dataset: Protein_dataset,
    test_dataset: Protein_dataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    args: argparse.Namespace,
    job_id: str,
):
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Start time of epoch {epoch}: {datetime.now()}")
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        train_losses.append(train_loss)

        logger.info(f"Running test set evaluation after {epoch} epochs:")
        eval_loss = evaluate(model, test_dataset, args)
        eval_losses.append(eval_loss)

        model_name = (job_id + "-" if job_id is not None else "") + args.model.strip(os.sep).split(os.sep)[-1]
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join("checkpoints", f"{model_name}-finetuned", f"e{epoch}")
            os.makedirs(checkpoint_path, exist_ok=True)
            
            model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"), pretty=True)

            if args.save_optimizer:
                logger.info("Saving optimizer and scheduler...")
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

            logger.info(f"Model saved at: {checkpoint_path}")
    return model, train_losses, eval_losses


def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        logger.debug(f"Slurm job id: {job_id}")
    else:
        logger.warning("No Slurm job ID found.")

    # loading data and tokenizer
    if os.path.exists(os.path.join(args.model, "tokenizer.json")):
        tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    else:
        tokenizer: Tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.enable_padding(
        direction="right", pad_id=0, pad_token="<|pad|>", length=1024
    )
    tokenizer.enable_truncation(max_length=1024)

    train_data, prefixes = load_data(args.train_file)
    test_data, prefixes_test = load_data(args.test_file)
    logger.info(f"Found prefixes: {prefixes}")
    assert prefixes == prefixes_test, "Prefixes in train and test data must be the same"
    tokenizer.add_tokens(prefixes)

    train_data = Protein_dataset(train_data, tokenizer)
    test_data = Protein_dataset(test_data, tokenizer)
    logger.debug(f"Train data size: {len(train_data)}")
    logger.debug(f"Test data size: {len(test_data)}")

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU. Please consider using a GPU for training.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.debug(f"hyperparameters: effective batch={args.batch_size * args.accumulation_steps}, {args.batch_size=}, {args.accumulation_steps=}, {args.epochs=}, {args.lr=}, {args.warmup_steps=}, {args.checkpoint_rate=}")

    # loading model
    logger.info(f"Loading model: {args.model}...")
    model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    logger.info(f"Model loaded. Parameter count: {model.num_parameters() // 1e6} M")
    init_new_embeddings(model, prefixes)

    # creating optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = (
        args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps)
    )
    if training_steps > 0:
        logger.debug(f"Weight updates per epoch: {training_steps / args.epochs}")
    logger.debug(f"Total weight updates: {training_steps}")
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    if args.eval_before_train:
        logger.info("Runnning evaluation on test set before training...")
        evaluate(model, test_data, args, before_train=True)

    # training loop
    model, train_losses, test_losses = train(
        model,
        tokenizer,
        train_data,
        test_data,
        optimizer,
        scheduler,
        args,
        job_id,
    )

    logger.info("Finetuning finished.")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Test losses: {test_losses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small",
        help="Name of the model checkpoint to be finetuned.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on. Default: \"cuda\"",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data file. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test data file. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
        help="How many steps to accumulate gradients before updating weights. Default: 4",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate. Check out also the '--decay' argument. Default: 1e-4",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
        help="Number of warmup steps for learning rate scheduler. Linearly increasing form 0 to --lr. Default: 200",
    )
    parser.add_argument(
        "--checkpoint_rate", type=int, default=5, help="Save model checkpoint every n epochs. Default: 5"
    )
    parser.add_argument(
        "--decay",
        type=str,
        choices=["cosine", "linear", "constant"],
        default="cosine",
        help="Learning rate decay. Default: \"cosine\"",
    )
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        default=False,
        help="Should we also save the optimizer and scheduler at every checkpoint",
    )
    parser.add_argument(
        "--eval_before_train",
        action="store_true",
        default=False,
        help="Run evaluation on test set before training. default: False",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging level.",
    )
    args = parser.parse_args()

    main(args)
