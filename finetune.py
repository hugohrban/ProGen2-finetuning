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
    print("initializing new embeddings")
    #assert len(prefixes) >= 2, "must have at least 2 new embeddings"
    if len(prefixes) <= 2:
        print("No new embeddings to initialize")
        return
    new_embs = torch.zeros((len(prefixes) - 2, model.config.n_embd)).to(model.device)
    
    unk_token_emb: torch.Tensor = model.transformer.wte.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()
    # print("unk token emb", unk_token_emb)
    # print("mean_unk_emb", mean_unk_emb)
    # print("std_unk_emb", std_unk_emb)

    # initialize new embeddings with normal distribution same as untrained embeddings
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.config.vocab_size_emb = new_embs.shape[0]
    
#    print(new_embs)
#    print("new embs shape", new_embs.shape)


def get_lr_schedule(optimizer, args, train_steps):
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
        raise ValueError(f"Invalid learning rate decay type. Must be 'cosine', 'linear', or 'constant'. Got: {args.decay}")
    return scheduler


def train_epoch(model: ProGenForCausalLM, dataset: Protein_dataset, optimizer, scheduler, epoch, args, eval_dataset=None):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(total=len(dataloader) // args.accumulation_steps)
    for i, batch in enumerate(dataloader):
        batch = batch.to(args.device)
        loss = model(batch, labels=batch).loss
        loss = loss / args.accumulation_steps
        loss.backward()
        total_loss = total_loss + loss.item()
        # using gradient accumulation to save memory
        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update()
            #print(f"\rTRAIN step: {i // args.accumulation_steps} / {len(dataloader) // args.accumulation_steps}: loss: {loss.item()}", end="")
        # evaluate on test set in the middle of training epoch
        if eval_dataset is not None and i == len(dataloader) // 2:
            print(f"Running test set evaluation after {epoch}.5 epochs...")
            evaluate(model, eval_dataset, args)
    pbar.close()
    print(f"\nTRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}")
    print(f"Last learning rate: {scheduler.get_last_lr()}")
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataset, args, before_train=False):
    model.eval()
    total_loss = 0
    if before_train:
        # batch_size needs to be 1 so that we dont have different lengths of rows in the tensor
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size * 4, shuffle=True)
    total_length = len(dataloader) if not before_train else 5000
    pbar = tqdm(total=total_length)
    for i, batch in enumerate(dataloader):
        if before_train:
            # truncate padding, because the base model wasn't trained with padding
            non_zero_length = (batch != 0).sum().item()
            batch = batch[:, :non_zero_length]
            if i == 5000:
                break
        batch = batch.to(args.device)
        loss = model(batch, labels=batch).loss
        total_loss += loss.item()
        pbar.update()
        #print(f"\rEVAL step: {i} / {total_length}: loss: {loss.item()}", end="")
    pbar.close()
    print(f"\nEVAL loss: {total_loss / total_length}")
    return total_loss / total_length


def train(
    model,
    tokenizer, 
    train_dataset,
    test_dataset,
    optimizer,
    scheduler,
    args,
    job_id,
):
    train_losses = []
    eval_losses = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n=========================")
        print(f"Start time of epoch {epoch}: {datetime.now()}")
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args)
        train_losses.append(train_loss)
        print(f"\nRunning test set evaluation after {epoch} epochs:")
        eval_loss = evaluate(model, test_dataset, args)
        eval_losses.append(eval_loss)
        model_name = job_id + "-" + args.model.strip("/").split("/")[-1]
        if epoch % args.checkpoint_rate == 0 or epoch == args.epochs:
            dir_path = f"./checkpoints/{model_name}-finetuned/e{epoch}/"
            os.makedirs(dir_path, exist_ok=True)
            print(f"Saving model, optimizer, and scheduler at epoch {epoch}...")
            # torch.save(model.state_dict(), dir_path + "pytorch_model.bin")
            model.save_pretrained(dir_path)
            tokenizer.save(os.path.join(dir_path, "tokenizer.json"), pretty=True)
            if args.save_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(dir_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(dir_path, "scheduler.pt"))
            print(f"Model saved at: {dir_path}")
    return model, train_losses, eval_losses




def main(args: argparse.Namespace):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #print("Finetuning start time:", datetime.now())

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        print(f"slurm job id: {job_id}")
    else:
        print("no slurm job id found")

    tokenizer: Tokenizer = Tokenizer.from_file("tokenizer.json")
    tokenizer.enable_padding(
        direction="right", pad_id=0, pad_token="<|pad|>", length=1024
    )
    tokenizer.enable_truncation(max_length=1024)

    train_data, prefixes = load_data(args.train_file)
    test_data, prefixes_test = load_data(args.test_file)

    print("Found prefixes:", prefixes)
    #print("Test prefixes", prefixes_test)
    assert prefixes == prefixes_test, "Prefixes in train and test data must be the same"

    # print(train_data[0])
    tokenizer.add_tokens(prefixes)
    train_data = Protein_dataset(train_data, tokenizer)
    test_data = Protein_dataset(test_data, tokenizer)

    print(train_data[0])
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))
    
    # return 0

    # prefixes = list(prefixes)
    # tokenizer.add_tokens(prefixes)
    # print("prefixes:", prefixes)
    #print("Tokenizer vocab:", tokenizer.get_vocab())
    #print("Tokenizer vocab size:", tokenizer.get_vocab_size())

    
    # return 0
    
    # saving the expanded tokenizer
    # tokenizer.save("tokenizer_expanded.json", pretty=True)
    # return 0


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"hyperparameters: effective batch={args.batch_size * args.accumulation_steps}, {args.batch_size=}, {args.accumulation_steps=}, {args.epochs=}, {args.lr=}, {args.warmup_steps=}, {args.checkpoint_rate=}"
    )

    print(f"Using device: {args.device}")
    print(f"Loading model: {args.model}...")
    model = ProGenForCausalLM.from_pretrained(args.model).to(args.device)
    print("Model loaded")
    print(f"Model parameters: {model.num_parameters() // 1e6} M")
    init_new_embeddings(model, prefixes)
    
    
    # print(model.config)
    # model.save_pretrained("new_config_dummy_model")
    # return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training_steps = (
        args.epochs * len(train_data) // (args.batch_size * args.accumulation_steps)
    )
    if training_steps > 0:
        print("Weight updates per epoch", training_steps / args.epochs)
    print("Total weight updates:", training_steps)
    scheduler = get_lr_schedule(optimizer, args, training_steps)

    if args.eval_before_train:
        print("Runnning evaluation on test set before training...")
        evaluate(model, test_data, args, before_train=True)

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
    print("Finetuning finished.")
    print("Train losses", train_losses)
    print("Test losses", test_losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./checkpoints/progen2-small",
        help="Path to the model checkpoint to be finetuned.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path relative to ~/progen/progen2. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path relative to ~/progen/progen2. Must contain preprocessed data (includes prefixes and one protein per line, e.g. not fasta format).",
    )
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Check out also the '--decay' argument. Default: 1e-4")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--checkpoint-rate", type=int, default=5, help="save checkpoint every n epochs"
    )
    parser.add_argument(
        "--decay", type=str, choices=["cosine", "linear", "constant"], default="cosine", help="Learning rate decay. Default: 'cosine'"
    )
    parser.add_argument("--save-optimizer", action="store_true", default=False, help="Should we also save the optimizer (and scheduler) at every checkpoint")
    parser.add_argument("--eval-before-train", action="store_true", default=False, help="Run evaluation on test set before training")
    args = parser.parse_args()

    main(args)

