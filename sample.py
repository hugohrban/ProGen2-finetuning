import os
import argparse
import logging

import torch
from tqdm import tqdm
import re

from tokenizers import Tokenizer, Encoding
from models.progen.modeling_progen import ProGenForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(
    model,
    tokenizer,
    device,
    prompt: str,
    max_length,
    num_return_sequences,
    temp=1.0,
    top_k=None,
):
    model.eval()
    encoding: Encoding = tokenizer.encode(prompt)
    ids = torch.tensor(encoding.ids)                                 # (T,)
    ids = ids[:len(torch.nonzero(ids))] 

    x = torch.zeros((num_return_sequences, ids.shape[0]))            # (B, T)
    x = x + ids
    x = x.to(device).to(torch.int32)

    past_key_values = None
    generated_seqs = x
    for _ in tqdm(range(ids.shape[0], max_length)):
        # using cached attn outputs from previous iterations
        output = model(x, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits                                       # (B, T, V)
        # get logits only for the last token
        logits = logits[:, -1, :]                                    # (B, V)
        logits = logits / temp
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)                 # (B, k)
            logits[logits < v[:, -1].unsqueeze(-1)] = -1e9           # (B, V)
        probs = torch.softmax(logits, dim=-1)                        # (B, V)
        x = torch.multinomial(probs, num_samples=1)                  # (B, 1)
        generated_seqs = torch.cat([generated_seqs, x], dim=-1)      # (B, T+1)

    decoded: list[str] = tokenizer.decode_batch(
        [row.detach().cpu().numpy().tolist() for row in generated_seqs]
    )
    return decoded


def truncate(seq: str) -> str:
    """
    Remove family special tokens, initial 1 or 2 token and truncate
    the sequence to the first 1 or 2 token found.
    
    Sequences begginning with 2 (C -> N generation) are reversed.
    """

    seq = re.sub(r"<\|pf\d+\|>", "", seq)
    terminus = seq[0]
    seq = seq[1:]

    min_1 = seq.find("1")
    if min_1 == -1:
        min_1 = len(seq)

    min_2 = seq.find("2")
    if min_2 == -1:
        min_2 = len(seq)

    seq = seq[: min(min_1, min_2)]
    if terminus == "1":
        return seq
    else:
        return seq[::-1]


def main(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    logger.info(f"Loading model from {args.model}")
    model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    logger.debug("Model loaded.")

    logger.info("Loading tokenizer")
    tokenizer = Tokenizer.from_pretrained(args.model)
    logger.debug("Tokenizer loaded.")
    logger.debug(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    logger.debug(f"Tokenizer vocab: {tokenizer.get_vocab()}")

    samples_dir = os.path.join("generated_samples", args.model.split("/")[-1])
    os.makedirs(samples_dir, exist_ok=True)
    output_file = os.path.join(
        samples_dir, f"samples_ctx{args.prompt}_k{args.k}_t{args.t}.fa"
    )
    logger.info(f"Generated samples will be saved to file {output_file}")

    if args.k == 0:
        args.k = None

    logger.debug(f"Sampling parameters: top_k={args.k}, temperature={args.t}")
    tokens = tokenizer.encode(args.prompt).tokens
    logger.info(f"Prompt tokens: {tokens[:tokens.index('<|pad|>')]}")

    with open(output_file, "w") as f:
        for i in range(args.iters):
            logger.info(f"Sampling batch {i+1} / {args.iters}")
            samples = sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.prompt,
                num_return_sequences=args.batch_size,
                temp=args.t,
                max_length=args.max_length,
                top_k=args.k,
            )
            for j, c in enumerate(samples):
                print(f">seq_{i * args.batch_size + j}", file=f)
                c = truncate(c)
                print(c, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="hugohrban/progen2-small-mix7",
        help="Hugging Face model name or path to the model directory. If path, should contain tokenizer.json, config.json and pytorch_model.bin. Default: hugohrban/progen2-small-mix7",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--prompt",
        type=str,
        default="1",
        help="Fixed initial part of sequence we continue the generation from.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="How many iterations of gerneration to run. Default: 1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="How many sequences to generate at one iteration. Default: 64",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum length of the generated sequence. Default: 1024",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Top-k sampling parameter. 0 means no top-k sampling. Default: 15",
    )
    parser.add_argument(
        "--t", type=float, default=1.0, help="Temperature for sampling. Default: 1.0"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging level.")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
    logger.info("Sampling finished.")
