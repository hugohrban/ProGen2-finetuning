import os
import argparse
import logging

import torch
from tqdm import tqdm
import re
from typing import Optional, Union

from tokenizers import Tokenizer, Encoding
from models.progen.modeling_progen import ProGenForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def sample(
    model: ProGenForCausalLM,
    tokenizer: Tokenizer,
    device: torch.device,
    prompt: Union[str, torch.Tensor],
    max_length: int,
    num_return_sequences: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> list[str]:
    """
    Generate samples from the model given a prompt. Using top-k sampling with temperature.
    """
    model.eval()

    if isinstance(prompt, str):
        encoding: Encoding = tokenizer.encode(prompt)
        ids = torch.tensor(encoding.ids)                             # (T,)
        ids = ids[:len(torch.nonzero(ids))]

        x = torch.zeros((num_return_sequences, ids.shape[0]))        # (B, T)
        x = x + ids
        x = x.to(device).to(torch.int32)
    # prompt is a tensor of token ids, with shape (B, T), in case of bidi smapling
    elif isinstance(prompt, torch.Tensor):
        x = prompt.to(device).to(torch.int32)                        # (B, T)
    else:
        raise ValueError("Prompt should be either string or torch.Tensor")

    past_key_values = None
    generated = x

    pbar = tqdm(total=max_length - generated.shape[-1])
    while generated.shape[-1] < max_length:
        # using cached attn outputs from previous iterations
        output = model(x, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits                                       # (B, T, V)
        # get logits only for the last token
        logits = logits[:, -1, :]                                    # (B, V)
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)                 # (B, k)
            logits = torch.where(logits >= v[:, -1:], logits, -1e9)  # (B, V)
        probs = torch.softmax(logits, dim=-1)                        # (B, V)
        x = torch.multinomial(probs, num_samples=1)                  # (B, 1)
        generated = torch.cat([generated, x], dim=-1)                # (B, T+1)
        pbar.update()
    pbar.close()

    decoded = [tokenizer.decode(row.detach().cpu().numpy().tolist()) for row in generated]
    return decoded


def truncate(seq: str) -> str:
    """
    Remove family special tokens, initial 1 or 2 token and truncate
    the sequence to the first 1 or 2 token found.
    
    Sequences begginning with 2 (C -> N generation) are reversed.
    """

    # remove family token
    seq = re.sub(r"<\|.*\|>", "", seq)

    # remove initial terminus
    terminus = seq[0]
    seq = seq[1:]

    min_1 = seq.find("1")
    if min_1 == -1:
        min_1 = len(seq)

    min_2 = seq.find("2")
    if min_2 == -1:
        min_2 = len(seq)

    # truncate the sequence to next terminus token
    seq = seq[: min(min_1, min_2)]
    if terminus == "1":
        return seq
    else:
        return seq[::-1]


def reverse(seq: str) -> str:
    """
    Reverse a sequence that starts with a family token and initial terminus.
    Then continue generating the sequence in opposite direction.
    """
    prefix_pattern = re.compile(r"<\|.*\|>")
    m = re.search(prefix_pattern, seq)
    prefix = m.group() if m else ""

    # remove family token
    seq = seq.replace(prefix, "")

    # remove initial terminus and reverse the sequence
    start_terminus = seq[0]
    seq = seq[1:]
    seq = seq[::-1]

    # if we generated end of sequence
    if seq[0] in ["1", "2"]:
        seq = seq[1:]

    # reverse and put opposite terminus
    if start_terminus == "1":
        return prefix + "2" + seq
    else:
        return prefix + "1" + seq


def main(args):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    if str(device) == "cpu" and args.batch_size > 1:
        logger.warning(f"You are using CPU for inference with a relatively high batch size of {args.batch_size}, therefore inference might be slow. Consider using a GPU or smaller batch.")

    logger.info(f"Loading model from {args.model}")
    model = ProGenForCausalLM.from_pretrained(args.model).to(device)
    logger.debug("Model loaded.")

    logger.info("Loading tokenizer")
    if os.path.exists(os.path.join(args.model, "tokenizer.json")):
        tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    else:
        tokenizer: Tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.no_padding()
    logger.debug("Tokenizer loaded.")
    logger.debug(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    logger.debug(f"Tokenizer vocab: {tokenizer.get_vocab()}")

    samples_dir = os.path.join("generated_samples", args.model.split("/")[-1])
    os.makedirs(samples_dir, exist_ok=True)
    output_file = os.path.join(samples_dir, f"samples_ctx{args.prompt}_k{args.k}_t{args.t}.fa")

    if args.k == 0 or args.k > model.config.vocab_size_lm_head:
        args.k = None

    logger.debug(f"Sampling parameters: top_k={args.k}, temperature={args.t}")
    tokens = tokenizer.encode(args.prompt).tokens
    logger.info(f"Prompt tokens: {tokens}")

    if args.bidirectional:
        args.max_length = (args.max_length - len(tokens)) // 2
        if len(tokens) <= 2:
            logger.warning("Prompt is too short for bidirectional sampling. Please provide a longer prompt.")

    with open(output_file, "w") as f:
        for i in range(args.iters):
            logger.info(f"Sampling batch {i+1} / {args.iters}")
            samples = sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.prompt,
                num_return_sequences=args.batch_size,
                temperature=args.t,
                max_length=args.max_length,
                top_k=args.k,
            )
            if args.bidirectional:
                reversed_samples = [reverse(s) for s in samples]
                samples = []
                for rs in reversed_samples:
                    prompt = torch.tensor(tokenizer.encode(rs).ids).view(1, -1).to(model.device)
                    samples.extend(sample(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        num_return_sequences=1,
                        temperature=args.t,
                        max_length=args.max_length * 2,
                        top_k=args.k,
                    ))
            for j, c in enumerate(samples):
                print(f">seq_{i * args.batch_size + j}", file=f)
                c = truncate(c)
                print(c, file=f)
    logger.info(f"Generated samples were saved to file {output_file}")


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
        "--batch_size",
        type=int,
        default=64,
        help="How many sequences to generate at one iteration. Default: 64",
    )
    parser.add_argument(
        "--max_length",
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
    parser.add_argument(
        "--bidirectional",
        action="store_true", 
        help="Enable bidirectional sampling. After generating half of the sequence, it is flipped and model generates the other half in opposite direction."
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    main(args)
    logger.info("Sampling finished.")
