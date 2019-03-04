import argparse
import os
from typing import Dict, List, Tuple

import mxnet as mx
import numpy as np

from .log import setup_main_logger, log_sockeye_version
from . import arguments
from . import utils
from .vocab import vocab_to_json

logger = setup_main_logger(__name__, console=True, file_logging=False)


def load_vec(vec_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load vocabulary and weight matrix from .vec file.
    """
    with open(vec_path, 'r') as vec:
        embed_dim = int(vec.readline().split()[1])
    vocab = np.loadtxt(vec_path, dtype=str, comments=None, skiprows=1, usecols=0).tolist()
    weight = np.loadtxt(vec_path, comments=None, skiprows=1, usecols=range(1, embed_dim + 1))
    return vocab, weight


def convert_vocab(vocab: List[str]) -> Dict[int, str]:
    """
    Convert a vocabulary list to Sockeye-compatible dictionary.
    Add special tokens for Sockeye and give id's to each word.
    """
    new_list = ['<pad>', '<unk>', '<s>'] + vocab
    new_dict = {word: id for id, word in enumerate(new_list)}
    return new_dict


def convert_weight(weight: np.ndarray,
                   org_params: Dict[str, mx.nd.NDArray],
                   side: str) -> mx.nd.NDArray:
    """
    Convert a numpy embedding matrix to MXNet NDArray.
    Add embeddings for Sockeye special tokens from org_params.
    """
    if side == 'source':
        org_np_ndarray = org_params['source_embed_weight'].asnumpy()
    elif side == 'target':
        org_np_ndarray = org_params['target_embed_weight'].asnumpy()
    new_np_ndarray = np.concatenate((org_np_ndarray[:3], weight))
    new_mx_ndarray = mx.nd.array(new_np_ndarray)
    return new_mx_ndarray


def replace_embeddings(args: argparse.Namespace):
    log_sockeye_version(logger)

    embed_basename = os.path.basename(args.embed_file)
    if not args.output_params:
        if args.side == 'source':
            output_params = args.params + ".source_embed_weight-" + embed_basename
        elif args.side == 'target':
            output_params = args.params + ".target_embed_weight-" + embed_basename
    else:
        output_params = args.output_params
    if not args.vocab_file:
        if args.side == 'source':
            vocab_file = "vocab.src.0." + embed_basename + ".json"
        elif args.side == 'target':
            vocab_file = "vocab.trg.0." + embed_basename + ".json"
    else:
        vocab_file = args.vocab_file

    vocab, weight = load_vec(args.embed_file)

    new_vocab = convert_vocab(vocab)
    vocab_to_json(new_vocab, vocab_file)

    arg_params, aux_params = utils.load_params(args.params)
    new_weight = convert_weight(weight, arg_params, args.side)
    if args.side == 'source':
        arg_params['source_embed_weight'] = new_weight
    elif args.side == 'target':
        arg_params['target_embed_weight'] = new_weight
        arg_params['target_output_bias'] = mx.ndarray.zeros((len(vocab) + 3,))
    utils.save_params(arg_params, output_params, aux_params)


def main():
    """
    Commandline interface to replace embedding weights with pretrained word representations.
    """
    params = argparse.ArgumentParser(description='Replace embeddings with given .vec file.')
    arguments.add_replace_embedding_args(params)
    args = params.parse_args()
    replace_embeddings(args)


if __name__ == '__main__':
    main()
