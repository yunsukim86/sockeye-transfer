## Transfer Learning in Sockeye

This version of Sockeye contains codes to transfer a pre-trained model to another translation task. It includes the following additional components to Sockeye:

- Replacing embedding weights with pretrained embedding files ([fasttext](https://github.com/facebookresearch/fastText) format)
- Injecting artificial noises on training data (insertion, deletion, permutation)

If you use this code, please cite:

- Yunsu Kim, Yingbo Gao and Hermann Ney. [Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies](https://www-i6.informatik.rwth-aachen.de/publications/download/1102/Kim-ACL-2019.pdf). ACL 2019.
- Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton and Matt Post. [Sockeye: A Toolkit for Neural Machine Translation](https://arxiv.org/abs/1712.05690). arXiv preprint.


### Installation

```bash
> pip install -r requirements/requirements.txt
> pip install .
```
after cloning the repository from git.

If you want to run on a GPU you need to make sure your version of Apache MXNet
Incubating contains the GPU bindings. Depending on your version of CUDA you can do this by
running the following:

```bash
> pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install .
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).


### Usage

To extract embedding weights from a (pretrained) model file, use `tools/extract-embed.sh` script:
```bash
> ./extract-embed.sh {model_file} {vocabulary_file} (source|target)
```
The output embedding file is compatible with [MUSE](https://github.com/facebookresearch/MUSE) for a cross-lingual mapping.

To replace embedding weights in a (pretrained) model file, use `replace_embedding` module:
```bash
> python -m sockeye.replace_embedding -p {model_file} \
                                      -e {embedding_file} \
                                      -s (source|target) \
                                      -o {output_model_file} \
                                      -v {output_vocab_file}

```
The embedding file must be in [fasttext](https://github.com/facebookresearch/fastText) format. Unless `-o` and `-v` options are used, the output model/vocabulary files are generated with suffixes derived from the given embedding file. Please use the output model and vocabulary files in the child task training via `--params` and `--source-vocab` (or `--target-vocab`) options.

To pretrain a parent model with artificial noises, turn on `--source-noise-train` with detailed noise options (`--source-noise-insertion`, `--source-noise-insertion-vocab`, `--source-noise-deletion`, `--source-noise-permutation`). Optionally, you can also switch on `--source-noise-validation` to evaluate your models on a noisy validation set during the training. Example:
```bash
> python -m sockeye.train -s {training_data} \
                          -t {training_data} \
                          -vs {validation_data} \
                          -vt {validation_data} \
                          --source-noise-train \
                          --source-noise-permutation 3 \
                          --source-noise-deletion 0.1 \
                          --source-noise-insertion 0.1 \
                          --source-noise-insertion-vocab 50 \
                          .... (other options)
```
Injecting noises into the target side is analogous by replacing `source` with `target` in the option names.

Please refer to "Effective Cross-lingual Transfer of Neural Machine Translation Models without Shared Vocabularies" for further explanations of the transfer procedure.