# Author(s): Yunsu Kim, Jiahui Geng, Miguel Graca

from . import config
from . import constants as C
from . import utils

import copy
import mxnet as mx
import numpy as np


class NoiseModelConfig(config.Config):
    def __init__(self,
                 permutation: str,
                 deletion: float,
                 insertion: float,
                 insertion_vocab: int):
        super().__init__()
        self.permutation = permutation
        self.deletion = deletion
        self.insertion = insertion
        self.insertion_vocab = insertion_vocab

def get_noise_model(config: NoiseModelConfig):
    if config.permutation < 1 and config.deletion <= 0.0 and \
       (config.insertion <= 0.0 or config.insertion_vocab < 1):
        raise ValueError("Noise parameters are not effective (no noise will be applied)")
    else:
        return NoiseModel(config)

class NoiseModel:
    def __init__(self,
                 config: NoiseModelConfig):
        self.config = config

    def apply_noise(self, sentences):
        """
        Apply noise to a sentence as described in
        "Improving Unsupervised Word-by-Word Translation with Language Model and Denoising Autoencoder" (https://www-i6.informatik.rwth-aachen.de/publications/download/1075/Kim-EMNLP-2018.pdf).

        :param sentences:
         NDArray of batched sentences in shape (batch_size, seq_length, 1)
        :return:
         NDArray of batched noisy sentences in shape (batch_size, seq_length, 1)
        """
        assert(sentences.shape[2] == 1)

        max_sequence_length = sentences.shape[1]

        # Convert to np.ndarray since following operations are done on numpy
        # They are not done in mxnet since there is no support for ndarray.nonzero()
        sentences = sentences.asnumpy()
        for b in list(range(0, sentences.shape[0])):
            sentence = sentences[b, :, 0]
            sentence = np.take(sentence, sentence.nonzero())[0]  # Temporarily remove paddings
            sentence = self._deletion(sentence)
            sentence = self._permutation(sentence)
            sentence = self._insertion(sentence, max_sequence_length)

            # Write the new (possibly shortened) sentence
            new_length = sentence.shape[0]
            sentences[b, 0:new_length, 0] = sentence

            # Guarantee that tailing words are all padding tokens
            if new_length < max_sequence_length:
                sentences[b, new_length:, 0] = C.PAD_ID

        return mx.nd.array(sentences)

    def _permutation(self, sentence):
        length = sentence.shape[0]
        offset = np.random.randint(0, self.config.permutation + 1, size=length)
        indices = np.arange(length) + offset
        sentence = np.take(sentence, np.argsort(indices))
        return sentence

    def _deletion(self, sentence):
        length = sentence.shape[0]
        keep_value = np.random.rand(length) > self.config.deletion
        sentence = np.take(sentence, keep_value.nonzero())[0]
        return sentence

    def _insertion(self, sentence, max_sequence_length):
        length = sentence.shape[0]
        if length == max_sequence_length:
            return sentence

        insert_value = np.random.rand(length) <= self.config.insertion
        offset = 0
        for i in range(length):
            if insert_value[i]:
                sentence = np.insert(sentence,
                                     i + offset,
                                     np.random.randint(0, self.config.insertion_vocab) + len(C.VOCAB_SYMBOLS))
                offset += 1
                if length + offset >= max_sequence_length:
                    break
        return sentence
