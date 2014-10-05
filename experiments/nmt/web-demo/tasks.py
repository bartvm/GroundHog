import cPickle
import HTMLParser
import os
from subprocess import Popen, PIPE

from celery import Celery, signals
from celery.bin import Option
import numpy

from experiments.nmt import (parse_input, prototype_phrase_state,
                             RNNEncoderDecoder)
from experiments.nmt.sample import BeamSearch, sample


app = Celery('tasks')
app.config_from_object('celeryconfig')
app.user_options['preload'].add(
    Option("--state", type='string', help="File with state to use")
)
app.user_options['preload'].add(
    Option("--disable-tokenization", action="store_true", default=False,
           help="Disable the tokenization of words")
)
app.user_options['preload'].add(
    Option("--model", type='string', default='',
           help="File with trained model")
)

sampler = None


class Sampler(object):
    def __init__(self, state, lm_model, indx_word, idict_src, beam_search,
                 alignment_func, tokenizer_cmd=None, detokenizer_cmd=None):
        self.__dict__.update(locals())
        h = HTMLParser.HTMLParser()
        del self.self

    def sample(self, sentence, beam_width, ignore_unk):
        print "Input sentence: {}".format(sentence)
        if self.tokenizer_cmd:
            tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
            sentence, _ = tokenizer.communicate(sentence)
            print "Detokenized sentence: {}".format(sentence)

        seq_in, _ = parse_input(self.state, self.indx_word,
                                sentence, idx2word=self.idict_src)
        print "Parsed to: {}".format(seq_in)

        print "Performing beam search..."
        trans, cost, seq_out = sample(self.lm_model, seq_in, beam_width,
                                     beam_search=self.beam_search,
                                     normalize=True, ignore_unk=ignore_unk)

        print "Detokenizing"
        if self.detokenizer_cmd:
            detokenizer = Popen(self.detokenizer_cmd, stdin=PIPE, stdout=PIPE)
            detokenized_sentence, _ = detokenizer.communicate(trans[0])
        else:
            detokenized_sentence = trans[0]

        unknown_words = [word for word, index
                         in zip(sentence.split(), seq_in)
                         if index == 1]

        print "Calculating alignment"
        _, alignment = self.alignment_func(numpy.array(seq_in),
                                           numpy.array(seq_out[0]))

        # Return detokenized output, tokenized input, tokenized output,
        # unknown input words, alignment scores
        return (detokenized_sentence, unknown_words, sentence, trans[0],
                alignment.reshape(alignment.shape[:-1]).tolist())


@app.task
def translate(sentence, beam_width=5, ignore_unk=False):
    return sampler.sample(sentence, beam_width, ignore_unk)


@signals.user_preload_options.connect
def handle_preload_options(options, **kwargs):
    # Only load the model for workers that were told to on the command line
    if options['model']:
        load_model(options)


def load_model(options):
    # We store the sampler in a global variable which all tasks can access
    global sampler

    # Configure the model's state
    state = prototype_phrase_state()
    if options['state']:
        with open(options['state'], 'rb') as f:
            state.update(cPickle.load(f))

    # Load the vocabulary
    with open(state['indx_word'], 'rb') as f:
        idict_src = cPickle.load(f)
    with open(state['word_indx'], 'rb') as f:
        indx_word = cPickle.load(f)

    # Build the encoder-decoder and load parameters
    rng = numpy.random.RandomState(state['seed'])
    enc_dec = RNNEncoderDecoder(state, rng, skip_init=True,
                                compute_alignment=True)
    enc_dec.build()

    lm_model = enc_dec.create_lm_model()
    lm_model.load(options['model'])

    # Load the beam search
    beam_search = BeamSearch(enc_dec)
    beam_search.compile()

    # Set up tokenization
    if options['disable_tokenization']:
        tokenizer_cmd, detokenizer_cmd = None, None
    else:
        tokenizer_cmd = [os.getcwd() + '/tokenizer.perl',
                         '-l', 'en', '-q', '-']
        detokenizer_cmd = [os.getcwd() + '/detokenizer.perl',
                           '-l', 'fr', '-q', '-']

    # Create function to calculate alignment
    alignment_func = enc_dec.create_probs_computer(True)

    # Load a sampler instance which performs the sampling procedure
    sampler = Sampler(state, lm_model, indx_word, idict_src,
                      beam_search, alignment_func,
                      tokenizer_cmd=tokenizer_cmd,
                      detokenizer_cmd=detokenizer_cmd)
    return sampler
