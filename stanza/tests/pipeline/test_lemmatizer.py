"""
Basic testing of lemmatization
"""

import copy
import os

import pytest
import torch
import stanza

from stanza.tests import *
from stanza.models.common.doc import TEXT, UPOS, LEMMA

pytestmark = pytest.mark.pipeline

EN_DOC = "Joe Smith was born in California."

EN_DOC_IDENTITY_GOLD = """
Joe Joe
Smith Smith
was was
born born
in in
California California
. .
""".strip()

EN_DOC_LEMMATIZER_MODEL_GOLD = """
Joe Joe
Smith Smith
was be
born bear
in in
California California
. .
""".strip()


def test_identity_lemmatizer():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en', 'lemma_use_identity': True}, download_method=None)
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_IDENTITY_GOLD == "\n".join(word_lemma_pairs)

def test_full_lemmatizer():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'}, download_method=None)
    doc = nlp(EN_DOC)
    word_lemma_pairs = []
    for w in doc.iter_words():
        word_lemma_pairs += [f"{w.text} {w.lemma}"]
    assert EN_DOC_LEMMATIZER_MODEL_GOLD == "\n".join(word_lemma_pairs)

def find_unknown_word(lemmatizer, base):
    for i in range(10):
        # pos_dict: pos -> word -> lemma
        # make sure that none of the pos slices contain this word
        base = base + "z"
        if all(base not in x for x in lemmatizer.pos_dict.values()):
            return base
    raise RuntimeError("wtf?")

def test_store_results():
    nlp = stanza.Pipeline(**{'processors': 'tokenize,pos,lemma', 'dir': TEST_MODELS_DIR, 'lang': 'en'}, lemma_store_results=True, download_method=None)
    lemmatizer = nlp.processors["lemma"]._trainer
    # the processor keeps a cache of words it has predicted, since those will not
    # change the second time it needs to predict that word
    remembered = nlp.processors["lemma"]._results

    az = find_unknown_word(lemmatizer, "a")
    bz = find_unknown_word(lemmatizer, "b")
    cz = find_unknown_word(lemmatizer, "c")

    # try sentences with the order long, short
    doc = nlp("I found an " + az + " in my " + bz + ".  It was a " + cz)
    stuff = doc.get([TEXT, UPOS, LEMMA])
    assert len(stuff) == 12
    assert stuff[3][0] == az
    assert stuff[6][0] == bz
    assert stuff[11][0] == cz

    assert remembered[(az, stuff[3][1])] == stuff[3][2]
    assert remembered[(bz, stuff[6][1])] == stuff[6][2]
    assert remembered[(cz, stuff[11][1])] == stuff[11][2]

    doc2 = nlp("I found an " + az + " in my " + bz + ".  It was a " + cz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    dz = find_unknown_word(lemmatizer, "d")
    ez = find_unknown_word(lemmatizer, "e")
    fz = find_unknown_word(lemmatizer, "f")

    # try sentences with the order short, long
    doc = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff = doc.get([TEXT, UPOS, LEMMA])
    assert len(stuff) == 12
    assert stuff[3][0] == dz
    assert stuff[8][0] == ez
    assert stuff[11][0] == fz

    assert remembered[(dz, stuff[3][1])] == stuff[3][2]
    assert remembered[(ez, stuff[8][1])] == stuff[8][2]
    assert remembered[(fz, stuff[11][1])] == stuff[11][2]

    doc2 = nlp("It was a " + dz + ".  I found an " + ez + " in my " + fz)
    stuff2 = doc2.get([TEXT, UPOS, LEMMA])

    assert stuff == stuff2

    # the shipped dictionary is never written to at all now
    assert all(az not in x for x in lemmatizer.pos_dict.values())
    assert all(dz not in x for x in lemmatizer.pos_dict.values())

def test_caseless_lemmatizer():
    """
    Test that setting the lemmatizer as caseless at Pipeline time lowercases the text
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    # the capital letter here should throw off the lemmatizer & it won't remove the plural
    # although weirdly the current English model *does* lowercase the A
    doc = nlp("Here is an Excerpt")
    assert doc.sentences[0].words[-1].lemma == 'excerpt'

    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None, lemma_caseless=True)
    # with the model set to lowercasing, the word will be treated as if it were 'antennae'
    doc = nlp("Here is an Excerpt")
    assert doc.sentences[0].words[-1].lemma == 'Excerpt'

def test_latin_caseless_lemmatizer():
    """
    Test the Latin caseless lemmatizer
    """
    nlp = stanza.Pipeline('la', package='ittb', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None)
    lemmatizer = nlp.processors['lemma']
    assert lemmatizer.config['caseless']

    doc = nlp("Quod Erat Demonstrandum")
    expected_lemmas = "qui sum demonstro".split()
    assert len(doc.sentences) == 1
    assert len(doc.sentences[0].words) == 3
    for word, expected in zip(doc.sentences[0].words, expected_lemmas):
        assert word.lemma == expected

def test_contextual_lemmatizer():
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR, package={"lemma": "default_accurate"}, download_method=None)
    lemmatizer = nlp.processors['lemma']._trainer
    # the accurate model should have a 's classifier
    assert len(lemmatizer.contextual_lemmatizers) > 0
    doc = nlp("He's added a contextual lemmatizer")
    assert len(doc.sentences) == 1
    assert doc.sentences[0].words[1].text == "'s"
    assert doc.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert doc.sentences[0].words[1].lemma == "have"

    doc = nlp("He's a little tired")
    assert len(doc.sentences) == 1
    assert doc.sentences[0].words[1].text == "'s"
    assert doc.sentences[0].words[1].pos == "AUX"
    # this test should be simple enough that the
    # contextual classifier gets it right,
    # unless it gets retrained really badly
    assert doc.sentences[0].words[1].lemma == "be"


def test_contextual_lemmatizer_results_are_not_cached():
    """lemma_store_results must not remember a word a contextual lemmatizer decides.

    The seq2seq prediction is made before update_contextual_preds runs, so one
    word, pos answer for 's would stand in for every occurrence: have in "He's
    added a lemmatizer" and be in "He's a little tired". The output itself is
    saved by the contextual pass running on every call, so this asserts on the
    dict rather than on the lemmas.
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          package={"lemma": "default_accurate"}, lemma_store_results=True,
                          download_method=None)
    trainer = nlp.processors['lemma']._trainer
    assert len(trainer.contextual_lemmatizers) > 0

    assert trainer.is_contextual_target("'s", "AUX")
    assert trainer.is_contextual_target("'S", "AUX")      # matched lowercased
    assert not trainer.is_contextual_target("'s", "PART")  # a different tag is not a target
    assert not trainer.is_contextual_target("dog", "NOUN")
    assert trainer.drop_contextual([("'s", "AUX", "be"), ("dog", "NOUN", "dog")]) == [("dog", "NOUN", "dog")]

    # the shipped dict already knows 's, so skip_seq2seq skips it before the cache
    # can see it; dropping those entries exercises the path the filter guards
    for pos_dict in trainer.pos_dict.values():
        pos_dict.pop("'s", None)
    assert trainer.skip_seq2seq([("'s", "AUX")]) == [False]

    doc = nlp("He's added a contextual lemmatizer")
    assert doc.sentences[0].words[1].lemma == "have"
    assert "'s" not in trainer.pos_dict.get("AUX", {})


def test_contextual_lemma_survives_the_results_cache():
    """With lemma_store_results on, the cache must never answer for a word the contextual
    lemmatizer decides, however many times that word has been through the lemmatizer.

    "her" is PRON in both senses, so the upos cannot separate them and only the contextual
    lemmatizer can: she when it is the object, her when it is the possessive. Alternating
    the two senses through one pipeline pins that behaviour.
    """
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          package={"lemma": "default_accurate"}, lemma_store_results=True,
                          download_method=None)
    expected = [
        ("I saw her yesterday.", "she"),
        ("Her book is on the table.", "her"),
        ("They thanked her warmly.", "she"),
        ("Her car broke down.", "her"),
        ("We invited her to dinner.", "she"),
        ("Her sister called.", "her"),
    ]
    for text, lemma in expected:
        words = [word for word in nlp(text).sentences[0].words if word.text.lower() == "her"]
        assert len(words) == 1, text
        assert words[0].pos == "PRON", f"{text}: upos {words[0].pos}, expected PRON"
        assert words[0].lemma == lemma, f"{text}: lemma {words[0].lemma!r}, expected {lemma!r}"


def test_store_results_is_on_by_default_and_bounded():
    """The cache has an upper bound and is on by default. Words the dictionary already
    knows never reach it, so this uses words it does not know."""
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          download_method=None)
    assert nlp.processors['lemma'].store_results, "store_results should be on by default"

    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                          lemma_store_results=True, lemma_store_results_size=3,
                          download_method=None)
    lemmatizer = nlp.processors['lemma']
    assert lemmatizer.store_results_size == 3

    words = [find_unknown_word(lemmatizer.trainer, base) for base in ["flurbing", "zonkled", "quibbage", "snarfled", "brindling"]]
    for word in words:
        nlp(f"They were {word} loudly.")
        assert len(lemmatizer._results) <= 3, f"cache grew past its cap at {word}"
    # the oldest are gone, the newest are kept
    remembered = [word for word, _ in lemmatizer._results]
    assert words[0] not in remembered and words[1] not in remembered
    assert words[-1] in remembered


def test_store_results_does_not_change_any_lemma():
    """Remembering an answer must not change it, on the first pass or a later one, and the
    model's own dictionary answers before the cache does."""
    text = ("The dogs were running and the children laughed. "
            "She was running while the dogs barked at the children.")

    def lemmas(**kwargs):
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', model_dir=TEST_MODELS_DIR,
                              download_method=None, **kwargs)
        shipped = copy.deepcopy(nlp.processors['lemma'].trainer.pos_dict)
        first = [word.lemma for sent in nlp(text).sentences for word in sent.words]
        second = [word.lemma for sent in nlp(text).sentences for word in sent.words]
        return first, second, nlp, shipped

    cold_first, cold_second, _, _ = lemmas(lemma_store_results=False)
    warm_first, warm_second, nlp, shipped = lemmas(lemma_store_results=True)
    assert cold_first == cold_second
    assert warm_first == warm_second
    assert warm_first == cold_first, "caching changed a lemma on the first pass"
    assert warm_second == cold_second, "caching changed a lemma once the cache was warm"

    # the cache lives beside the shipped dictionary, never inside it.
    # the words above are all in the dictionary, so made up words are
    # needed to put something in the cache
    lemmatizer = nlp.processors['lemma']
    verb = find_unknown_word(lemmatizer.trainer, "flurbing")
    noun = find_unknown_word(lemmatizer.trainer, "quibbages")
    nlp(f"They were {verb} the {noun}.")
    assert len(lemmatizer._results) > 0, "the made up words should have gone through the cache"
    assert lemmatizer.trainer.pos_dict == shipped, "the cache changed the shipped dictionary"

    lemmatizer._results[("dogs", "NOUN")] = "WRONG"
    doc = nlp("The dogs barked.")
    dogs = [word for word in doc.sentences[0].words if word.text == "dogs"]
    assert dogs[0].lemma == "dog", "the model's dictionary should answer before the cache"



DEVICES = [False] + ([True] if torch.cuda.is_available() else [])

# made up words, grouped by what the edit classifier of the English test model
# does with them: the decoder writes the lemma of the first group (two of them
# copy a character the model has never seen), the second group is returned
# unchanged and the third is lowercased
DECODED = [("snarfiest", "ADJ"), ("wugs", "NOUN"), ("grobbled", "VERB"), ("QUAXES", "NOUN"),
           ("Wuggeries", "NOUN"), ("blicжets", "NOUN"), ("snarŋings", "NOUN")]
KEPT = [("Zorbles", "PROPN"), ("Snarfington", "PROPN"), ("Quorbish", "ADJ"), ("quibbagesz", "NOUN")]
LOWERED = [("BLORF", "NOUN"), ("ZORBLAX", "X")]
KNOWN = [("The", "DET"), ("dogs", "NOUN"), ("were", "AUX"), ("running", "VERB")]

# the lemmas Stanza gave these words before decoding was sped up, on CPU and GPU
NOCHARLM_LEMMAS = dict(zip(DECODED + KEPT + LOWERED + KNOWN,
                           ["snarfy", "wug", "grobble", "quax", "wuggery", "blicжet", "snarŋing",
                            "Zorbles", "Snarfington", "Quorbish", "quibbagesz", "blorf", "zorblax",
                            "the", "dog", "be", "run"]))
# Zorbles is left out: the charlm model gave it Zorble on CPU and Zorbles on GPU
# before the speedup too
CHARLM_LEMMAS = dict(zip(DECODED + KEPT[1:] + LOWERED + KNOWN,
                         ["snarfy", "wug", "grobble", "qual", "wuggery", "blicжet", "snarŋing",
                          "Snarfington", "quorbish", "quibbagesz", "blorf", "zorblax",
                          "the", "dog", "be", "run"]))

def charlm_options():
    models = os.path.join(TEST_MODELS_DIR, "en")
    return {"lemma_model_path": os.path.join(models, "lemma", "combined_charlm.pt"),
            "lemma_forward_charlm_path": os.path.join(models, "forward_charlm", "1billion.pt"),
            "lemma_backward_charlm_path": os.path.join(models, "backward_charlm", "1billion.pt"),
            "lemma_pretrain_path": os.path.join(models, "pretrain", "conll17.pt")}

def tagged_doc(sentences):
    return stanza.Document([[{"id": i + 1, "text": word, "upos": upos} for i, (word, upos) in enumerate(sentence)]
                            for sentence in sentences])

def pretagged_pipeline(use_gpu, batch_size=50, **kwargs):
    return stanza.Pipeline('en', processors='tokenize,lemma', model_dir=TEST_MODELS_DIR, download_method=None,
                           tokenize_pretokenized=True, lemma_pretagged=True, lemma_batch_size=batch_size,
                           use_gpu=use_gpu, **kwargs)

def lemmas_of(doc):
    return [(word.text, word.upos, word.lemma) for sentence in doc.sentences for word in sentence.words]

def assert_made_up(trainer, words):
    for word, _ in words:
        assert all(word not in entries for entries in trainer.pos_dict.values()), f"{word} is in the dictionary"

def mixed_sentences(words):
    """The words in sentences of different lengths, in different orders, with repeats"""
    decoded = [w for w in DECODED if w in words]
    others = [w for w in KEPT + LOWERED + KNOWN if w in words]
    return [decoded[:3] + others[:4], others[::-1] + decoded[2:] + decoded[:2], decoded[::-1],
            decoded[3:] * 2 + others[4:], others + decoded]

@pytest.mark.parametrize("use_gpu", DEVICES)
@pytest.mark.parametrize("model", ["nocharlm", "charlm"])
@pytest.mark.parametrize("options", [{"lemma_store_results": False}, {"lemma_store_results": True},
                                     {"lemma_ensemble_dict": False}, {"lemma_beam_size": 3}, {"lemma_dict_only": True}])
def test_lemmas_do_not_change_with_the_batch(use_gpu, model, options):
    """Every word gets the lemma Stanza gave it before decoding was sped up, whatever
    the batch size and whatever words share its batch: words the decoder writes,
    words returned unchanged or lowercased, dictionary words, and repeats."""
    expected = NOCHARLM_LEMMAS if model == "nocharlm" else CHARLM_LEMMAS
    extra = charlm_options() if model == "charlm" else {}
    if options.get("lemma_dict_only"):
        # without the seq2seq model, a word the dictionary does not know is its own lemma
        expected = {key: (lemma if key in KNOWN else key[0]) for key, lemma in expected.items()}
    sentences = mixed_sentences(expected)
    for batch_size in (1, 2, 3, 7, 1000):
        nlp = pretagged_pipeline(use_gpu, batch_size=batch_size, **extra, **options)
        assert_made_up(nlp.processors['lemma'].trainer, [w for w in expected if w not in KNOWN])
        got = lemmas_of(nlp(tagged_doc(sentences)))
        assert got == [(word, upos, expected[(word, upos)]) for sentence in sentences for word, upos in sentence], \
            f"lemma_batch_size={batch_size} changed a lemma"

@pytest.mark.parametrize("use_gpu", DEVICES)
def test_documents_with_nothing_to_decode(use_gpu):
    """An empty document, and one whose words are all in the dictionary, get through."""
    nlp = pretagged_pipeline(use_gpu, lemma_store_results=False)
    assert lemmas_of(nlp(stanza.Document([]))) == []
    assert lemmas_of(nlp(tagged_doc([KNOWN]))) == [(word, upos, NOCHARLM_LEMMAS[(word, upos)]) for word, upos in KNOWN]

def count_decoded_rows(nlp):
    """Records the number of words in each decoding step of the seq2seq model."""
    model = nlp.processors['lemma'].trainer.model
    rows = []
    decode = model.decode
    def counting(dec_inputs, *args, **kwargs):
        rows.append(dec_inputs.size(0))
        return decode(dec_inputs, *args, **kwargs)
    model.decode = counting
    return rows

@pytest.mark.parametrize("use_gpu", DEVICES)
def test_decoder_skips_the_words_it_does_not_write(use_gpu):
    """Words the edit classifier returns unchanged or lowercased get their lemma
    without being decoded; the decoder only reads the words whose lemma it writes."""
    nlp = pretagged_pipeline(use_gpu, lemma_store_results=False)
    assert_made_up(nlp.processors['lemma'].trainer, DECODED + KEPT + LOWERED)
    rows = count_decoded_rows(nlp)
    sentence = KEPT + LOWERED
    assert lemmas_of(nlp(tagged_doc([sentence]))) == [(w, u, NOCHARLM_LEMMAS[(w, u)]) for w, u in sentence]
    assert rows == [], "words returned unchanged or lowercased were decoded"

    sentence = KEPT + DECODED + LOWERED
    assert lemmas_of(nlp(tagged_doc([sentence]))) == [(w, u, NOCHARLM_LEMMAS[(w, u)]) for w, u in sentence]
    assert rows, "the made up words should reach the decoder"
    assert max(rows) == len(DECODED), "the decoder should read exactly the words whose lemma it writes"

def count_gpu_waits(nlp, doc):
    """Runs nlp on doc and counts the times the host waits for the GPU."""
    import warnings
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nlp(doc)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(warning.message) for warning in caught)

def made_up_nouns(ending, n=32):
    """n made up nouns that end in ending, so that all of them have one length"""
    import itertools
    return [("".join(x) + ending, "NOUN") for x in itertools.product("bdgkmpst", repeat=2)][:n]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_waits_do_not_grow_with_the_words_decoded():
    """On a GPU, decoding more words of the same length in one batch adds no waits
    for the device: the decoded characters come to the host once per step."""
    nlp = pretagged_pipeline(True, batch_size=1000, lemma_store_results=False)
    words = made_up_nouns("orfings")
    assert_made_up(nlp.processors['lemma'].trainer, words)
    few = count_gpu_waits(nlp, tagged_doc([words[:8]]))
    many = count_gpu_waits(nlp, tagged_doc([words]))
    assert many - few < 32 - 8, f"{few} waits for 8 words, {many} for 32: at least one more per word"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_waits_once_per_decoding_step():
    """On a GPU, each decoding step waits for the device once, to read the decoded
    characters: longer lemmas take more steps, and no more than one wait each."""
    nlp = pretagged_pipeline(True, batch_size=1000, lemma_store_results=False)
    short, long = made_up_nouns("orfings", 8), made_up_nouns("orfiplonkerbatings", 8)
    assert_made_up(nlp.processors['lemma'].trainer, short + long)
    steps = count_decoded_rows(nlp)
    short_waits = count_gpu_waits(nlp, tagged_doc([short]))
    short_steps = len(steps)
    long_waits = count_gpu_waits(nlp, tagged_doc([long]))
    long_steps = len(steps) - short_steps
    assert long_steps > short_steps, "the longer words should take more decoding steps"
    assert long_waits - short_waits <= long_steps - short_steps, \
        f"{long_steps - short_steps} more steps took {long_waits - short_waits} more waits"
