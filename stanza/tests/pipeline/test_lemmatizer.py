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

DECODED = [("snarfiest", "ADJ"), ("wugs", "NOUN"), ("grobbled", "VERB"), ("QUAXES", "NOUN"),
           ("Wuggeries", "NOUN"), ("blicжets", "NOUN"), ("snarŋings", "NOUN")]
KEPT = [("Zorbles", "PROPN"), ("Snarfington", "PROPN"), ("Quorbish", "ADJ"), ("quibbagesz", "NOUN")]
LOWERED = [("BLORF", "NOUN"), ("ZORBLAX", "X")]
KNOWN = [("The", "DET"), ("dogs", "NOUN"), ("were", "AUX"), ("running", "VERB"),
         ("saw", "NOUN"), ("saw", "VERB"), ("left", "ADJ")]

NOCHARLM_LEMMAS = dict(zip(DECODED + KEPT + LOWERED + KNOWN,
                           ["snarfy", "wug", "grobble", "quax", "wuggery", "blicжet", "snarŋing",
                            "Zorbles", "Snarfington", "Quorbish", "quibbagesz", "blorf", "zorblax",
                            "the", "dog", "be", "run", "saw", "see", "left"]))
# Zorbles is left out because the charlm model gives it Zorble on CPU and Zorbles on GPU
CHARLM_LEMMAS = dict(zip(DECODED + KEPT[1:] + LOWERED + KNOWN,
                         ["snarfy", "wug", "grobble", "qual", "wuggery", "blicжet", "snarŋing",
                          "Snarfington", "quorbish", "quibbagesz", "blorf", "zorblax",
                          "the", "dog", "be", "run", "saw", "see", "left"]))

BATCH_SIZES = (1, 2, 3, 1000)

def charlm_options():
    models = os.path.join(TEST_MODELS_DIR, "en")
    return {"lemma_model_path": os.path.join(models, "lemma", "combined_charlm.pt"),
            "lemma_forward_charlm_path": os.path.join(models, "forward_charlm", "1billion.pt"),
            "lemma_backward_charlm_path": os.path.join(models, "backward_charlm", "1billion.pt"),
            "lemma_pretrain_path": os.path.join(models, "pretrain", "conll17.pt")}

def tagged_doc(sentences):
    return stanza.Document([[{"id": i + 1, "text": word, "upos": upos} for i, (word, upos) in enumerate(sentence)]
                            for sentence in sentences])

def pretagged_pipeline(use_gpu, **kwargs):
    return stanza.Pipeline('en', processors='tokenize,lemma', model_dir=TEST_MODELS_DIR, download_method=None,
                           tokenize_pretokenized=True, lemma_pretagged=True, use_gpu=use_gpu, **kwargs)

def lemmas_of(doc):
    return [(word.text, word.upos, word.lemma) for sentence in doc.sentences for word in sentence.words]

def mixed_sentences(words):
    """The words in sentences of different lengths and orders, with repeats"""
    decoded = [w for w in DECODED if w in words]
    others = [w for w in KEPT + LOWERED + KNOWN if w in words]
    return [decoded[:3] + others[:4], others[::-1] + decoded[2:] + decoded[:2], decoded[::-1],
            decoded[3:] * 2 + others[4:], others + decoded]

def assert_made_up(trainer, words):
    for word, _ in words:
        assert all(word not in entries for entries in trainer.pos_dict.values()), f"{word} is in the dictionary"

def assert_edit_classes(nlp):
    """The test model writes the lemmas of DECODED and returns KEPT and LOWERED as they are or lowercased"""
    from stanza.models.lemma.data import DataLoader
    from stanza.models.lemma.edit import EDIT_TO_ID
    lemmatizer = nlp.processors['lemma']
    words = DECODED + KEPT + LOWERED
    batch = DataLoader(tagged_doc([words]), 1000, lemmatizer.config, vocab=lemmatizer.vocab,
                       evaluation=True, expand_unk_vocab=True)
    _, edits = lemmatizer.trainer.predict(next(iter(batch)), 1, batch.vocab)
    expected = [EDIT_TO_ID[name] for name, group in (("none", DECODED), ("identity", KEPT), ("lower", LOWERED))
                for _ in group]
    assert edits == expected, "the model no longer sorts the made up words into the edit classes the tests need"

@pytest.mark.parametrize("use_gpu", DEVICES)
@pytest.mark.parametrize("model", ["nocharlm", "charlm"])
@pytest.mark.parametrize("options", [{}, {"lemma_ensemble_dict": False}, {"lemma_beam_size": 3}, {"lemma_dict_only": True}])
def test_lemmas_do_not_change_with_the_batch(use_gpu, model, options):
    """Every word keeps its lemma whatever the batch size and the words sharing its batch:
    words the seq2seq model writes, words it returns unchanged or lowercased, dictionary words."""
    expected = NOCHARLM_LEMMAS if model == "nocharlm" else CHARLM_LEMMAS
    if options.get("lemma_dict_only"):
        expected = {key: (lemma if key in KNOWN else key[0]) for key, lemma in expected.items()}
    extra = charlm_options() if model == "charlm" else {}
    nlp = pretagged_pipeline(use_gpu, lemma_store_results=False, **extra, **options)
    assert_made_up(nlp.processors['lemma'].trainer, [w for w in expected if w not in KNOWN])
    if model == "nocharlm" and not options:
        assert_edit_classes(nlp)
    sentences = mixed_sentences(expected)
    gold = [(word, upos, expected[(word, upos)]) for sentence in sentences for word, upos in sentence]
    for batch_size in BATCH_SIZES:
        nlp.processors['lemma'].config['batch_size'] = batch_size
        assert lemmas_of(nlp(tagged_doc(sentences))) == gold, f"lemma_batch_size={batch_size} changed a lemma"

@pytest.mark.parametrize("use_gpu", DEVICES)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_lemmas_do_not_change_with_the_batch_when_results_are_stored(use_gpu, batch_size):
    """With lemma_store_results on, as by default, each word keeps its lemma on the first
    document, which fills the cache, and on the second, which reads it."""
    nlp = pretagged_pipeline(use_gpu, lemma_batch_size=batch_size)
    sentences = mixed_sentences(NOCHARLM_LEMMAS)
    gold = [(word, upos, NOCHARLM_LEMMAS[(word, upos)]) for sentence in sentences for word, upos in sentence]
    assert lemmas_of(nlp(tagged_doc(sentences))) == gold
    assert lemmas_of(nlp(tagged_doc(sentences))) == gold

@pytest.mark.parametrize("use_gpu", DEVICES)
def test_documents_with_nothing_to_decode(use_gpu):
    """An empty document gets through, and a document whose words the seq2seq model
    only returns unchanged or lowercased gets their lemmas."""
    nlp = pretagged_pipeline(use_gpu, lemma_store_results=False, lemma_batch_size=1000)
    assert lemmas_of(nlp(stanza.Document([]))) == []
    sentence = KEPT + LOWERED
    assert lemmas_of(nlp(tagged_doc([sentence]))) == [(w, u, NOCHARLM_LEMMAS[(w, u)]) for w, u in sentence]

@pytest.mark.parametrize("use_gpu", DEVICES)
def test_raw_text_lemmas_do_not_change_with_the_batch(use_gpu):
    """Raw text through tokenize, mwt, pos and lemma, with made up words, gets the same
    lemmas at any lemma_batch_size."""
    text = "Joe's wugs grobbled the snarfiest Wuggeries. The BLORF's dogs were running to Snarfington."
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', model_dir=TEST_MODELS_DIR, download_method=None,
                          use_gpu=use_gpu, lemma_store_results=False, lemma_batch_size=1000)
    reference = lemmas_of(nlp(text))
    assert ("wugs", "NOUN", "wug") in reference, "the seq2seq model should write the lemma of wugs"
    for batch_size in BATCH_SIZES[:-1]:
        nlp.processors['lemma'].config['batch_size'] = batch_size
        assert lemmas_of(nlp(text)) == reference, f"lemma_batch_size={batch_size} changed a lemma"

def decoded_by_batch_size(nlp, words):
    """The characters the seq2seq model decodes for each word, before the lemmatizer
    falls back to the word for an output that runs to max_dec_len"""
    from stanza.models.lemma.data import DataLoader
    lemmatizer = nlp.processors['lemma']
    decoded = {}
    for batch_size in BATCH_SIZES:
        batches = DataLoader(tagged_doc([words]), batch_size, lemmatizer.config, vocab=lemmatizer.vocab,
                             evaluation=True, expand_unk_vocab=True)
        decoded[batch_size] = [characters for batch in batches
                               for characters in lemmatizer.trainer.predict(batch, 1, batches.vocab)[0]]
    return decoded

@pytest.fixture(scope="module")
def no_edit_no_copy_model(tmp_path_factory):
    from stanza.models import lemmatizer
    from stanza.tests.lemma.test_lemma_trainer import TRAIN_DATA, DEV_DATA
    tmp_path = tmp_path_factory.mktemp("no_edit_no_copy")
    train_file, dev_file = str(tmp_path / "train.conllu"), str(tmp_path / "dev.conllu")
    for filename, text in ((train_file, TRAIN_DATA), (dev_file, DEV_DATA)):
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write(text)
    lemmatizer.main(["--train_file", train_file, "--eval_file", dev_file, "--output_file", str(tmp_path / "pred.conllu"),
                     "--num_epoch", "2", "--save_dir", str(tmp_path), "--save_name", "lemma.pt",
                     "--shorthand", "en_test", "--no_edit", "--no_copy"])
    return str(tmp_path / "lemma.pt")

@pytest.mark.parametrize("use_gpu", DEVICES)
def test_model_without_edit_classifier_or_copy(use_gpu, no_edit_no_copy_model):
    """With a model trained with --no_edit --no_copy, each word keeps its lemma, and the
    characters the model decodes for it, whatever the batch size."""
    nlp = pretagged_pipeline(use_gpu, lemma_model_path=no_edit_no_copy_model, lemma_store_results=False)
    model = nlp.processors['lemma'].trainer.model
    assert not model.edit and not model.copy
    words = DECODED + KEPT + LOWERED
    assert_made_up(nlp.processors['lemma'].trainer, words)
    decoded = decoded_by_batch_size(nlp, words)
    assert all(characters for characters in decoded[1]), "the model should decode characters for every word"
    assert all(characters == decoded[1] for characters in decoded.values()), "decoded characters changed with the batch size"

    sentences = mixed_sentences(words)
    nlp.processors['lemma'].config['batch_size'] = 1
    reference = lemmas_of(nlp(tagged_doc(sentences)))
    for batch_size in BATCH_SIZES[1:]:
        nlp.processors['lemma'].config['batch_size'] = batch_size
        assert lemmas_of(nlp(tagged_doc(sentences))) == reference, f"lemma_batch_size={batch_size} changed a lemma"
