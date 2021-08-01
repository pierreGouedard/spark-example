import os
from typing import List, Dict, Callable
import spacy
import re
from pathlib import Path
from pyspark import SparkConf, SparkContext
from operator import add


nlp = spacy.load('en_core_web_sm')
STOPWORDS = {sw: True for sw in nlp.Defaults.stop_words}
SPACY_LABEL_OUT = ['GPE', 'DATE', 'CARDINAL', 'MONEY', 'ORDINAL', 'TIME', 'PERCENT', 'QUANTITY', 'FAC', 'NORP', 'LOC']
date_regex = re.compile(r'\([aA-zZ]{3,6}.? [0-9]{4}\)')
general_regex = re.compile(r'[aA-zZ]+&[aA-zZ]|[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')


def clean_text(
    text: str, spacy_labels_out: List[str], spacy_nlp: Callable, d_stopwords: Dict[str, bool], date_regex: re.Pattern,
        general_regex: re.Pattern
) -> List[str]:
    """
    Clean a text string for NLP analysis.

    Parameters
    ----------
    text: str
        A text (string) to normalize.
    spacy_labels_out: list
        todo
    spacy_nlp: spacy nlp pipe
        todo
    d_stopwords: spacy nlp pipe
        todo

    Returns
    ----------
    cleaned_tokens : list
        A list of cleaned tokens.

    """
    # Filter detected name entities if enough context
    if len(text) > 70:
        text = remove_entities(text, spacy_labels_out, spacy_nlp)

    tokens = tokenize_text_pattern(text.lower(), date_regex, general_regex, d_stopwords)

    return tokens


def remove_entities(text: str, l_labels: List[str], nlp: Callable) -> str:
    """

    Parameters
    ----------
    text
    l_labels

    Returns
    -------

    """
    # Get entities word
    entities = set(list([x.text for x in nlp(text).ents if x.label_ in l_labels]))

    # Remove it from text
    for x in entities:
        text = text.replace(x, ' ')

    return text


def tokenize_text_pattern(text: str, date_regex, general_regex, d_stopwords) -> List[str]:
    """
    Tokenize text

    Remove campaigns date, seek for <token>x<token> pattern and <c>&<c> patterns using re.pattern technique.

    Parameters
    ----------
    text : str
        text that should be tokenized.

    Returns
    -------
    list
        list of token (str) built from input text.
    """
    # Remove date
    text = date_regex.sub('', text)

    # Get rest of words
    other_tokens = [x for x in general_regex.findall(text) if len(x) >= 2]

    # Remove stopwords
    l_tokens = [w for w in other_tokens if not d_stopwords.get(w, False)]

    return l_tokens


def wordcount():

    # Create conf and spark context
    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))

    )
    sc = SparkContext(conf=conf)

    d_stopwords_bc = sc.broadcast(STOPWORDS)
    spacy_labels_out_bc = sc.broadcast(SPACY_LABEL_OUT)
    date_regex_bc = sc.broadcast(date_regex)
    general_regex_bc = sc.broadcast(general_regex)
    nlp_bc = sc.broadcast(nlp)

    # load large text
    with open(os.path.join(os.getenv('DATA_DIR'), 'bible.txt'), 'r') as f:
        l_large_text = f.readlines()

    # wordcount using spark and functional logic
    lines = sc.parallelize(l_large_text)
    counts = lines.flatMap(
        lambda x: clean_text(
            x, spacy_labels_out_bc.value, nlp_bc.value, d_stopwords_bc.value, date_regex_bc.value, general_regex_bc.value
        )
    ) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(add)

    counts.persist()
    
    l_tops = counts.map(lambda x:  x[1]) \
        .sortBy(lambda x: x, False) \
        .take(10)

    max_bc = sc.broadcast(l_tops[-1])

    top_counts = counts.filter(lambda x: x[1] >= max_bc.value) \
        .collect()

    # Display result
    for (word, count) in top_counts:
        print("%s: %i" % (word, count))

    sc.stop()


if __name__ == "__main__":
    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {
        'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077",
        "APP_NAME": "word-count"
    }
    param_env = {'URL_DATA': "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    wordcount()
