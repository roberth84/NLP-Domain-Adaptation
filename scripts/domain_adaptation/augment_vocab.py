#!/usr/bin/env python3

"""BERT vocabulary update functionality."""
import logging
import argparse
from pathlib import Path
from typing import List, Union, Optional

import os
import shutil
import glob
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from boto3 import client, resource

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

logger = logging.getLogger(__name__)
s3, s3r = None, None

VOCAB_CACHE_PREFIX = 'temp-in-domain'

def parse_args():
    parser = argparse.ArgumentParser(
        "Augment BERT's vocabulary with relevant in-domain tokens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tokenizer_vocab', type=str, nargs='+',
                        help='Path to original BERT vocabulary text file.')
    parser.add_argument('--corpus', required=True,
                        type=lambda paths: paths.split(','),
                        help='Path to in-domain corpus text file.')
    parser.add_argument('--dst', type=str, required=True,
                        help='Directory to output the augmented '
                             'vocabulary text file.')
    parser.add_argument('--no-lowercase',
                        action='store_false', dest='lowercase',
                        help='If provided, will not perform lowercasing of '
                             'corpus.')
    parser.add_argument('--vocab-size', type=int, default=30519,
                        help='Vocabulary size of newly trained '
                             'WordPieceTokenizer')
    parser.add_argument('--rank-by', choices=('count', 'tfidf'),
                        default='count', help='Ranking heuristic')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='If provided, train tokenizer vocabulary from '
                             'scratch.')
    parser.add_argument('--tokenizer_type', type=str, default='bert',
                        help='type of tokenizer ("bert" or "roberta")')
    parser.add_argument('--s3_mem_limit', type=int, default=30000,
                        help="Max memory to use for documents downloaded from s3.")

    args, _ = parser.parse_known_args()

    return args


def train_tokenizer(corpus: Union[str, List[str]],
                    vocab_size: int = 30519,
                    overwrite: bool = True,
                    lowercase: bool = True,
                    save_vocab: bool = False,
                    dst: Optional[str] = None,
                    in_domain_vocab: str = VOCAB_CACHE_PREFIX,
                    tokenizer_type: str = 'bert',
                    tokenizer_kwargs: dict = None
                   ) -> Union[BertWordPieceTokenizer, ByteLevelBPETokenizer]:
    """Train a WordPiece tokenizer from scratch.

    Arguments:
        corpus {Union[str, List[str]]} -- In-domain corpus / corpora

    Keyword Arguments:
        vocab_size {int} -- Size of trained vocabulary (default: 30519)
        lowercase {bool} -- If True, perform lowercasing (default: True)
        save_vocab {bool} -- If True, save vocab to `in_domain_vocab`
                             (default: Fakse)
        in_domain_vocab {str} -- Path to save trained tokenizer vocabulary
                                 (default: {'in-domain-vocab.txt'})
        tokenizer_type {str} -- Type of tokenizer ('bert' or 'roberta')

    Returns:
        A BertWordPieceTokenizer or ByteLevelBPETokenizer trained on in-domain corpora.
    """
    if tokenizer_type == 'bert':
        tokenizer_class = BertWordPieceTokenizer
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    elif tokenizer_type == 'roberta' or tokenizer_type == 'bpe_from_scratch':
        tokenizer_class = ByteLevelBPETokenizer
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    elif tokenizer_type == 'bpe_from_scratch':
        tokenizer_class = ByteLevelBPETokenizer
        vocab_size = 50000
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    else:
        raise Exception("unsupported tokenizer type")

    if isinstance(corpus, list):
        if isdir(corpus[0]):
            old_corpus = corpus
            corpus = []
            for corp in old_corpus:
                corpus.extend(get_text_files(corp))
    else:
        corpus = get_text_files(corpus)

    # Load cached vocab if possible
    if not overwrite:
        cached_vocab = Path(dst) / (VOCAB_CACHE_PREFIX + '-vocab.txt')

        if cached_vocab.exists():
            logger.info(f'Loading cached vocabulary at {cached_vocab}')
            return tokenizer_class(str(cached_vocab))
        else:
            logger.info(f'Cached vocabulary not found at {cached_vocab}')

    # Train tokenizer
    logger.info('Training new WordPiece tokenizer on in-domain corpora')
    tokenizer = tokenizer_class(**tokenizer_kwargs)
    tokenizer.train(corpus, vocab_size=vocab_size, special_tokens=special_tokens)

    if save_vocab:
        tokenizer.save('.' if dst is None else dst, in_domain_vocab)
    logger.info('Saved in-domain vocabulary to '
                f'{Path(dst) / (in_domain_vocab + "-vocab.txt")}')
    return tokenizer


def tokenize(texts: List[str],
             tokenizer: Union[BertWordPieceTokenizer, ByteLevelBPETokenizer],
             flat_map: bool = False,
            ) -> Union[List[str],
                       List[List[str]]]:
    """Tokenize texts using BERT WordPiece or ByteLevelBPE tokenizer implemented in Rust.

    Arguments:
        texts {List[str]} -- Text data to tokenize
        tokenizer {BertWordPieceTokenizer, ByteLevelBPETokenizer}
            -- A BertWordPieceTokenizer or ByteLevelBPETokenizer from the `tokenizers` library
        flat_map {bool} -- If True, flat maps results into a List[str],
                           instead of List[List[str]].

    Returns:
        A tokenized string or a list of tokenized string.
    """
    # Instantiate the tokenizer
    if not hasattr(tokenizer, 'encode_batch'):
        raise AttributeError(f'Provided `tokenizer` is not from `tokenizers` '
                             'library.')

    if flat_map:
        tokenized = [t for enc in tokenizer.encode_batch(texts)
                       for t in enc.tokens]
    else:
        tokenized = [enc.tokens for enc in tokenizer.encode_batch(texts)]
    return tokenized


def rank_tokens(tokenized_docs: List[List[str]],
                mode: str = 'count'
               ) -> List[str]:
    """Rank in-domain tokens.

    This ranking is used to decide which tokens are used to replcae the
    [USUSED*] tokens in BERT's vocabulary.

    Ranking heuristic:
        "count" -- Rank tokens by freq. of occurence in desc. order
        "tfidf" -- Rank tokens by TFIDF in desc. order

    Arguments:
        tokens {List[str]} -- The tokenized corpus. Inner list represents
                              a tokenized document in the corpus.

    Keyword Arguments:
        mode {str} -- Ranking heuristic. Choose between {'count' and 'tfidf'}
                      (default: {'count'})

    Returns:
        List[str] -- A ranked list of tokens
    """
    MODES = ('count', 'tfidf')
    if mode not in MODES:
        raise ValueError(f'Invalid mode {mode} provided. '
                         f'Expecting value from {MODES}.')

    logger.info(f'Ranking tokens by {mode}')
    if mode == 'count':
        return _rank_tokens_by_count(tokenized_docs)
    else:
        return _rank_tokens_by_tfidf(tokenized_docs)


def _rank_tokens_by_count(tokenized_docs: List[List[str]]) -> List[str]:
    tokens = [t for tokens in tokenized_docs for t in tokens]
    tokens = pd.Series(tokens)
    ranked = tokens.value_counts().index.tolist()
    return ranked


def _rank_tokens_by_tfidf(tokenized_docs: List[List[str]]) -> List[str]:
    # Convert the list of tokens in each doc into a space-delimited string
    documents = pd.Series(tokenized_docs).apply(lambda x: ' '.join(x))

    # Fit a TfidfVectorizer
    tfidf = TfidfVectorizer(lowercase=False, max_features=5000,
                            use_idf=True, smooth_idf=True,
                            token_pattern='\S+')
    tfidf.fit(documents)

    # Get TFIDFs for corpora
    corpus_str = ' '.join([t for tokens in tokenized_docs for t in tokens])
    tfidfs = np.array(tfidf.transform([corpus_str]).todense()).squeeze()
    ranked = (
        pd.Series(tfidfs, index=sorted(tfidf.vocabulary_.keys(),
                                       key=lambda x: x.__getitem__))
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return ranked


def _get_stop_words() -> List[str]:
    """Load stop words from NLTK.

    Will attempt to download stopwords and reload up to 5 times.
    """
    num_retries = 5
    for _ in range(num_retries):
        try:
            stopwords = nltk.corpus.stopwords.words('english')
        except LookupError:  # Stopwords folder not yet downloaded
            nltk.download('stopwords')
            continue
        break
    else:
        raise ValueError(f'{num_retries} attempts at loading stopwords failed.')
    return stopwords


def create_updated_vocab_txt(top_terms: List[str],
                             ori_vocab_path: str,
                             updated_vocab_path: str
                             ) -> None:
    """Update tokenizer vocabulary with relevant in-domain tokens.

    This is done by replacing '[unused*]' tokens in BERT's vocabulary with
    in-domain terms that do not already exist in the existing vocabulary.

    Results are saved in a txt file.

    Arguments:
        top_terms {List[str]} -- Ranked in-domain terms in descending order

    Keyword Arguments:
        ori_vocab_path {str} -- Path to existing vocabulary txt file
        updated_vocab_path {str} -- Path to save updated vocabulary txt file
        tokenizer_type {str} -- Type of tokenizer (bert or roberta)
    """

    logger.info('Updating vocabulary')
    # Get stop words
    stopwords = _get_stop_words() + ["[CLS]", "[SEP]"]

    # Get original vocab
    with open(ori_vocab_path) as f:
        vocab = [x.strip() for x in f.readlines()]
        unused_tokens = [x for x in vocab if '[unused' in x]

    # Filter out tokens that are not stop words or part of existing vocab
    top_terms = [
        x
        for x in top_terms
        if (x not in stopwords and x not in vocab)
    ]

    # Create top term generator

    mapping = dict(zip(unused_tokens, top_terms))
    assert len(unused_tokens) <= len(top_terms)  # TODO Handle the inverse situation

    # Update original vocab with the next top term if the token is '[unused*]'
    for i, ori_term in enumerate(vocab):
        if ori_term in mapping:
            vocab[i] = mapping[ori_term]

    # Saves vocab
    if updated_vocab_path[:5] == 's3://':
        with open('vocab.txt', 'w') as f:
            f.write('\n'.join(vocab))
        bucket, key = updated_vocab_path[5:].split('/', 1)
        updated_vocab_txt = f"{key}/vocab.txt"
        get_s3().upload_file('vocab.txt', bucket, updated_vocab_txt)
    else:
        updated_vocab_txt = (Path(updated_vocab_path) / 'vocab.txt').as_posix()
        with open(updated_vocab_txt, 'w+') as f:
            f.write('\n'.join(vocab))
    logger.info(f'Updated vocabulary saved at {updated_vocab_txt}')


def create_updated_vocab_for_bpe(top_terms: List[str],
                                 ori_vocab_path: List[str],
                                 updated_vocab_path: str,
                                 updated_merges_file: str
                                 ) -> None:
    """Update tokenizer vocabulary with relevant in-domain tokens.

    This is done by replacing '[unused*]' tokens in BERT's vocabulary with
    in-domain terms that do not already exist in the existing vocabulary.

    Results are saved in a txt file.

    Arguments:
        top_terms {List[str]} -- Ranked in-domain terms in descending order

    Keyword Arguments:
        ori_vocab_path {str} -- Path to existing vocabulary txt file
        updated_vocab_path {str} -- Path to save updated vocabulary txt file
        tokenizer_type {str} -- Type of tokenizer (bert or roberta)
    """
    logger.info('Updating vocabulary')
    # Get stop words
    stopwords = _get_stop_words() + ["[CLS]", "[SEP]"]

    ori_vocab = ori_vocab_path[0]
    ori_merges = ori_vocab_path[1]

    # Get original vocab
    with open(ori_vocab) as f:
        data = json.load(f)
        vocab = [None] * len(data)
        for token, id in data.items():
            vocab[id] = token
        if None in vocab:
            print("missing vocab")

    # Filter out tokens that are not stop words or part of existing vocab
    orig_number_top_terms = len(top_terms)
    top_terms = [
        x
        for x in top_terms
        if (x not in stopwords and x not in vocab)
    ]
    print(f"removed {orig_number_top_terms - len(top_terms)} tokens from top_terms")

    # Create top term generator
    assert len(top_terms) >= 1000
    vocab.extend(top_terms[0:1000])

    # get original merges
    orig_merges_data = open(ori_merges, encoding='utf-8').read().split('\n')[1:-1]
    updated_merges_data = open(updated_merges_file, encoding='utf-8').read().split('\n')[1:-1]

    # combine merges
    combined_merges_data = set(orig_merges_data + updated_merges_data)

    trimmed_merges = []
    # go through and remove any merges of tokens which aren't in vocab
    for merge_data in combined_merges_data:
        first, second = merge_data.split(' ')
        if first not in vocab or second not in vocab:
            continue
        trimmed_merges.append(merge_data)

    # write updated merges to output
    if updated_vocab_path[:5] == 's3://':
        with open('merges.txt', 'w') as f:
            f.write('\n'.join(trimmed_merges))
        bucket, key = updated_vocab_path[5:].split('/', 1)
        updated_merges_txt = f"{key}/merges.txt"
        get_s3().upload_file('merges.txt', bucket, updated_merges_txt)
        updated_merges_txt = f"s3://{bucket}/{updated_merges_txt}"
    else:
        updated_merges_txt = (Path(updated_vocab_path) / 'merges.txt').as_posix()
        if os.path.exists(updated_merges_txt):
            os.remove(updated_merges_txt)
        with open(updated_merges_txt, 'w') as f:
            f.write('\n'.join(trimmed_merges))
    logger.info(f"Saved final merges to {updated_merges_txt}")

    # write updated vocab to output
    if updated_vocab_path[:5] == 's3://':
        with open('vocab.json', 'w') as f:
            json.dump({i: v for v, i in enumerate(vocab)}, f, ensure_ascii=False)
        bucket, key = updated_vocab_path[5:].split('/', 1)
        updated_vocab_json = f"{key}/vocab.json"
        get_s3().upload_file('vocab.json', bucket, updated_vocab_json)
        updated_vocab_json = f"s3://{bucket}/{updated_vocab_json}"
    else:
        updated_vocab_json = (Path(updated_vocab_path) / 'vocab.json').as_posix()
        if os.path.exists(updated_vocab_json):
            os.remove(updated_vocab_json)
        with open(updated_vocab_json, 'w') as f:
            json.dump({i: v for v, i in enumerate(vocab)}, f, ensure_ascii=False)
    logger.info(f"Saved final vocab to {updated_vocab_json}")


def create_new_vocab_for_bpe(top_terms: List[str],
                             updated_vocab_path: str,
                             updated_merges_file: str
                             ) -> None:
    """Create tokenizer vocabulary with relevant in-domain tokens.

    Results are saved in a txt file.

    Arguments:
        top_terms {List[str]} -- Ranked in-domain terms in descending order

    Keyword Arguments:
        updated_vocab_path {str} -- Path to save updated vocabulary json file
        updated_merges_file {str} -- Path to save updated merges txt file
        tokenizer_type {str} -- Type of tokenizer (bert or roberta)
    """
    logger.info('Updating vocabulary')
    vocab = top_terms

    # get original merges
    merges_data = open(updated_merges_file, encoding='utf-8').read().split('\n')[1:-1]

    # combine merges
    merges_data = set(merges_data)

    trimmed_merges = []
    # go through and remove any merges of tokens which aren't in vocab
    for merge_data in merges_data:
        first, second = merge_data.split(' ')
        if first not in vocab or second not in vocab:
            continue
        trimmed_merges.append(merge_data)

    # write updated merges to output
    if updated_vocab_path[:5] == 's3://':
        with open('merges.txt', 'w') as f:
            f.write('\n'.join(trimmed_merges))
        bucket, key = updated_vocab_path[5:].split('/', 1)
        updated_merges_txt = f"{key}/merges.txt"
        get_s3().upload_file('merges.txt', bucket, updated_merges_txt)
        updated_merges_txt = f"s3://{bucket}/{updated_merges_txt}"
    else:
        updated_merges_txt = (Path(updated_vocab_path) / 'merges.txt').as_posix()
        if os.path.exists(updated_merges_txt):
            os.remove(updated_merges_txt)
        with open(updated_merges_txt, 'w') as f:
            f.write('\n'.join(trimmed_merges))
    logger.info(f"Saved final merges to {updated_merges_txt}")

    # write updated vocab to output
    if updated_vocab_path[:5] == 's3://':
        with open('vocab.json', 'w') as f:
            json.dump({i: v for v, i in enumerate(vocab)}, f, ensure_ascii=False)
        bucket, key = updated_vocab_path[5:].split('/', 1)
        updated_vocab_json = f"{key}/vocab.json"
        get_s3().upload_file('vocab.json', bucket, updated_vocab_json)
        updated_vocab_json = f"s3://{bucket}/{updated_vocab_json}"
    else:
        updated_vocab_json = (Path(updated_vocab_path) / 'vocab.json').as_posix()
        if os.path.exists(updated_vocab_json):
            os.remove(updated_vocab_json)
        with open(updated_vocab_json, 'w') as f:
            json.dump({i: v for v, i in enumerate(vocab)}, f, ensure_ascii=False)
    logger.info(f"Saved final vocab to {updated_vocab_json}")


def get_s3():
    global s3
    if s3 is None:
        s3 = client('s3')
    return s3

def get_s3r():
    global s3r
    if s3r is None:
        s3r = resource('s3')
    return s3r

def get_text_files(corpus, s3_mem_limit=None):
    if corpus[:5] == 's3://':
        bucket, key = corpus[5:].split('/', 1)
        os.makedirs(key, exist_ok=True)
        s3_bucket = get_s3r().Bucket(bucket)
        filenames = [object.key for object in s3_bucket.objects.filter(Prefix=key)]
        mem_per_file = int((s3_mem_limit * 1e6) // len(filenames))
        for filename in filenames:
            if filename == key:
                continue
            if not os.path.exists(filename):
                if s3_mem_limit is None:
                    s3_bucket.download_file(filename, filename)
                else:
                    data = get_s3().get_object(Bucket=bucket, Key=filename, Range=f"bytes=0-{mem_per_file}")['Body'].read()
                    try:
                        text_data = data.decode('utf-8-sig')
                    except UnicodeDecodeError:
                        try:
                            text_data = data.decode('utf-8')
                        except UnicodeDecodeError:
                            text_data = data.decode('iso-8859-1')
                    with open(filename, 'w') as f:
                        f.write(text_data)
                    mem_per_file
        return filenames
    else:
        return glob.glob(f"{corpus}/**.txt")


def isdir(corpus):
    if corpus[:5] == 's3://':
        bucket, key = corpus[5:].split('/', 1)
        contents = get_s3().list_objects(Bucket=bucket, Prefix=key)['Contents']
        return len(contents) > 1
    else:
        return os.path.isdir(corpus)


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create directory
    Path(args.dst).mkdir(exist_ok=True, parents=True)

    # create tokenizer kwargs
    if args.tokenizer_type == 'bert':
        tokenizer_kwargs = {'lowercase': args.lowercase}
        update_vocab_fn = create_updated_vocab_txt
    elif args.tokenizer_type == 'roberta':
        tokenizer_kwargs = {} #{'vocab_file': args.tokenizer_vocab[0], 'merges_file': args.tokenizer_vocab[1]}
        update_vocab_fn = create_updated_vocab_for_bpe
    elif args.tokenizer_type == 'bpe_from_scratch':
        tokenizer_kwargs = {}
        update_vocab_fn = create_new_vocab_for_bpe
    else:
        raise Exception("unsupported tokenizer type")

    corpus = args.corpus
    if isinstance(corpus, list):
        if isdir(corpus[0]):
            old_corpus = corpus
            corpus = []
            for corp in old_corpus:
                corpus.extend(get_text_files(corp, s3_mem_limit=args.s3_mem_limit))
    else:
        corpus = get_text_files(corpus)

    tokenizer = train_tokenizer(corpus,
                                overwrite=args.overwrite_cache,
                                lowercase=args.lowercase,
                                dst=args.dst,
                                save_vocab=True,
                                tokenizer_type=args.tokenizer_type,
                                tokenizer_kwargs=tokenizer_kwargs)

    if args.tokenizer_type == 'bpe_from_scratch':
        shutil.move(f"{args.dst}/temp-in-domain-merges.txt", f"{args.dst}/merges.txt", )
        shutil.move(f"{args.dst}/temp-in-domain-vocab.json", f"{args.dst}/vocab.json", )
        return

    # Load corpus / corpora
    tokenized_corpus = []
    for c in corpus:
        logger.info(f'Tokenizing {c} with in-domain tokenizer')
        with open(c) as f:
            text = f.readlines()
            tokenized_corpus += tokenize(text, tokenizer=tokenizer)
    #tokenized_corpus = tokenize([open(c).read() for c in corpus], tokenizer=tokenizer)

    # Rank tokens
    ranked_tokens = rank_tokens(tokenized_corpus, mode=args.rank_by)

    # Save augmented vocabulary to text file
    updated_vocab_path = args.dst
    tokenizer_vocab = args.tokenizer_vocab

    update_kwargs = {'updated_vocab_path': updated_vocab_path}
    if args.tokenizer_type == 'bert':
        if isinstance(tokenizer_vocab, list):
            tokenizer_vocab = tokenizer_vocab[0]
        update_kwargs['ori_vocab_path'] = tokenizer_vocab
    elif args.tokenizer_type == 'roberta':
        # this is needed for roberta so the merges file can be obtained in the update_vocab_fn
        tokenizer._tokenizer.model.save('.', 'additional')
        updated_merges_file = 'additional-merges.txt'
        update_kwargs['updated_merges_file'] = updated_merges_file
        update_kwargs['ori_vocab_path'] = tokenizer_vocab
    elif args.tokenizer_type == 'bpe_from_scratch':
        # this is needed for roberta so the merges file can be obtained in the update_vocab_fn
        tokenizer._tokenizer.model.save('.', 'additional')
        updated_merges_file = 'additional-merges.txt'
        update_kwargs['updated_merges_file'] = updated_merges_file

    update_vocab_fn(ranked_tokens, **update_kwargs)

if __name__ == '__main__':
    main()
