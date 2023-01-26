import nltk
import sys
import os
import string
import itertools
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    d = {}
    for path, _, files in os.walk(directory):
        for file in files:
            with open(os.path.join(path, file), "r", encoding="utf8") as f:
                content = f.read()
            d[file] = content

    return d


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """

    def not_only_punctuation(s):
        # Returns False if all characters in s are punctuation
        # Returns True if s contains at least one char not in string.punctuation
        return any(c for c in s if c not in string.punctuation)

    tokenized = nltk.tokenize.word_tokenize(document)

    return [i.lower()
            for i in tokenized
            if not_only_punctuation(i) and i not in nltk.corpus.stopwords.words("english")]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    doc_num = len(documents)
    words = set(itertools.chain.from_iterable(documents.values()))
    # Keep track of number of documents that contains word
    contains_word = {}

    for word in words:
        count = 0
        for document in documents.values():
            if word in document:
                count += 1
        contains_word[word] = count

    return {word: math.log(doc_num / count) for word, count in contains_word.items()}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Keep track of file ranks
    ranks = []
    for file in files:
        # Ranks according to sum of tf-idf values
        tf_idf = 0
        for word in query:
            if word in files[file]:
                tf_idf += idfs[word] * files[file].count(word)
        ranks.append((file, tf_idf))

    ranks.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Keep track of file ranks
    ranks = []
    for sentence in sentences:
        # Ranks according to matching word measure, then query term density
        match_word_measure = 0
        for word in query:
            if word in sentences[sentence]:
                match_word_measure += idfs[word]

        query_term_density = sum([sentences[sentence].count(word) for word in query]) / len(sentences[sentence])
        ranks.append((sentence, match_word_measure, query_term_density))

    ranks.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in ranks[:n]]


if __name__ == "__main__":
    main()
