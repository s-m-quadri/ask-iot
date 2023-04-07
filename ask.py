import nltk
import sys
import os
import string
import math

FILE_MATCHES = 5
SENTENCE_MATCHES = 3


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

    while True:
        # Prompt user for query
        query = set(tokenize(input("Ask>>> ")))
        if "exit" in query:
            sys.exit()

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
        print("Best Match Answers:")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match}")
        print()


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    filenames = os.listdir(directory)
    for filename in filenames:
        with open(os.path.join(directory, filename)) as text_file:
            files[filename] = "\n".join(text_file.readlines())
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = list()
    stop_words = nltk.corpus.stopwords.words("english")
    punctuations = string.punctuation
    for token in nltk.word_tokenize(document):
        # Filter stop words
        if token.lower() in stop_words:
            continue

        # Filter words in which punctuation is present
        valid = True
        for mark in punctuations:
            if mark in token:
                valid = False
        if not valid:
            continue

        # Lower the case and append
        words.append(token.lower())
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Organize data in sets, for faster operations onwards
    words = set()
    doc_words = list()
    for document in documents:
        cur_words = set(documents[document])
        doc_words.append(cur_words)
        words.update(cur_words)

    # Calculate Inverse Document frequency of each word
    # i.e prioritizing unique words
    idfs = dict()
    n_docs = len(documents)
    for word in words:
        n_docs_have_word = 0
        for doc_word in doc_words:
            if word in doc_word:
                n_docs_have_word += 1
        idfs[word] = math.log(n_docs / n_docs_have_word)
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Calculate key for sorting
    ranking = {filename: 0 for filename in files}
    for raw_word in query:
        word = raw_word.lower()
        if word in idfs:
            for filename in files:
                freq_word = files[filename].count(word)
                ranking[filename] += freq_word * idfs[word]

    # Rank the document based on keys
    ranking = sorted(ranking, key=lambda x: ranking[x], reverse=True)
    return ranking[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Calculate keys for sorting
    ranking = {sentence: 0 for sentence in sentences}
    density = {sentence: 0 for sentence in sentences}
    for sentence in sentences:
        word_freq = 0
        for word in sentences[sentence]:
            if word in query:
                ranking[sentence] += idfs[word]
                word_freq += 1
        density[sentence] = word_freq / len(sentences[sentence])

    # Rank the sentence based on keys
    final_ranking = sorted(ranking, key=lambda x: (
        ranking[x], density[x]*len(x)), reverse=True)
    return final_ranking[:n]


if __name__ == "__main__":
    main()
