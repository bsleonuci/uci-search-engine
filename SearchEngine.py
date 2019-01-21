import pickle
import re
import os
import sys
import math
from bs4 import BeautifulSoup
import heapq
from collections import defaultdict


K_VALUE = 10
PICKLE_PATH = "saved_index"
WEBPAGE_PATH = "webpages/WEBPAGES_RAW"
BOOKKEEPING = "webpages/WEBPAGES_RAW/bookkeeping.tsv"
STOPWORDS = \
{"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
 "are", "aren", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
 "but", "by", "can", "cannot", "could", "couldn", "did", "didn", "do", "does", "doesn", "doing",
 "don", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn",
 "have", "haven", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
 "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "let", "me", "more", "most", "mustn",
 "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
 "our", "ours", "ourselves", "out", "over", "own", "same", "shan", "she", "should", "shouldn",
 "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
 "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
 "was", "wasn", "we", "were", "weren", "what", "when", "where", "which", "while", "who", "whom",
 "why", "with", "would", "wouldn", "you", "your", "yours", "yourself", "yourselves", "0",
 "1", "2", "3", "4", "5", "6", "7", "8", "9",}


USAGE_STR = """<--- Usage Information --->
    Commands:
    make_index - create index from scratch using raw webpage data on disk
    print_stats - print index unique term count, byte size, and document count
    save_index - save index to file using shelve
    load_index - load index from saved file
    query - query against current index
    usage - print usage information
    quit - end program"""


class Index:
    def __init__(self):
        self.dict = defaultdict(PostingsList)
        self.doc_length_dict = dict()
        self.doc_term_dict = defaultdict(set)

    def get_size(self):
        return sys.getsizeof(self.dict)

    def add_posting(self, token, posting, tag):
        copy = posting.get_copy()
        copy.add_tag(tag)
        self.dict[token].push(copy)

    def get_term_count(self):
        return len(self.dict.keys())

    def get_dict(self):
        #solely for update
        return self.dict

    def set_doc_term(self, posting, term):
        self.doc_term_dict[posting.get_index_pair()].add(term)

    def set_doc_length(self, posting, length):
        self.doc_length_dict[posting.get_index_pair()] = length

    def get_doc_count(self):
        return len(self.doc_length_dict.keys())

    def get_doc_term_dict(self):
        return self.doc_term_dict

    def get_doc_length_dict(self):
        return self.doc_length_dict

    def print_doclengths(self):
        for key, value in self.doc_length_dict.items():
            print(key, ":", value)

    def __getitem__(self, key):
        return self.dict.__getitem__(key)

    def __setitem__(self, key, value):
        self.dict.__setitem__(key, value)


class PostingsList:
    # PostingsList uses a dictionary as the underlying structure. Some methods will
    # sort the postings, but only when absolutely necessary for the operation.

    def __init__(self):
        self.dict = dict()

    def push(self, posting):
        self.dict[(posting.get_index_pair())] = posting

    def get_document_frequency(self):
        return len(self.dict.keys())

    def get_docset(self):
        return set([posting.get_index_pair() for posting in self.dict.values()])

    def __contains__(self, index_pair):
        if index_pair in self.dict:
            return True

    def __getitem__(self, index_pair):
        return self.dict[index_pair]

    def __iter__(self):
        self.i = 0
        self.sorted = list(self.dict.values())
        heapq.heapify(self.sorted)
        self.length = len(self.sorted)
        return self

    def __next__(self):
        if self.i < self.length:
            self.i += 1
            return heapq.heappop(self.sorted)
        else:
            raise StopIteration

    def __repr__(self):
        values = list(self.dict.values())
        string = "PostingsList[" + str(values[0])
        for posting in values[1:]:
            string += ", " + str(posting)
        string += "]"
        return string


class Posting:
    def __init__(self, i1, i2, tf):
        self.index1 = int(i1)
        self.index2 = int(i2)
        self.raw_tf = tf
        self.tagset = set()
        self.tf_idf = 0

    def get_tf_idf(self):
        return self.tf_idf

    def calculate_tf_idf(self, total_words, docs_containing, total_docs):
        self.tf_idf = calculate_tf_idf(self.raw_tf, total_words, docs_containing, total_docs)

    def get_copy(self):
        return Posting(self.index1, self.index2, self.raw_tf)

    def get_index_pair(self):
        return self.index1, self.index2

    def increment_tf(self):
        self.raw_tf += 1

    def get_tagset(self):
        return self.tagset

    def add_tag(self, tag):
        self.tagset.add(tag)

    def get_tf(self):
        return self.raw_tf


    #hash based on document index, not necessarily posting
    def __hash__(self):
        return hash((self.index1, self.index2))

    def __lt__(self, other):
        return self.index1 < other.index1 or (self.index1 == other.index1 and self.index2 < other.index2)


    def __le__(self, other):
        return self.index1 < other.index1 or (self.index1 == other.index1 and self.index2 <= other.index2)

    def __eq__(self, other):
        return self.index1 == other.index1 and self.index2 == other.index2

    def __gt__(self, other):
        return self.index1 > other.index1 or (self.index1 == other.index1 and self.index2 > other.index2)

    def __ge__(self, other):
        return self.index1 > other.index1 or (self.index1 == other.index1 and self.index2 >= other.index2)

    def __ne__(self, other):
        return self.index1 != other.index1 or self.index2 != other.index2

    def __repr__(self):
        return "Posting(" + str(self.index1) + ", " + \
                            str(self.index2) + ", " + \
                            "tf: " + str(self.raw_tf) + ", " + \
                            "tf_idf: " + str(self.tf_idf) + ", " + \
                            "tags: " + str(self.tagset) + " )"


def create_webpage_url_dict():
    # create a dictionary mapping dirnum/filenum to URL
    url_dict = dict()
    with open(BOOKKEEPING, "r") as file:
        for line in file:
            split = line.split()
            url_dict[split[0]] = split[1]
    print("URL dict created...\n")
    return url_dict


def get_url_for_path(fullpath, url_dict):
    reference = fullpath.split("\\")[1]
    return url_dict[reference]


# This is the tokenizing code from my Assignment 1 Part A
def tokenize(fullpath, index, url):
    #print("Tokenizing file: " + fullpath)
    full_doc_index = fullpath.split("\\")[1]
    doc_index1 = full_doc_index.split("/")[0]
    doc_index2 = full_doc_index.split("/")[1]
    base_posting = Posting(doc_index1, doc_index2, 1)
    token_count = 0
    # print("Baseposting is: " + str(base_posting))
        # set up regex to identify sequences of alphanumeric characters.
        # assume 4 integers in a row is a year, which is also an acceptable token
        # other integer-only tokens are not acceptable
    tokendef = re.compile(r"(?:[0-2][0-9][0-9][0-9])|(?:[a-z][a-z](?:[a-z]|[0-9])+)")
        #print(string.punctuation)
            #open and read file from command line
            #NOTE: only reads first argument, others can be supplied but will be ignored
    try:
        with open(fullpath, 'r', encoding="utf-8") as file:
            try:
                    text = file.read()
                    document = BeautifulSoup(text, features="lxml")
                    #remove non-visible tags from document to save parse time
                    [tag.decompose() for tag in document(['style', 'script', '[document]', 'meta'])]
                    for string in document.strings:
                        if not string.isspace():
                            tag = string.parent.name
                            #print(string, "descendant of", tag)
                            tokens = re.findall(tokendef, string.string.lower())
                            for token in tokens:
                                if token not in STOPWORDS:
                                    token_count += 1
                                    index.set_doc_term(base_posting, token)
                                    if base_posting.get_index_pair() in index[token]:
                                        current_posting = index[token][base_posting.get_index_pair()]
                                        current_posting.increment_tf()
                                        current_posting.add_tag(tag)
                                    #print(token + " -> " + str(index[token][base_posting]))
                                    else:
                                        index.add_posting(token, base_posting, tag)

                    if token_count > 0:
                        index.set_doc_length(base_posting, token_count)
                        #print(base_posting.get_url(), ":", token_count)
                # For every link found, print it, and add it to outputLinks if it is valid
            except UnicodeDecodeError:
                print("Could not read file: ", fullpath)
            except FileNotFoundError:
                print("File not in local directory: ", fullpath)
            except IOError:
                print("Error while reading file: ", fullpath)
            #except:
                #print("Document content could not be parsed from: " + fullpath + "\n")

    except UnicodeDecodeError:
        print("Could not read file: ", fullpath)
    except FileNotFoundError:
        print("File not in local directory: ", fullpath)
    except IOError:
        print("Error while reading file: ", fullpath)


def construct_index(url_dict):
    index = Index()
    count = 0
    for dirpath, dirnames, filenames in os.walk(WEBPAGE_PATH):
        for name in filenames:
            if (name != "bookkeeping.json" and name != "bookkeeping.tsv"):
                # print(dirpath + "/" + name)
                fullpath = dirpath + "/" + name
                url = get_url_for_path(fullpath, url_dict)
                tokenize(fullpath, index, url)
                count += 1
                if count == 5000:
                    count = 0
                    print("Processed 5000 documents...")
    print("Index construction complete!")
    return index

def min_set(index, terms):
    min = sys.maxsize
    minpair = ("", set())
    for term in terms:
        document_frequency = index[term].get_document_frequency()
        if document_frequency < min:
            min = document_frequency
            minpair = (term, index[term].get_docset())
    return minpair


def create_term_windows(terms, window, length):
    term_windows = []
    pos = 0
    while(pos + window <= length):
        term_windows.append(terms[pos:pos+window])
        pos += 1
    return term_windows

# does not consider all combinations, but prioritizes
# documents that contain greater numbers of query words,
# split up as ngrams

def process_ngrams(index, this_tier, terms, window, length):
    term_windows = create_term_windows(terms, window, length)
    for term_window in term_windows:
        minpair = min_set(index, term_window)
        intersection = minpair[1]
        for term in term_window:
            if term != minpair[0]:
                current_set = index[term].get_docset()
                intersection.intersection_update(current_set)
        #print("Intersection is:", intersection, "\nfor terms: ", term_window)
        this_tier = this_tier.union(intersection)
    return this_tier


def calculate_norm(vector):
    sum = 0
    for component in vector:
        sum += component * component
    return math.sqrt(sum)


def calculate_tf(raw_frequency, total_words):
    return 1 + math.log(raw_frequency / total_words, 10)


def calculate_idf(docs_containing, total_docs):
    return math.log((total_docs / (docs_containing + 1)), 10)


def calculate_tf_idf(raw_frequency, total_words, docs_containing, total_docs):
    tf = calculate_tf(raw_frequency, total_words)
    idf = calculate_idf(docs_containing, total_docs)
    return tf * idf


def cosine_similarity(query, doc, query_norm, doc_norm):
    length = len(query)
    dot_product_sum = 0
    for i in range(0, length):
        query_component = query[i]
        doc_component = doc[i]
        dot_product_sum += query_component * doc_component
    full_norm = query_norm * doc_norm

    return dot_product_sum / full_norm


def calculate_query_norm(query, index):
    sum = 0
    for word in query:
        tf_idf = calculate_tf_idf(1, 1, index[word].get_document_frequency(), index.get_doc_count())
        sum += tf_idf * tf_idf
    return math.sqrt(sum)


def calculate_doc_norm(doc_words, index, doc_index):
    sum = 0
    for word in doc_words:
        tf_idf = index[word][doc_index].get_tf_idf()
        sum += tf_idf * tf_idf
    return math.sqrt(sum)

def rank_tier(index, query, topk, K_VALUE, this_tier):
    ranked = []
    query_words = set(query)
    query_norm = calculate_query_norm(query, index)
    total_docs = index.get_doc_count()
    for doc_index in this_tier:
        score = 0
        tagbump = 0
        doc_words = index.get_doc_term_dict()[doc_index]
        doc_norm = calculate_doc_norm(doc_words, index, doc_index)
        query_vector = []
        doc_vector = []
        for term in query:
            docs_containing = index[term].get_document_frequency()
            if term in query_words:
                query_vector.append(calculate_tf_idf(1, len(query), docs_containing, total_docs))
                if term in doc_words and 'title' in index[term][doc_index].get_tagset():
                    tagbump += 0.10
            else:
                query_vector.append(0)
            if term in doc_words:
                doc_vector.append(index[term][doc_index].get_tf_idf())
            else:
                doc_vector.append(0)
        score = cosine_similarity(query_vector, doc_vector, query_norm, doc_norm)
        #prioritize results where query term appears in title
        score += tagbump
        ranked.append((score, doc_index))
    ranked = sorted(ranked, reverse=True)
    for doc in ranked:
        # avoid repeats
        if doc not in topk:
            topk.append(doc)
        if len(topk) == K_VALUE:
            return topk
    return topk


# ranking process begins here. we will only gather queries in topk up to K_VALUE.
# queries will be split into tiers based on number of query words contained.
# those tiers will then be ranked separately amongst themselves.
# as soon as a document's final score in that tier is determined, the values
# from that tier will be added to topk, sorted by highest score. if the
# number of documents in the tier is < K_VALUE, then the algorithm will
# calculate scores for the next tier down and repeat the process. The
# algorithm immediately stops once topk contains K_VALUE documents.

def get_topk(index, terms, K_VALUE):
    topk = list()
    length = len(terms)
    window = length
    while window > 0:
        this_tier = set()
        if window == 1:
            for term in terms:
                this_tier = this_tier.union(index[term].get_docset())
            topk = rank_tier(index, terms, topk, K_VALUE, this_tier)
        else:
            this_tier = process_ngrams(index, this_tier, terms, window, length)
            topk = rank_tier(index, terms, topk, K_VALUE, this_tier)
        if len(topk) == K_VALUE:
            return topk
        window -= 1
    # uh oh, didn't find K_VALUE results
    return topk


def trim_terms(index, terms):
    trimmed = []
    for term in terms:
        if term not in STOPWORDS:
            if term in index.get_dict().keys():
                trimmed.append(term)
    return trimmed


def process_query(index, query):
    terms = query.split()
    trimmed = trim_terms(index, terms)
    topk = get_topk(index, trimmed, K_VALUE)
    print("Top", K_VALUE, "results: ")
    count = 1
    for doc in topk:
        key = str(doc[1][0]) + "/" + str(doc[1][1])
        url = url_dict[key]
        print(url)
        count += 1


def query_phase(index):
    print("<--- QUERIES --->")
    query = ""
    while(query != "?!back"):
        query = input('Please input a query(enter "?!back" to return to shell): \n').lower()
        if query == "?!back":
            return
        else:
            process_query(index, query)
    print("Quitting query...")


def initialize_tf_idfs(index):
    doc_count = index.get_doc_count()
    for postings_list in index.get_dict().values():
        document_frequency = postings_list.get_document_frequency()
        for posting in postings_list:
            posting.calculate_tf_idf(index.get_doc_length_dict()[posting.get_index_pair()], index.get_doc_count(), document_frequency)
    return

def main_loop():
    index_loaded = False
    from_shelf = False
    command = ""
    print(USAGE_STR)
    while command != "quit":
        command = input(">")
        if command == "make_index":

            index = construct_index(url_dict)
            print("Initializing tf_idfs...")
            initialize_tf_idfs(index)
            print("Initialization complete!")
            index_loaded = True

        elif command == "print_stats":
            if index_loaded:
                print("Number of unique terms is", index.get_term_count())
                print("Number of docs is", index.get_doc_count())
            else:
                print("Cannot print stats... no index has been created/loaded...")
        elif command == "save_index":
            if index_loaded:
                print("Saving index...")
                with open(PICKLE_PATH, "wb") as f:
                    pickle.dump(index, f, pickle.HIGHEST_PROTOCOL)
                print("Index save complete!")
            else:
                print("Cannot save... no index has been created/loaded...")
        elif command == "load_index":
            if os.path.isfile(PICKLE_PATH):
                print("Loading index...")
                with open(PICKLE_PATH, "rb") as f:
                    index = pickle.load(f)
                index_loaded = True
                print("Index load complete!")
            else:
                print("Cannot load index, no index has been previously saved...")
        elif command == "query":
            if index_loaded:
                query_phase(index)
            else:
                print("Cannot query... no index has been created/loaded...")
        elif command == "usage":
            print(USAGE_STR)
        elif command == "quit":
            break

        else:
            print("Command not recognized...")
            print(USAGE_STR)

url_dict = create_webpage_url_dict()


if __name__ == "__main__":
    main_loop()
