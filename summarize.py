import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


def read_article(text):
    article = text.split(". ")    # split the text by sentences using ". "
    sentences = []
    for sentence in article:             # iterate thru sentences, printing each and generate list of wards for each sentence
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))    # replace any non character by " "
    #sentences.pop()   ##### systematically eliminate last sentence of the text from the returned sentences??
    
    return sentences

def sentence_similarity(sentence_1, sentence_2, stopwords=None):
    if stopwords is None:
        stopwords = []     # create an empty list to avoid error below
 
    sentence_1 = [w.lower() for w in sentence_1]
    sentence_2 = [w.lower() for w in sentence_2]

    all_words = list(set(sentence_1 + sentence_2))  # create total vocabulary of unique words for the two sentences compared

    vector1 = [0] * len(all_words)                  # prepare one-hot vectors for each sentence over all vocab
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sentence_1:
        if w in stopwords:
            continue 
        vector1[all_words.index(w)] += 1           # list.index(element) returns the index of the given element in the list

    # build the vector for the second sentence
    for w in sentence_2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - np.nan_to_num(cosine_distance(vector1, vector2))   # Cosine = 0 for similar sentences => returns 1 if perfectly similar
    
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))  # create a square matrix with dim the num of sentences
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences (diagonal of the square matrix)
                continue
            # similarity of each sentence to all other sentences in the text is measured and logged in the matrix
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n=5, show=False):
    #stop_words = stopwords.words('english')
    stop_words = stopwords.words('indonesian')
    summarize_text = []
    
    # Step 1 - Read text and tokenize
    sentences =  read_article(text)
    print("number of sentences in text : ", len(sentences))
    
    # Step 2 - Generate Similary Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Step 3 - Rank sentences in similarity matrix. let’s convert the similarity matrix into a graph. 
    # The nodes of this graph will represent the sentences and the edges will represent the similarity scores between
    # the sentences. On this graph, we will apply the PageRank algorithm to arrive at the sentence rankings.
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Step 4 - Sort the rank and pick top sentences extract the top N sentences based on their rankings for summary generation
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    if show :
        print("Indexes of top ranked_sentence order are ", ranked_sentence)
    # extract the top N sentences based on their rankings for summary generation
    if top_n < len(sentences):
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    else:
        for i in range(len(sentences)-1):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    # Step 5 - Output the summarize text
    print("done")
    return ". ".join(summarize_text)