import numpy as np
import scipy

def topic_relevance_score(topic_term_dist, doc_topic_dist):
    raise NotImplemented

def uniform_topic_over_words(topic_term_prob):
    """Kullback-Leibler divergence between P(w|t) and 1/len(w)"""

    n_topics, n_terms = topic_term_prob.shape
    uniform = np.ones(n_terms) / n_terms
    kl = np.zeros(n_topics)
    for t in range(n_topics):
        kl[t] = scipy.stats.entropy(topic_term_prob[t], uniform)
    return kl

def weighted_uniform_topic_over_words(topic_term_prob, topic_prob):
    n_topics, n_terms = topic_term_prob.shape
    uniform = np.dot(topic_prob, topic_term_prob)
    kl = np.zeros(n_topics)
    for t in range(n_topics):
        kl[t] = scipy.stats.entropy(topic_term_prob[t], uniform)
    return kl

def uniform_topic_over_docs(doc_topic_prob):
    n_docs, n_topics = doc_topic_prob.shape
    uniform = np.ones(n_docs) / n_docs
    kl = np.zeros(n_topics)
    for t in range(n_topics):
        td = doc_topic_prob[:,t]
        td = td / td.sum()
        td = td.reshape(-1)
        kl[t] = scipy.stats.entropy(td, uniform)
    return kl

def uniform_word_over_topics(topic_term_prob, term_prob):
    n_topics, n_terms = topic_term_prob.shape
    term_topic_prob = topic_term_prob.T
    uniform = np.ones(n_topics) / n_topics
    kl = np.zeros(n_terms)
    for t in range(n_terms):
        tt = term_topic_prob[t]
        tt = tt / tt.sum()
        kl[t] = scipy.stats.entropy(tt, uniform)
    return kl