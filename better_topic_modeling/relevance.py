import numpy as np
import scipy

def topic_relevance_score(topic_term_dist, doc_topic_dist):
    raise NotImplemented

def w_uniform(topic_term_prob):
    """Kullback-Leibler divergence between P(w|t) and 1/len(w)"""

    n_topics, n_terms = topic_term_prob.shape
    u_prob = np.ones(n_terms) / n_terms
    kl_w = np.zeros(n_topics)
    for t in range(n_topics):
        kl_w[t] = scipy.stats.entropy(topic_term_prob[t], u_prob)
    return kl_w

def ww_uniform(topic_term_prob, topic_prob):
    n_topics, n_terms = topic_term_prob.shape
    u_prob = np.dot(topic_prob, topic_term_prob)
    kl_ww = np.zeros(n_topics)
    for t in range(n_topics):
        kl_ww[t] = scipy.stats.entropy(topic_term_prob[t], u_prob)
    return kl_ww

def t_uniform(doc_topic_prob):
    n_docs, n_topics = doc_topic_prob.shape
    uniform = np.ones(n_docs) / n_docs
    kl_t = np.zeros(n_topics)
    for t in range(n_topics):
        td = doc_topic_prob[:,t]
        td = td / td.sum()
        td = td.reshape(-1)
        kl_t[t] = scipy.stats.entropy(td, uniform)
    return kl_t