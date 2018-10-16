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