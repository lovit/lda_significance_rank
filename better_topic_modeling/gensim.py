def get_parameters(lda, corpus):
    topic_term_prob = _get_topic_term_prob(lda)
    doc_topic_prob = _get_doc_topic_prob(lda, corpus)
    return topic_term_prob, doc_topic_prob

def _get_topic_term_prob(lda):
    topic_term_freq = lda.state.get_lambda()
    topic_term_prob = topic_term_freq / topic_term_freq.sum(axis=1)[:, None]
    return topic_term_prob

def _get_doc_topic_prob(lda, corpus):
    doc_topic_prob, _ = lda.inference(corpus)
    return doc_topic_prob