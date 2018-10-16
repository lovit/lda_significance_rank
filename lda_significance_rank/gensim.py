import numpy as np
import six

from gensim import utils
from gensim.matutils import dirichlet_expectation
from six.moves import xrange

def get_parameters(lda, corpus):
    topic_term_prob = _get_topic_term_prob(lda)
    doc_topic_freq = _get_doc_topic_freq(lda, corpus)

    doc_topic_prob = doc_topic_freq / doc_topic_freq.sum(axis=1)[:, None]
    topic_prob = doc_topic_freq.sum(axis=0) / doc_topic_freq.sum()

    return topic_term_prob, doc_topic_prob, topic_prob

def _get_topic_term_prob(lda):
    topic_term_freq = lda.state.get_lambda()
    topic_term_prob = topic_term_freq / topic_term_freq.sum(axis=1)[:, None]
    return topic_term_prob

def _get_doc_topic_freq(lda, corpus, verbose=True):
    try:
        doc_topic_freq, _ = lda.inference(corpus)
    except:
        doc_topic_freq = inference_doc_topic_freq(lda, corpus, verbose)
    return doc_topic_freq

def inference_doc_topic_freq(lda, chunk, verbose=True, debug=True):
    """Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
    for each document in the chunk.

    This function was copied from Gensim github. Lower version of Gensim LdaModel is not compatible higher version.

    see more. https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py
    ----------
    chunk : {list of list of (int, float), scipy.sparse.csc}
        The corpus chunk on which the inference step will be performed.
    Returns
    -------
   numpy.ndarray
       Gamma matrix. doc - topic distribution
    """

    num_docs = len(chunk)
    num_topics = lda.num_topics
    dtype = np.float32
    random_state = utils.get_random_state(None)
    expElogbeta = lda.expElogbeta
    iterations = lda.iterations
    alpha = lda.alpha
    gamma_threshold = lda.gamma_threshold

    DTYPE_TO_EPS = {
        np.float16: 1e-5,
        np.float32: 1e-35,
        np.float64: 1e-100,
    }

    # Initialize the variational distribution q(theta|gamma) for the chunk
    gamma = random_state.gamma(100., 1. / 100., (num_docs, num_topics)).astype(dtype, copy=False)
    Elogtheta = dirichlet_expectation(gamma)
    expElogtheta = np.exp(Elogtheta)

    converged = 0

    # Now, for each document d update that document's gamma and phi
    # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
    # Lee&Seung trick which speeds things up by an order of magnitude, compared
    # to Blei's original LDA-C code, cool!).
    for d, doc in enumerate(chunk):
        if len(doc) > 0 and not isinstance(doc[0][0], six.integer_types + (np.integer,)):
            # make sure the term IDs are ints, otherwise np will get upset
            ids = [int(idx) for idx, _ in doc]
        else:
            ids = [idx for idx, _ in doc]
        cts = np.array([cnt for _, cnt in doc], dtype=dtype)
        gammad = gamma[d, :]
        Elogthetad = Elogtheta[d, :]
        expElogthetad = expElogtheta[d, :]
        expElogbetad = expElogbeta[:, ids]

        # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
        # phinorm is the normalizer.
        # TODO treat zeros explicitly, instead of adding epsilon?
        eps = DTYPE_TO_EPS[dtype]
        phinorm = np.dot(expElogthetad, expElogbetad) + eps

        # Iterate between gamma and phi until convergence
        for _ in xrange(iterations):
            lastgamma = gammad
            # We represent phi implicitly to save memory and time.
            # Substituting the value of the optimal phi back into
            # the update for gamma gives this update. Cf. Lee&Seung 2001.
            gammad = alpha + expElogthetad * np.dot(cts / phinorm, expElogbetad.T)
            Elogthetad = dirichlet_expectation(gammad)
            expElogthetad = np.exp(Elogthetad)
            phinorm = np.dot(expElogthetad, expElogbetad) + eps
            # If gamma hasn't changed much, we're done.
#             meanchange = mean_absolute_difference(gammad, lastgamma)
            meanchange = np.mean(np.abs(gammad - lastgamma))
            if meanchange < gamma_threshold:
                converged += 1
                break
        gamma[d, :] = gammad

        if verbose and d % 100 == 0:
            print('\rinfering %d / %d docs' % (d+1, num_docs), end='', flush=True)
        if debug and d >= 100:
            break

    if verbose:
        print('\rinfering %d / %d docs was done' % (d, num_docs))

    return gamma