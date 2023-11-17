import numpy as np


def mean_phd(samples):
    '''
    samples: sample size adjacency matrices
    Returns: Pairwise Hamming Distance (PHD) in the batch
    Hamming Distance describes the proportion of disagreeing components between two boolean adjacency matrices
    '''

    batch_distance = 0.
    for i in samples:
        edge_mul = i * samples
        sparse_mul = (1 - i) * (1 - samples)
        mul = edge_mul + sparse_mul
        normalized_distance = 1. - mul.sum() / samples.size  # normalization at the node level
        batch_distance += normalized_distance
    normalized_batch_distance = batch_distance / len(samples)
    return normalized_batch_distance


def shd(target, pred):
    # Structural Hamming Distance (SHD) for comparing graphs by their adjacency matrix
    edge_mul = target * pred
    sparse_mul = (1 - target) * (1 - pred)
    mul = edge_mul + sparse_mul
    normalized_distance = 1. - mul.sum() / pred.size  # normalization at the node level
    return normalized_distance


def mean_shd(target, pred_samples):
    structural_hamming_distance = 0.
    for pred in pred_samples:
        structural_hamming_distance += shd(target, pred)
    return structural_hamming_distance / len(pred_samples)


def jensen_shannon_divergence(distribution1, distribution2):
    assert isinstance(distribution1, dict)
    assert isinstance(distribution2, dict)

    graphs = sorted(list(distribution1.keys()), key=len)

    # Get the two distributions aligned
    log_probs1, log_probs2 = [], []
    for graph in graphs:
        log_probs1.append(distribution1[graph])
        log_probs2.append(distribution2[graph])
    log_probs1 = np.array(log_probs1, dtype=np.float_)
    log_probs2 = np.array(log_probs2, dtype=np.float_)

    # Compute the mean distribution
    log_probs_mean = np.log(0.5) + np.logaddexp(log_probs1, log_probs2)

    # Compute the JSD
    kl1 = np.exp(log_probs1) * (log_probs1 - log_probs_mean)
    kl2 = np.exp(log_probs2) * (log_probs2 - log_probs_mean)
    return 0.5 * np.sum(kl1 + kl2)


def entropy(distribution):
    assert isinstance(distribution, dict)

    log_probs = np.asarray(list(distribution.values()))
    return -np.sum(np.exp(log_probs) * log_probs)
