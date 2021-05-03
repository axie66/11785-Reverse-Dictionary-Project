import torch
import numpy as np

def count_related_forms(topk, labels, wn_data, idx2target, related_forms):
    '''
    topk: (batch, k)
    '''
    correct_counts = {form: 0 for form in related_forms}
    total_counts = {form: 0 for form in related_forms}
    topk = topk.cpu().numpy()

    for p, idx in zip(topk, labels):
        word_correct = []
        word_total = []
        for form in related_forms:
            related = wn_data[idx2target[idx]][form]
            if len(related) == 0:
                continue
            occurrences = intersection(p, np.array(related))
            correct_counts[form] += occurences
            total_counts[form] += len(related)

    results = {}
    for form in related_form: 
        results[form] = correct_counts[form] / (total_counts[form]
                                                if total_counts[form] else 1)

    return results

def intersection(a, b):
    return len(np.intersect1d(a, b, assume_unique=True))
            
