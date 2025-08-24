import numpy as np
import torch
#from sklearn.metrics.pairwise import cosine_similarity 
from sentence_transformers.util import cos_sim

def specificity(
    embs_in, # questions
    embs_out, # answers
    metric = cos_sim, # cosine_similarity,
    weighted:bool=False,
    softmax:bool=True,
    log:bool=True,
):
    """
    embs_in (N_in  x d)
    embs_out (N_out x d)
    """
    n_in = len(embs_in)
    n_out = len(embs_out)

    if weighted:
        weights = metric(embs_in[:len(embs_out)], embs_in) # N_in x N_in
    else:
        weights = torch.ones((n_out,n_in)) # N_out x N_in
    
    if softmax:
        matrix_sims = (weights*metric(embs_out, embs_in)).exp() # N_out x N_in : M_ij = exp(w_ij * cos(embs1_i, embs2_j))
    else: 
        matrix_sims = weights*metric(embs_out, embs_in) # N_out x N_in : M_ij = w_ij * cos(embs1_i, embs2_j)
    
    numerators = matrix_sims.diag() # N_out
    matrix_sims[range(len(matrix_sims)), range(len(matrix_sims))] = 0 # avoid self-influence
    normalizers = weights.sum(axis=1)-1 # N_out 
    denominators = matrix_sims.sum(axis=1)/normalizers # N_out 

    if log:
        return np.log((numerators/denominators).numpy()) # N_out
    else: 
        return (numerators/denominators).numpy()