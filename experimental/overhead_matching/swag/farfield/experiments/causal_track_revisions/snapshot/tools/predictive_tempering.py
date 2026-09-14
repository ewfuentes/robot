"""Experimental prior-predictive likelihood tempering; no truth inputs."""
import torch

def temper_factor(prior, factor):
    if prior.shape!=factor.shape:
        raise ValueError('Prior and factor shapes differ')
    predictive=(prior.double()*factor.double()).sum()/prior.double().sum()
    uniform=factor.double().mean()
    if not bool(torch.isfinite(predictive)) or not bool(torch.isfinite(uniform)) or float(uniform)<=0:
        raise ValueError('Invalid predictive evidence')
    ratio=float(predictive/uniform)
    exponent=max(0.,min(1.,ratio))
    decision=dict(predictive_over_uniform=ratio,exponent=exponent,
        policy='min_one_prior_predictive_over_uniform_v1')
    return (factor if exponent==1. else factor.pow(exponent)),decision
