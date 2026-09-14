import unittest
import torch
from predictive_tempering import temper_factor

class PredictiveTemperingTest(unittest.TestCase):
    def test_uniform_or_compatible_prior_is_exact(self):
        factor=torch.tensor([1.,3.])
        self.assertIs(temper_factor(torch.tensor([.5,.5]),factor)[0],factor)
        self.assertIs(temper_factor(torch.tensor([.1,.9]),factor)[0],factor)

    def test_conflict_is_softened_and_scale_invariant(self):
        prior=torch.tensor([.9,.1]);factor=torch.tensor([1.,9.])
        softened,decision=temper_factor(prior,factor)
        self.assertGreater(decision['exponent'],0.)
        self.assertLess(decision['exponent'],1.)
        self.assertAlmostEqual(decision['exponent'],temper_factor(prior,7*factor)[1]['exponent'],places=7)
        plain=prior*factor;plain/=plain.sum()
        updated=prior*softened;updated/=updated.sum()
        self.assertLess(float((updated-prior).abs().sum()),float((plain-prior).abs().sum()))

    def test_reference_prior_is_not_mutated(self):
        prior=torch.tensor([.7,.3]);saved=prior.clone()
        first=temper_factor(prior,torch.tensor([.2,1.]))[1]
        temper_factor(prior,torch.tensor([3.,.1]))
        self.assertEqual(first,temper_factor(prior,torch.tensor([.2,1.]))[1])
        self.assertTrue(torch.equal(prior,saved))

if __name__=='__main__':unittest.main()
