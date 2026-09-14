import unittest
import torch
from track_revision_ratio import replacement_ratio

class ReplacementTests(unittest.TestCase):
    def test_static_pose_telescopes_without_recounting_old_observations(self):
        prior=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
        a=torch.tensor([2.,.2,1.,3.],dtype=torch.float64)
        b=torch.tensor([.4,.5,3.,1.],dtype=torch.float64)
        c=torch.tensor([1.,4.,.2,2.],dtype=torch.float64)
        for alpha in [0.,.45,1.]:
            actual=prior*a.pow(alpha)*replacement_ratio(b,a,alpha)*replacement_ratio(c,b,alpha)
            self.assertTrue(torch.allclose(actual,prior*c.pow(alpha),atol=1e-14,rtol=1e-14))
            self.assertTrue(torch.equal(replacement_ratio(a,a,alpha),torch.ones_like(a)))

    def test_invalid_factors_fail_instead_of_corrupting_posterior(self):
        a=torch.ones(3)
        for bad in [torch.zeros(3),torch.full((3,),float('nan')),torch.ones(2)]:
            with self.assertRaises(ValueError):replacement_ratio(a,bad,1.)

if __name__=='__main__':unittest.main()
