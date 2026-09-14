import unittest
from prefix_release_policy_v2 import ValidBirthPrefixRelease,birth_usable


class BirthAdmissionTest(unittest.TestCase):
    def track(self,ok):
        return dict(track_id=1,status='closed',birth_keyframe=0,
            close_reason='mask_dead' if ok else 'birth_fragmented',
            records=[dict(keyframe=0,action='birth',health=dict(ok=ok),supports=[])])

    def test_explicit_rejected_birth_never_emits(self):
        policy=ValidBirthPrefixRelease()
        self.assertIsNone(policy.consider(self.track(False),{0:0.,1:5.},1,5.))
        self.assertIsNone(policy.consider(self.track(False),{0:0.,1:5.},1,5.))

    def test_short_valid_track_remains_eligible_and_is_immutable(self):
        policy=ValidBirthPrefixRelease()
        track=self.track(True)
        emission=policy.consider(track,{0:0.,1:5.},1,5.)
        self.assertIsNotNone(emission)
        track['records'][0]['health']['ok']=False
        self.assertTrue(emission['track']['records'][0]['health']['ok'])

    def test_unknown_birth_health_is_not_silently_accepted(self):
        track=self.track(True)
        track['records'][0].pop('health')
        with self.assertRaises(ValueError):birth_usable(track)


if __name__=='__main__': unittest.main()
