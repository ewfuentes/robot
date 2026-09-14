import copy
import unittest
from prefix_release_evidence import build_evidence
from prefix_release_policy import SUPPORT_CLASSES


class PrefixEvidenceTest(unittest.TestCase):
    def setUp(self):
        support=next(iter(SUPPORT_CLASSES))
        self.track=dict(track_id=1,birth_keyframe=0,birth_obs_id='o0',status='alive',records=[
            dict(keyframe=k,action='birth' if k==0 else 'continue',
                 mask_bbox_window=[0,0,10,10],window_origin=[170+k,0],
                 supports=[{'class':support,'obs_id':f'o{k}'}]) for k in range(3)])
        self.emission=dict(track=self.track,release_keyframe=2,available_time_s=10.)
        self.obs={f'o{k}':dict(frame_idx=k,primary_tag_key='natural',primary_tag_value='water',
                                additional_tags=[['distance_estimate','500m_to_2km']]) for k in range(3)}
        self.times={0:0.,1:5.,2:10.,3:15.}

    def build(self):
        return build_evidence(self.emission,self.obs,self.times,tracklet_id='prefix#T1',
                pano_width=360,mount_bearing_camera_cw_deg=0.,
                signatures={'water':dict(canonical_tags={'natural':'water'},landmark_ids=['lake'])})

    def test_live_prefix_has_geometry_semantics_and_no_close_requirement(self):
        out=self.build()
        self.assertEqual(len(out['measurements']),1)
        self.assertEqual(out['measurements'][0]['anchor_keyframe_idx'],1)
        self.assertEqual(out['measurements'][0]['range_max_m'],2000.)
        self.assertEqual([e['landmark_id'] for e in out['table']['entries']],['lake'])
        self.assertFalse(out['semantic_evidence']['provenance']['track_was_closed'])
        self.assertEqual(self.track['status'],'alive')

    def test_unreferenced_future_detections_cannot_change_result(self):
        expected=self.build()
        self.obs['future']=dict(frame_idx=3,primary_tag_key='natural',primary_tag_value='wood',additional_tags=[])
        self.assertEqual(self.build(),expected)

    def test_future_or_misjoined_support_is_rejected(self):
        self.obs['o1']['frame_idx']=3
        with self.assertRaises(ValueError): self.build()
        self.obs['o1']['frame_idx']=1
        self.track['records'].append(dict(keyframe=3))
        with self.assertRaises(ValueError): self.build()

    def test_unsupported_tail_is_excluded_without_using_future_audit(self):
        self.track['records'][2]['supports']=[]
        out=self.build()
        self.assertEqual(out['last_supported_keyframe'],1)
        self.assertEqual([x['keyframe_idx'] for x in out['camera_observations']],[0,1])


if __name__=='__main__': unittest.main()
