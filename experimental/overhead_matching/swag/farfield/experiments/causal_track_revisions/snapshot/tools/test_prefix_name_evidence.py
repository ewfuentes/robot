import copy
import unittest
from prefix_name_evidence import add_prefix_names


class PrefixNameTest(unittest.TestCase):
    def setUp(self):
        self.event=dict(release_keyframe=2,semantic_evidence=dict(provenance=dict(
            supported_frame_count_including_birth=2,observations=[dict(obs_id='a',frame=0,frame_vote_share=1.)])),
            table=dict(kind='CompatibilityTable',tracklet_id='T1',matcher_version='base',
                entries=[dict(kind='CompatibilityEntry',landmark_id='other',log_lr=-2.)],
                default_log_lr=-12.,clip_lo=-12.,clip_hi=4.,status='fast'))
        self.obs={'a':dict(frame_idx=0,confidence='high',additional_tags=[['name','The Blue Lake']])}

    def test_names_are_soft_and_diluted_by_frame_support(self):
        table,info=add_prefix_names(self.event,self.obs,{'blue lake':{'named'}})
        self.assertEqual(info['supported_landmark_weights'],{'named':.4})
        self.assertEqual({e['landmark_id'] for e in table['entries']},{'other','named'})
        self.assertEqual(self.event['table']['matcher_version'],'base')

    def test_unknown_low_confidence_and_future_names(self):
        table,_=add_prefix_names(self.event,self.obs,{})
        self.assertEqual(table,self.event['table'])
        self.obs['a']['confidence']='medium'
        table,_=add_prefix_names(self.event,self.obs,{'blue lake':{'named'}})
        self.assertEqual(table,self.event['table'])
        self.obs['a']['frame_idx']=3
        with self.assertRaises(ValueError):add_prefix_names(self.event,self.obs,{})

    def test_unreferenced_future_observation_cannot_change_table(self):
        expected=add_prefix_names(self.event,self.obs,{'blue lake':{'named'}})
        self.obs['future']=dict(frame_idx=99,confidence='high',additional_tags=[['name','Wrong Lake']])
        self.assertEqual(add_prefix_names(self.event,self.obs,{'blue lake':{'named'}}),expected)


if __name__=='__main__': unittest.main()
