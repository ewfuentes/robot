import unittest
from reservoir_canonicalization import canonical_tags, canonical_observation


class ReservoirCanonicalizationTest(unittest.TestCase):
    def test_old_and_new_tags_have_identical_canonical_form(self):
        old = {'landuse':'reservoir','name':'Example'}
        new = {'natural':'water','water':'reservoir','name':'Example'}
        canonical,rules = canonical_tags(old)
        self.assertEqual(canonical,new)
        self.assertTrue(rules)
        self.assertEqual(canonical_tags(canonical)[0],canonical)
        self.assertEqual(old,{'landuse':'reservoir','name':'Example'})

    def test_covered_and_conflicting_features_get_no_inference(self):
        for guard in [{'covered':'yes'},{'man_made':'reservoir_covered'},
                      {'man_made':'storage_tank'},{'location':'underground'},
                      {'natural':'wood'},{'water':'pond'}]:
            raw = {'landuse':'reservoir',**guard}
            self.assertEqual(canonical_tags(raw),(raw,[]))

    def test_prefixed_lifecycle_is_not_reactivated(self):
        raw = {'disused:landuse':'reservoir','covered':'yes'}
        self.assertEqual(canonical_tags(raw),(raw,[]))

    def test_query_alias_retains_confidence_without_obsolete_constraint(self):
        audit = {'primary_object':{'tags':[{'tag':'landuse=reservoir','weight':.9}]}}
        result,rules = canonical_observation(audit,.8)
        self.assertTrue(rules)
        self.assertEqual({x['tag']:x['weight'] for x in result['primary_object']['tags']},
                         {'natural=water':.9,'water=reservoir':.9})
        self.assertEqual(audit['primary_object']['tags'][0]['tag'],'landuse=reservoir')

    def test_ambiguous_query_is_not_forced_to_reservoir(self):
        audit = {'primary_object':{'tags':[{'tag':t,'weight':.9} for t in
                  ['landuse=reservoir','natural=water','natural=wood']]}}
        self.assertEqual(canonical_observation(audit,.8),(audit,[]))

    def test_matching_keeps_covered_guard_after_real_pruning(self):
        from experimental.overhead_matching.swag.farfield.catalog.catalog import prune_far_field_tags
        from water_compatibility_v2 import water_entries
        raw = {'open_old':{'landuse':'reservoir'},
               'open_new':{'natural':'water','water':'reservoir'},
               'covered':{'landuse':'reservoir','covered':'yes'},
               'conflicting':{'landuse':'reservoir','natural':'wood'}}
        signatures = {lid:dict(canonical_tags=prune_far_field_tags(canonical_tags(tags)[0]),
                               landmark_ids=[lid]) for lid,tags in raw.items()}
        old_query = {'primary_object':{'tags':[dict(tag='landuse=reservoir',weight=.9)]}}
        new_query = {'primary_object':{'tags':[dict(tag=t,weight=.9) for t in
                                               ['natural=water','water=reservoir']]}}
        entries = water_entries(canonical_observation(old_query,.8)[0],signatures)
        self.assertEqual(entries,water_entries(new_query,signatures))
        self.assertEqual({e['landmark_id'] for e in entries},{'open_old','open_new'})


if __name__=='__main__':
    unittest.main()
