import unittest
from experimental.overhead_matching.swag.farfield.calibration.online_course import OnlineCourse


class OnlineCourseTest(unittest.TestCase):
    def test_stationary_tail_waits_for_explicit_eof(self):
        producer = OnlineCourse(1)
        emitted = []
        for fix in [(0., 0., 0.), (1., 4., 0.), (2., 8., 0.), (3., 8., 0.)]:
            emitted.extend(producer.append(*fix))
        self.assertTrue(all(row['source_keyframe'] < 2 for row in emitted))
        tail = producer.finish()
        self.assertEqual([row['source_keyframe'] for row in tail], [2, 3])
        self.assertTrue(all(row['requires_eof'] for row in tail))
        with self.assertRaises(ValueError):
            producer.append(4., 12., 0.)

    def test_future_turn_does_not_revise_emitted_values(self):
        east = OnlineCourse(3)
        north = OnlineCourse(3)
        a, b = [], []
        for k in range(6):
            a.extend(east.append(float(k), 4. * k, 0.))
            b.extend(north.append(float(k), 4. * k, 0.))
        self.assertTrue(a)
        frozen = [dict(row) for row in a]
        east.append(6., 24., 0.)
        north.append(6., 20., 4.)
        self.assertEqual(a, frozen)
        self.assertEqual(a, b)
        self.assertTrue(all(row['available_keyframe'] <= 5 for row in a))


if __name__ == '__main__':
    unittest.main()
