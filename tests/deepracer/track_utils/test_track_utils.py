from deepracer.tracks import GeometryUtils

class TestGeometryUtils:

    def test_angle_zero(self):
        assert 0.0 == GeometryUtils.get_angle([0, 0], [1, 1], [0, 0])

    def test_angle_ninety(self):
        assert 90.0 == GeometryUtils.get_angle([0, 0], [1, 1], [2, 0])

    def test_angle_one_eighty(self):
        assert 180.0 == GeometryUtils.get_angle([0, 0], [1, 1], [2, 2])
