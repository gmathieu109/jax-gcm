import unittest

class TestSpeedyPhysicsUnit(unittest.TestCase):
    def setUp(self):
        global PhysicsState, SpeedyPhysics, ForcingData, Parameters, TerrainData, DateData, coords, ix, il, kx
        ix, il, kx = 1, 1, 8
        from jcm.physics_interface import PhysicsState
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        from jcm.forcing import ForcingData
        from jcm.physics.speedy.params import Parameters
        from jcm.terrain_data import TerrainData
        from jcm.date import DateData
        from jcm.physics.speedy.utils import get_speedy_coords

        coords = get_speedy_coords(layers=kx, nodal_shape=(ix, il))

    def test_speedy_forcing(self):
        grid_shape = (8,1,1)
        tendencies, data = SpeedyPhysics(coords=coords).compute_tendencies(
            state=PhysicsState.zeros(grid_shape),
            forcing=ForcingData.ones(grid_shape[1:]),
            terrain=TerrainData.single_column(num_levels=grid_shape[0]), ## FIX ME
            date=DateData.zeros()
        )
        self.assertIsNotNone(tendencies)
        self.assertIsNotNone(data)