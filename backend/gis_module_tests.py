import pytest
from gis_module import *
import numpy as np

def test_geocode_address():
    washington_monument = geocode("Washington Monument")
    assert (washington_monument['coords'][0] > 38.88) and (washington_monument['coords'][0] < 38.89)
    assert (washington_monument['coords'][1] > -77.04) and (washington_monument['coords'][1] < -77.03)

def test_geocode_zip():
    hyde_park_zip = geocode("60615")
    assert (hyde_park_zip['coords'][0] > 41.80) and (hyde_park_zip['coords'][0] < 41.81)
    assert (hyde_park_zip['coords'][1] > -87.60) and (hyde_park_zip['coords'][1] < -87.5)

def test_distance_miles():
    washington_monument = geocode("Washington Monument")
    lib_con = geocode("Library of Congress")

    dist_bin = distance_bin(washington_monument['coords'], lib_con['coords'])
    assert np.isclose(dist_bin[0], 1.6513911006515658)
    assert np.isclose(dist_bin[1], 0.0)