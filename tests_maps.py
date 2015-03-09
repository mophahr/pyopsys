# -*- coding: utf-8 -*-
from __future__ import division, print_function
'''
    tests for maps module
    
    Copyright © 2012-2015 Moritz Schönwetter
    
    This file is part of pyopsys.
    
    pyopsys is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    pyopsys is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with pyopsys.  If not, see <http://www.gnu.org/licenses/>.
'''
from nose.tools import assert_equals, assert_true
from math import sqrt,pi,cos
import numpy as np
import maps

# EPSILON is the maximum distance between two floats that they are still considered equal:
EPSILON=1.e-15

def test_standard_map_init():
    teststamap=maps.standard_map( .3 )
    print(teststamap.init_random_number)
    print(teststamap.random_vector())
    print(teststamap.random_vector(limits=[[2,3],[3,4]]))

def test_standard_map_mapping():
    pass

