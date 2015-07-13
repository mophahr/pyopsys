# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
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
"""
from nose.tools import assert_equal, assert_true, assert_almost_equal
import maps

def test_standard_map_init():
    teststamap = maps.StandardMap(.3)
    teststamap.init_random_number
    teststamap.random_vector()
    teststamap.random_vector(limits=[[2, 3], [3, 4]])


def test_standard_map_mapping():
    pass

def test_tent_map():
	testmap = maps.TentMap(.2,1.2)
	# check special points:
	assert_equal(testmap.mapping(0),0)
	assert_equal(testmap.mapping(1),0)
	assert_equal(testmap.mapping(.2),1.2/2)

def test_tent_map_escape():
	testmap = maps.TentMap(.2,2.2)
	#check if escapetimes are correct:
	assert_equal(testmap.time_until_hole(.2)[0], 1)
	assert_equal(testmap.time_until_hole(2)[0], 0)
	assert_equal(testmap.time_until_hole(.2, use_real_time = True)[0], 2.2/2-.2)
	mt=100000
	assert_equal(testmap.time_until_hole(.0, max_iterations = mt)[0], mt)
	assert_equal(testmap.time_until_hole(1.0, max_iterations = mt)[0], mt)

	
