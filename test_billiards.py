# -*- coding: utf-8 -*-
from __future__ import division, print_function
'''
	tests for billiards module
	
	Copyright © 2015 Moritz Schönwetter
	
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
from math import sqrt
import billiards

EPSILON=1.e-15

def test_stadium_init():
	radius=2.3
	length=4.5
	teststadium=billiards.stadium(radius, length)
	assert_equals(teststadium.radius,radius)
	assert_equals(teststadium.length,length)

def test_stadium_cartesian_coordinates():
	radius=2
	length=4
	teststadium=billiards.stadium(radius, length)
	
	s_number=200
	s_values=[i/(s_number-1) for i in range(s_number)]

	for s in s_values:
		[x,y]=teststadium.cartesian_coordinates(s)

		if x>-length/2 and x<length/2:
			assert_true( (y==0 or abs(y-2*radius)<EPSILON) )
		elif x<=-length/2:
			distance_to_left_center=sqrt((x+length/2)**2+(y-radius)**2)
			assert_true( abs(distance_to_left_center-radius)<=EPSILON )
		else:
			distance_to_right_center=sqrt((x-length/2)**2+(y-radius)**2)
			assert_true( abs(distance_to_right_center-radius)<=EPSILON )


