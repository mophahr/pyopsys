# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
    tests for billiards module
    
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
from nose.tools import assert_equals, assert_true
from math import sqrt, pi, cos
import billiards

# EPSILON is the maximum distance between two floats that they are still considered equal:
EPSILON = 1.e-15


def test_stadium_cartesian_coordinates():
    """Checks for if the output of cartesian_coordiates() is on the billiard boundary. """
    radius = 2
    length = 4
    teststadium = billiards.stadium(radius, length)

    s_number = 200
    s_values = [i / (s_number - 1) for i in range(s_number)]

    for s in s_values:
        [x, y] = teststadium.cartesian_coordinates(s)

        if -length / 2 < x < length / 2:
            assert_true((y == 0 or abs(y - 2 * radius) < EPSILON))
        elif x <= -length / 2:
            distance_to_left_center = sqrt(
                (x + length / 2) ** 2 + (y - radius) ** 2)
            assert_true(abs(distance_to_left_center - radius) <= EPSILON)
        else:
            distance_to_right_center = sqrt(
                (x - length / 2) ** 2 + (y - radius) ** 2)
            assert_true(abs(distance_to_right_center - radius) <= EPSILON)


def test_s_theta_to_vector():
    """A number of thests to check if the conversion from (s,theta) to (s,direction-vector) works."""
    radius = 2
    length = 4
    teststadium = billiards.stadium(radius, length)

    # 45° to the left of the unit vector:
    assert_true(
        abs(teststadium.s_theta_to_vector(0, pi / 4)[0] + 1 / sqrt(2.)) < EPSILON)
    assert_true(
        abs(teststadium.s_theta_to_vector(0, pi / 4)[1] - 1 / sqrt(2.)) < EPSILON)

    # 45° to the right of the unit vector:
    assert_true(
        abs(teststadium.s_theta_to_vector(0, -pi / 4)[0] - 1 / sqrt(2.)) < EPSILON)
    assert_true(
        abs(teststadium.s_theta_to_vector(0, -pi / 4)[1] - 1 / sqrt(2.)) < EPSILON)

    # check if negative values work as expected:
    assert_true(abs(teststadium.s_theta_to_vector(0, -pi / 4)[0] -
                    teststadium.s_theta_to_vector(0, 7 * pi / 4)[0]) < EPSILON)
    assert_true(abs(teststadium.s_theta_to_vector(0, -pi / 4)[1] -
                    teststadium.s_theta_to_vector(0, 7 * pi / 4)[1]) < EPSILON)


def test_reflection():
    """checks, if outgoing 'equals'(up to sign) incoming angle and if double-reflection results in the initial condition"""

    radius = 2
    length = 4
    teststadium = billiards.stadium(radius, length)

    # arbitrary point of reflection:
    s = 0.14
    reflection_point = teststadium.cartesian_coordinates(s)

    # arbitrary initial theta the 2*pi is here because reflect() expect an incoming ray:
    incoming_theta = 2 * pi - .3 * pi
    incoming = teststadium.s_theta_to_vector(s, incoming_theta)

    # reflect once and check if angle behaves correctly
    outgoing_theta, outgoing = teststadium.reflect(incoming, reflection_point)
    assert_true(abs(cos(incoming_theta) - cos(-outgoing_theta)) < EPSILON)

    # reflect again and check if we end up where we started:
    re_re_theta, re_re = teststadium.reflect(outgoing, reflection_point)
    assert_true(abs(re_re_theta - incoming_theta) < EPSILON)
    assert_true(abs(re_re[0] - incoming[0]) < EPSILON)
    assert_true(abs(re_re[1] - incoming[1]) < EPSILON)
