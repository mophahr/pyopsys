# -*- coding: utf-8 -*-
from __future__ import division, print_function

'''
    maps module

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
from math import pi, sin
import random

import numpy as np

random = random.SystemRandom()


class map:
    def __init__(self, dimension):
        self.dimension = dimension
        # create a random number to check st
        self.init_random_number = random.random()
        self.default_limits = [[0, 1] for _ in range(self.dimension)]

    def random_vector(self, limits=None):
        limits = limits or self.default_limits
        return np.array([random.uniform(limit[0], limit[1]) for limit in limits])


class standard_map(map):
    def __init__(self, nonlinearity_parameter):
        map.__init__(self, dimension=2)
        self.nonlinearity_parameter = nonlinearity_parameter

    def mapping(self, r):
        ''' apply the standard map (M: r->r_next=Mr)'''
        x = r[0]
        p = r[1]

        p_next = (p + self.nonlinearity_parameter * sin(2.0 * pi * x))
        x_next = (x + p_next) % 1

        return np.array([x_next, y_next])

