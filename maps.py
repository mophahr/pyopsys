# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
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
"""
from math import pi, sin
import random

import numpy as np

random = random.SystemRandom()


class Map:
    def __init__(self, dimension):
        self.dimension = dimension
        # create a random number to check st
        self.init_random_number = random.random()
        self.default_limits = [[0, 1] for _ in range(self.dimension)]

    def random_vector(self, limits=None):
        limits = limits or self.default_limits
        return np.array([random.uniform(limit[0], limit[1]) for limit in limits])


class StandardMap(Map):
	def __init__(self, nonlinearity_parameter):
		Map.__init__(self, dimension=2)
		self.nonlinearity_parameter = nonlinearity_parameter

	def mapping(self, r):
		"""apply the standard map (M: r->r_next=Mr)"""
		x = r[0]
		p = r[1]
		
		p_next = (p + self.nonlinearity_parameter * sin(2.0 * pi * x)) % 1
		x_next = (x + p_next) % 1
		
		return np.array([x_next, p_next])

	def time_until_hole(self,x,p,hole, use_real_time=False,max_iterations=1000):
		"""
		hole : [x_min, x_max, p_min, p_max] -- a point is in the hole when (s|p)_min <= (s|p) < (s|p)_max
		Returns the number of collisions or the trajectory length until the orbit reaches the hole.
		It also returns the image of the endpoint inside the hole.
		"""
		hole_hit=False
		iterations = 0
	
		current_location = np.array([x,p])

		if use_real_time:
			distance = 0.0
	
		for _ in range(max_iterations):
			#check if we're in the hole:
			if hole[0]<= current_location[0] <hole[1] and hole[2]<= current_location[1] <hole[3]:
				hole_hit=True

			#iterate:
			if use_real_time:
				old_location = current_location
			current_location = self.mapping(current_location)
	
			if hole_hit:
				#return stuff
				if use_real_time:
					return distance, current_location 
				else:
					return iterations, current_location
			else:
				iterations = iterations + 1
				if use_real_time:
					distance = distance + sqrt((current_location[0]-old_location[0]) ** 2 + (current_location[1]-old_location[1]) ** 2)

		if use_real_time:
			return distance, current_location 
		else:
			return iterations, current_location

	def number_of_hole_collisions(self,x,p,hole,max_iterations=1000):
		"""
		returns the number of collisions with the hole
		"""
		collisions = 0
		total_iterations = 0

		current_location = [x,p]

		while total_iterations<max_iterations and collisions<max_iterations:
			length, current_location = self.time_until_hole(current_location[0],current_location[1],hole,max_iterations=max_iterations)
			total_iterations = total_iterations + length +1
			if total_iterations<max_iterations:
				collisions = collisions + 1

		return collisions




