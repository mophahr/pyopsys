# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
    billiards module
    
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
from math import pi, sin, cos, asin, acos, sqrt
import numpy as np
from scipy.optimize import fsolve


class Billiard:
    def __init__(self, dimension):
        self.dimension = dimension

    @staticmethod
    def normalised_vector(v):
        """
        Returns the unit vector to the provided vector.
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        else:
            return v / norm


class Stadium(Billiard):
	def __init__(self, radius, length):
		Billiard.__init__(self, dimension=2)
		self.radius = radius
		self.length = length
		self.boundary_length = 2 * pi * self.radius + 2 * self.length

	def cartesian_coordinates(self, s):
		"""
		Returns the x,y coordinates of a point on the boundary in a coordinate
		system centered on the lower straight s is the normalised coordinate
		around the billiard's boundary in clockwise direction.
		"""
		half_length = self.length / 2

		# convert from relative s from [0,1[ to absolute s in[0,boundary_length[
		s_not_normalised = s * self.boundary_length

		arc_length = pi * self.radius
		if s_not_normalised < half_length:
			xy = [-s_not_normalised, 0]
		elif s_not_normalised < half_length + arc_length:
			angle = (s_not_normalised - half_length) / self.radius
			xy = [-half_length - self.radius * sin(angle),
		          self.radius * (1 - cos(angle))]
		elif s_not_normalised < 3 * half_length + arc_length:
			xy = [-half_length + (s_not_normalised - half_length - arc_length),
				2 * self.radius]
		elif s_not_normalised < 3 * half_length + 2 * arc_length:
			angle = (s_not_normalised - 3 * half_length - arc_length) / self.radius
			xy = [half_length + self.radius * sin(angle),
			self.radius * (1 + cos(angle))]
		else:
			xy = [self.boundary_length - s_not_normalised, 0]
		return xy

	def cartesian_to_s(self, cartesian):
		x = cartesian[0]
		y = cartesian[1]

		half_length = self.length / 2

		if -half_length <= x <= half_length:
			if y < self.radius:
				#lower straight:
				if x <= 0.:
					s_not_normalised = -x
				else:
					s_not_normalised = self.boundary_length - x
			else:
				#upper straight:
				offset = half_length + pi * self.radius
				s_not_normalised = offset + (half_length + x)
		elif x < -half_length:
			#left half circle
			offset = half_length
			s_not_normalised = offset + acos((self.radius - y) / self.radius) * self.radius
		else:
			#right half circle
			offset = pi * self.radius + 2 * self.length
			s_not_normalised = offset + acos((y - self.radius) / self.radius) * self.radius

		return s_not_normalised/self.boundary_length

	def s_theta_to_vector(self, s, theta):
		"""
		Returns an unit vector in direction of a ray with billiard-coordinates
		(s,theta) theta is measured from the incoming ray to the normal vector
		and from the normal vector to the outgoing ray we use that
		nv.dot(outgoing_vector)==cos(theta) and
		nv.cross(outgoing_vector)==sin(theta).
		"""
		[st, ct] = [sin(theta), cos(theta)]
		[nx, ny] = self.normal_vector_s(s)
		outgoing_vector = [nx * ct - ny * st, ny * ct + nx * st]
		return np.array(outgoing_vector)

	def normal_vector_s(self, s):
		"""
		Returns the normal vector pointing inside the billiard at a point on the
		boundary given by s.
		"""
		[x, y] = self.cartesian_coordinates(s)
		return self.normal_vector(x, y)

	def normal_vector(self, x, y):
		"""
		Returns the normal vector pointing inside the billiard at a point on the
		boundary given by (x,y).
		"""
		
		half_length = self.length / 2
		
		if -half_length <= x <= half_length:
			if y > 0:
				nv = [0, -1]
			else:
				nv = [0, 1]
		elif x < -half_length:
			nv = [-half_length - x, self.radius - y]
			nv = self.normalised_vector(nv)
		else:
			nv = [half_length - x, self.radius - y]
			nv = self.normalised_vector(nv)
		return np.array(nv)

	def reflect(self, incoming_vector, reflection_point):
		"""
		Returns the angle of incident to the normal and a unit vector of the
		reflection-image of incoming_vector, reflected at (x,y) of the
		reflection_point.
		
		See also
		https://en.wikipedia.org/wiki/Specular_reflection#Direction_of_reflection
		"""
		# normal vector pointing towards the inside of stadium:
		nv = self.normal_vector(reflection_point[0], reflection_point[1])
		
		projection_on_normal = incoming_vector.dot(nv)
		
		outgoing_vector = -2 * projection_on_normal * nv + incoming_vector
		
		cross_product = np.cross(incoming_vector, -nv)
		angle_of_incident = asin(cross_product)
		if angle_of_incident < 0:
			angle_of_incident += 2 * pi
		
		return angle_of_incident, self.normalised_vector(outgoing_vector)
	
	def next_intersection_point(self, outgoing_vector, reflection_point):
		# first, check trvial cases. i.e. orthogonal BB-orbits:
		if outgoing_vector[0] == 0 and outgoing_vector[1] < 0:
			return [reflection_point[0], 0.]
		elif outgoing_vector[0] == 0 and outgoing_vector[1] > 0:
			return [reflection_point[0], 2 * self.radius]
		if reflection_point[1] == self.radius and outgoing_vector[1] == 0:
			return [-reflection_point[0], reflection_point[1]]
		
		# bring it to easier form (y == m * x + c):
		x0 = reflection_point[0]
		y0 = reflection_point[1]
		m  = outgoing_vector[1]/outgoing_vector[0] 
		c  = y0 - m * x0
		r  = self.radius
		l  = self.length

		# find intersections with either upper or lower extended boundary to
		# decide what method to use to find the next intersection point:
		if outgoing_vector[1] < 0.:
			# we point downwards
			x1 = -c / m
			y1 = 0
		else:
			x1 = (2 * r - c) / m
			y1 = 2 * r
		
		# now we put in the two analytical solutions from solving
		# y == m * x +c
		# (see [https://de.wikipedia.org/wiki/Gerade#Punkt-Richtungs-Gleichung#Bestimmung%20der%20Gleichung%20einer%20Geraden%20in%20der%20Ebene])
		# and
		# (x \pm l/2)^2 + (y - r)^2 == r^2
		# for (x,y) and pick the one that's not the initial point and thats not inside the stadium

		if x1 < -self.length / 2:
			# we are on the left half circle.

			#possible intersections:
			xp1 = -(l + 2*c*m - 2*m*r + sqrt(-4*c**2 + 4*c*(l*m + 2*r) - m*(l**2*m + 4*l*r - 4*m*r**2)))/(2*(1 + m**2))
			yp1 = (2*c + m*(-l + 2*m*r - sqrt(-4*c**2 + 4*c*(l*m + 2*r) - m*(l**2*m + 4*l*r - 4*m*r**2))))/(2*(1 + m**2))
			xp2 = (-l - 2*c*m + 2*m*r + sqrt(-4*c**2 + 4*c*(l*m + 2*r) - m*(l**2*m + 4*l*r - 4*m*r**2)))/(2*(1 + m**2))
			yp2 = (2*c + m*(-l + 2*m*r + sqrt(-4*c**2 + 4*c*(l*m + 2*r) - m*(l**2*m + 4*l*r - 4*m*r**2))))/(2*(1 + m**2))

			if xp1 < -l/2 and xp2 < -l/2:
				#both points on the left round boundary ==> use the point further away from the initial point:
				d1 = (xp1 - x0)**2 + (yp1 - y0)**2
				d2 = (xp2 - x0)**2 + (yp2 - y0)**2

				if d1>=d2:
					intersection_point = [xp1,yp1]
				else:
					intersection_point = [xp2,yp2]

			elif xp1 < -l/2 and xp2 >= -l/2:
				intersection_point = [xp1,yp1]

			elif xp1 >= -l/2 and xp2 < -l/2:
				intersection_point = [xp2,yp2]

		elif x1 > self.length / 2:
			# we are on the right half circle.

			#possible intersections:
			xp1 = (l - 2*c*m + 2*m*r - sqrt(-4*c**2 - 4*c*l*m - l**2*m**2 + 8*c*r + 4*l*m*r + 4*m**2*r**2))/(2 + 2*m**2)
			yp1 = (2*c + m*(l + 2*m*r - sqrt(-4*c**2 + c*(-4*l*m + 8*r) + m*(-(l**2*m) + 4*l*r + 4*m*r**2))))/(2*(1 + m**2))
			xp2 = (l - 2*c*m + 2*m*r + sqrt(-4*c**2 - 4*c*l*m - l**2*m**2 + 8*c*r + 4*l*m*r + 4*m**2*r**2))/(2 + 2*m**2)
			yp2 = (2*c + m*(l + 2*m*r + sqrt(-4*c**2 + c*(-4*l*m + 8*r) + m*(-(l**2*m) + 4*l*r + 4*m*r**2))))/(2*(1 + m**2))

			if xp1 > l/2 and xp2 > l/2:
				#both points on the right round boundary ==> use the point further away from the initial point:
				d1 = (xp1 - x0)**2 + (yp1 - y0)**2
				d2 = (xp2 - x0)**2 + (yp2 - y0)**2

				if d1>=d2:
					intersection_point = [xp1,yp1]
				else:
					intersection_point = [xp2,yp2]

			elif xp1 > l/2 and xp2 <= l/2:
				intersection_point = [xp1,yp1]

			elif xp1 <= l/2 and xp2 > l/2:
				intersection_point = [xp2,yp2]

		else:
			# we hit a straight boundary and we can use the intersection point from earlier in this function:
			intersection_point = [x1,y1]
		
		return intersection_point
	
	def time_until_hole(self,s,theta,hole, use_real_time=False,max_iterations=1000):
		"""
		hole : [s_min, s_max, theta_min, theta_max] -- a point is in the hole when (s|theta)_min <= (s|theta) < (s|theta)_max
		Returns the number of collisions or the trajectory length until the orbit reaches the hole.
		It also returns the image of the endpoint inside the hole.
		"""
		hole_hit=False
		boundary_collisions = 0
	
		if use_real_time:
			distance = 0.0
	
		current_vector = self.s_theta_to_vector(s, theta)
		current_location = self.cartesian_coordinates( s )
	
		for _ in range(max_iterations):
			#check if we're in the hole:
			if hole[0]<= s <hole[1] and hole[2]<= theta <hole[3]:
				hole_hit=True
	
			#iterate:
			if use_real_time:
				old_location = current_location
			current_location = self.next_intersection_point( current_vector, current_location)
			theta, current_vector = self.reflect(current_vector, current_location )
			s = self.cartesian_to_s(current_location)

			if hole_hit:
				#return stuff
				if use_real_time:
					return distance, (s,theta)
				else:
					return boundary_collisions, (s,theta)
			else:
				boundary_collisions = boundary_collisions + 1
				if use_real_time:
					distance = distance + sqrt((current_location[0]-old_location[0]) ** 2 + (current_location[1]-old_location[1]) ** 2)

		if use_real_time:
			return distance, current_location 
		else:
			return boundary_collisions, current_location

	def number_of_hole_collisions(self,s,theta,hole,max_iterations=1000):
		"""
		returns the number of collisions with the hole
		"""
		collisions = 0
		total_iterations = 0

		current_location = [s,theta]

		while total_iterations<max_iterations and collisions<max_iterations:
			length, current_location = self.time_until_hole(current_location[0],current_location[1],hole,max_iterations=max_iterations)
			total_iterations = total_iterations + length +1
			if total_iterations<max_iterations:
				collisions = collisions + 1

		return collisions
