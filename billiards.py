# -*- coding: utf-8 -*-
from __future__ import division, print_function
'''
	billiards module
	
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
from math import pi,sin,cos,sqrt,asin
import numpy as np
from scipy.optimize import fsolve

class billiard:
	def __init__(self,dimension):
		self.dimension = dimension
		
	def normalised_vector(self,v):
		''' returns the unit vector to the provided vector'''
		norm = np.linalg.norm(v)
		if norm==0: 
			return v
		else: 
			return v/norm

class stadium(billiard):
	def __init__(self, radius, length):
		billiard.__init__(self,dimension = 2)
		self.radius = radius
		self.length = length
		self.boundary_length = 2*pi*self.radius + 2*self.length
	
	def cartesian_coordinates(self, s):
		''' returns the x,y coordinates of a point on the boundary in a coordiante system centered on the lower straight
			s is the normalised coordinate around the billiard's boundary in clockwise direction'''
	    
		half_length = self.length/2
	
		#convert from relative s from [0,1[ to absolute s in[0,boundary_length[
		s_not_normalised = s*self.boundary_length
	
		arc_length = pi*self.radius
		if s_not_normalised<half_length:
			xy = [-s_not_normalised,0]
		elif s_not_normalised<half_length + arc_length:
			angle = (s_not_normalised-half_length)/self.radius
			xy = [-half_length-self.radius*sin(angle),self.radius*(1-cos(angle))]
		elif s_not_normalised<3*half_length + arc_length:
			xy = [-half_length + (s_not_normalised-half_length-arc_length),2*self.radius]
		elif s_not_normalised<3*half_length + 2*arc_length:
			angle = (s_not_normalised-3*half_length-arc_length)/self.radius
			xy = [half_length + self.radius*sin(angle),self.radius*(1 + cos(angle))]
		else:
			xy = [self.boundary_length-s_not_normalised,0]
		return xy

	def s_theta_to_vector(self,s,theta):
		''' returns an unit vector in directoin of a ray with billiard-coordinates(s,theta)
			theta is measured from the incoming ray to the normal vector and from the normal vector to the outgoing ray
			we use that nv.dot(outgoing_vector)==cos(theta)
			and nv.cross(outgoing_vector)==sin(theta) and therefore:'''
		[st,ct] = [sin(theta),cos(theta)]
		[nx,ny] = self.normal_vector_s(s)
		outgoing_vector = [nx*ct-ny*st,ny*ct+nx*st]
		return np.array(outgoing_vector)

	def normal_vector_s(self, s):
		''' returns the normal vector pointing inside the billiard at a point on the boundary given by s'''
		[x,y] = self.cartesian_coordinates(s)
		return self.normal_vector(x, y)
	
	def normal_vector(self, x, y):
		''' returns the normal vector pointing inside the billiard at a point on the boundary given by (x,y)'''
	
		half_length = self.length/2
	
		if -half_length<=x<=half_length:
			if y>0:
				nv = [0,-1]
			else:
				nv = [0,1]
		elif x<-half_length:
			nv = [-half_length-x,self.radius-y]
			nv = self.normalised_vector(nv)
		else:
			nv = [half_length-x,self.radius-y]
			nv = self.normalised_vector(nv)
		return np.array(nv)
	
	def reflect(self, incoming_vector, reflection_point):
		''' returns the angle of incident to the normal and a unit vector of the reflection-image of incoming_vector,
			reflected at (x,y) of the reflection_point.'''
		#https://en.wikipedia.org/wiki/Specular_reflection#Direction_of_reflection
		#normal vector pointing towards the inside of stadium:
		nv = self.normal_vector(reflection_point[0], reflection_point[1])
		
		projection_on_normal = incoming_vector.dot(nv)
		
		outgoing_vector = -2*projection_on_normal*nv + incoming_vector
		
		cross_product = np.cross(incoming_vector,-nv)
		angle_of_incident = asin(cross_product)
		if angle_of_incident<0: angle_of_incident = 2*pi + angle_of_incident
		
		return angle_of_incident, self.normalised_vector(outgoing_vector)
	
	def next_intersection_point(self, outgoing_vector, reflection_point):
	    
		#first, check trvial cases. i.e. BB-orbits:
		if outgoing_vector[0]==0 and outgoing_vector[1]<0:
			return [reflection_point[0],0.]
		elif outgoing_vector[0]==0 and outgoing_vector[1]>0:
			return [reflection_point[0],2*self.radius]
		if reflection_point[1]==self.radius and outgoing_vector[1]==0:
			return [-reflection_point[0],reflection_point[1]]
		
		def ray(t):
			''' ray going from the reflection point in the direction of outgoing_vector parametrised by t '''
			assert t>0., 'Error: ray is going backwards or stuck.' 
			return reflection_point + t*outgoing_vector
		def ray_y_down(t):
			return ray(t)[1]
		def ray_y_up(t):
			return ray(t)[1]-2*self.radius
		
		#find intersections with either upper or lower extended boundary to decide what method to use
		#to find the next intersection point:
		if outgoing_vector[1]<0.:
			t_intersection = fsolve(ray_y_down,1)
			intersection_point = [ray(t_intersection)[0],0.]
		else:
			t_intersection = fsolve(ray_y_up,1)
			intersection_point = [ray(t_intersection)[0],2*self.radius]
		    
		if intersection_point[0]<-self.length/2:
			#we are on the left half circle.
			left_center = np.array([-self.length/2,self.radius])
			def passing_left_circle(t):
				return np.linalg.norm(-left_center + ray(t))-self.radius
			#something lthat lies outside the billard, so that we dont intersect with the 'inner' half circle:
			large_initial_guess = 2*self.length + 2*self.radius
			t_intersection = fsolve(passing_left_circle,large_initial_guess)
			intersection_point = ray(t_intersection)
			assert intersection_point[0]<-self.length/2, 'Error: Intersection point is not on boundary.'
		elif intersection_point[0]>self.length/2:
			#we are on the right half circle.
			right_center = np.array([self.length/2,self.radius])
			def passing_right_circle(t):
				return np.linalg.norm(-right_center + ray(t))-self.radius
			#something lthat lies outside the billard, so that we dont intersect with the 'inner' half circle:
			large_initial_guess = 2*self.length + 2*self.radius
			t_intersection = fsolve(passing_right_circle,large_initial_guess)
			intersection_point = ray(t_intersection)
			assert intersection_point[0]>self.length/2, 'Error: Intersection point is not on boundary.'
		
		else:
			#we hit a straight boundary and we can use the intersection_point:
			pass
		
		return intersection_point
