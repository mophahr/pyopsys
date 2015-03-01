'''
	billiards module
	
	Copyright © 2015 Moritz Schönwetter
	
	This file is part of pyopsys.
	
	pyopsys is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	Foobar is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with pyopsys.  If not, see <http://www.gnu.org/licenses/>.
'''
from math import pi,sin,cos,sqrt,asin
import numpy as np
from __future__ import division, print_function
from scipy.optimize import fsolve

class billiard2D:


class stadium:
	def __init__(self, radius, length):
		self.radius=radius
		self.length=length

	def boundary_length(length, radius):
		return 2*pi*radius+2*length
	
	def cartesian_coordinates(length, radius, s):
	''' returns the x,y coordinates of a point on the boundary in a coordiante system centered on the lower straight'''
	    
		half_length=length/2
	
		#convert from relative s from [0,1[ to absolute s in[0,boundary_length[
		s=s*boundary_length(length,radius)
	
		arc_length=pi*radius
		if s<half_length:
			xy=[-s,0]
		elif s<half_length+arc_length:
			angle=(s-half_length)/radius
			xy=[-half_length-radius*sin(angle),radius*(1-cos(angle))]
	    elif s<3*half_length+arc_length:
			xy=[-half_length+(s-half_length-arc_length),2*radius]
		elif s<3*half_length+2*arc_length:
			angle=(s-3*half_length-arc_length)/radius
			xy=[half_length+radius*sin(angle),radius*(1+cos(angle))]
		else:
			xy=[boundary_length(length,radius)-s,0]
		return xy
	
	def normal_vector_s(length, radius, s):
	''' returns the normal vector pointing inside the billiard at a point on the boundary given by s'''
		[x,y]=cartesian_coordinates(length, radius, s)
		return normal_vector(length, radius, x, y)
	
	def normalised_vector(v):
	''' returns the unit vector to the provided vector'''
		norm=np.linalg.norm(v)
		if norm==0: 
			return v
		return v/norm
	
	def normal_vector(length, radius, x, y):
	''' returns the normal vector pointing inside the billiard at a point on the boundary given by (x,y)'''
	
		half_length=length/2
	
		if -half_length<=x<=half_length:
			if y>0:
				nv=[0,-1]
			else:
				nv=[0,1]
		elif x<-half_length:
			nv=[-half_length-x,radius-y]
			nv=normalised_vector(nv)
		else:
			nv=[half_length-x,radius-y]
			nv=normalised_vector(nv)
		return np.array(nv)
	
	def reflect(length, radius, incoming_vector, reflection_point):
	''' returns the angle of incident to the normal and a unit vector of the reflection-image of incoming_vector,
		reflected at (x,y) of the reflection_point.'''
		#https://en.wikipedia.org/wiki/Specular_reflection#Direction_of_reflection
		#normal vector pointing towards the inside of stadium:
		nv=normal_vector(length, radius, reflection_point[0], reflection_point[1])
		
		projection_on_normal=incoming_vector.dot(nv)
		
		outgoing_vector=-2*projection_on_normal*nv+incoming_vector
		
		cross_product=np.cross(incoming_vector,-nv)
		angle_of_incident=asin(cross_product)
		if angle_of_incident<0: angle_of_incident=2*pi+angle_of_incident
		
		return angle_of_incident, normalised_vector(outgoing_vector)
	
	def next_intersection_point(length, radius, outgoing_vector, reflection_point):
	    
		#first, check trvial cases. i.e. BB-orbits:
		if outgoing_vector[0]==0 and outgoing_vector[1]<0:
			return [reflection_point[0],0.]
		elif outgoing_vector[0]==0 and outgoing_vector[1]>0:
			return [reflection_point[0],2*radius]
		if reflection_point[1]==radius and outgoing_vector[1]==0:
			return [-reflection_point[0],reflection_point[1]]
		
		def ray(t):
		''' ray going from the reflection point in the direction of outgoing_vector parametrised by t '''
			assert t>0., 'Error: ray is going backwards or stuck.' 
			return reflection_point+t*outgoing_vector
		def ray_y_down(t):
			return ray(t)[1]
		def ray_y_up(t):
			return ray(t)[1]-2*radius
		
		#find intersections with either upper or lower extended boundary to decide what method to use to find next intersection point:
		if outgoing_vector[1]<0.:
			t_intersection=fsolve(ray_y_down,1)
			intersection_point=[ray(t_intersection)[0],0.]
		else:
			t_intersection=fsolve(ray_y_up,1)
			intersection_point=[ray(t_intersection)[0],2*radius]
		    
		if intersection_point[0]<-length/2:
			#we are on the left half circle.
			left_center=np.array([-length/2,radius])
			def passing_left_circle(t):
				return np.linalg.norm(-left_center+ray(t))-radius
			#something lthat lies outside the billard, so that we dont intersect with the 'inner' half circle:
			large_initial_guess = 2*length+2*radius
			t_intersection=fsolve(passing_left_circle,large_initial_guess)
			intersection_point=ray(t_intersection)
			assert intersection_point[0]<-length/2, 'Error: Intersection point is not on boundary.'
		elif intersection_point[0]>length/2:
			#we are on the right half circle.
			right_center=np.array([length/2,radius])
			def passing_right_circle(t):
				return np.linalg.norm(-right_center+ray(t))-radius
			#something lthat lies outside the billard, so that we dont intersect with the 'inner' half circle:
			large_initial_guess = 2*length+2*radius
			t_intersection=fsolve(passing_right_circle,large_initial_guess)
			intersection_point=ray(t_intersection)
			assert intersection_point[0]>length/2, 'Error: Intersection point is not on boundary.'
		
		else:
			#we hit a straight boundary and we can use the intersection_point:
			pass
		
		return intersection_point
