# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
    dimension_algorithms module

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
import numpy as np
from math import log

# ============================================================================
# helper functions
# ============================================================================
def raster(points, epsilon, dim=1, norm=[1]):
    """
    returns a "dim"-dimensional array of zeros (empty) and ones (full) 
    indicating which boxes contain at least one element of "points".
    """
    if dim == 1:
        filled_boxes_size = int(norm[0] / epsilon)
        if filled_boxes_size * epsilon < norm[0]:
            filled_boxes_size += 1
        filled_boxes = np.zeros(filled_boxes_size)
        for point in points:
            index = int(point / (norm[0] * epsilon))
            filled_boxes[index] = 1
    else:
        filled_boxes_size = [int(norm[i] / epsilon[i]) for i in range(dim)]
        for i in range(dim):
            if filled_boxes_size[i] * epsilon[i] < norm[i]:
                filled_boxes_size[i] += 1
        filled_boxes = np.zeros(filled_boxes_size)

        for point in points:
            index = [int(point[i] / (norm[i] * epsilon[i])) for i in range(dim)]
            filled_boxes[index] = 1
    return epsilon, filled_boxes

# ============================================================================
# algorithms acting upon existing sets:
# ============================================================================

def box_counting_1D(points, epsilons = np.logspace(-5,-1,10), norm = 1.):
    """
    points:        list of 1d coordinates making up the set under consideration
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)

    most basic version of box-counting.
    It calculates the number of "epsilons"-sized boxes intesecting a set
    of points.
    
    returns epsilons, n_filled, used_epsilons, dimensions
    """

    n_filled=[] 
    
    #box-counting:
    for e_idx, e in enumerate(epsilons):
        n_boxes = int(1 / e)
        if n_boxes * e < norm:
            n_boxes += 1
        new_filled_boxes = np.zeros(n_boxes)
        
        for x_idx, x in enumerate(points):
            # mark boxes that contain points 
            box_counting_index = int(x / e)
            new_filled_boxes[box_counting_index] = 1
        
        n_filled += [sum(new_filled_boxes)]
        filled_boxes = new_filled_boxes[:]
    
    dimensions=[]
    used_epsilons=[]
    for e_idx,e in enumerate(epsilons[:-1]):
        if n_filled[e_idx] > 0 and n_filled[e_idx+1] > 0:
            used_epsilons += [e]
            dimensions += [-(log(n_filled[e_idx])- log(n_filled[e_idx+1])) / (log(e) - log(epsilons[e_idx + 1]))]
    
    return epsilons, n_filled, used_epsilons, dimensions

def box_counting_1D_raster(boxes, epsilons = np.logspace(-5,-1,10), norm = 1.):
    """
    boxes:       list indicating which box of size norm / len(boxes) is full (1) or empty (0)
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)

    version of box-counting for when the set is given as a list of boxes.
    
    returns epsilons, n_filled, used_epsilons, dimensions
    """
    input_box_size = norm / len(boxes)
    
    #find smallest useful epsilon and check which boxes are filled:
    for e_idx, e in enumerate(epsilons):
        if e >= input_box_size:
            n_boxes = int(1 / e)
            if n_boxes * e < norm:
                n_boxes += 1
            filled_boxes = np.zeros(n_boxes)
            start_index = e_idx
            for filled_idx, filled in enumerate(boxes):
                if filled > 0:
                    # mark larger boxes that contain either of the ends of the box 
                    box_counting_index = int(filled_idx * input_box_size / e)
                    filled_boxes[box_counting_index] = 1
                    if not filled_idx == len(boxes)-1:
                        box_counting_index = int((filled_idx + .99) * input_box_size / e)
                        filled_boxes[box_counting_index] = 1
            break
    
    n_filled=[sum(filled_boxes)] 
    
    #box-counting:
    for e_idx, e in enumerate(epsilons[start_index+1:]):
        n_boxes = int(1 / e)
        if n_boxes * e < norm:
            n_boxes += 1
        new_filled_boxes = np.zeros(n_boxes)
        
        for filled_idx, filled in enumerate(filled_boxes):
            if filled > 0:
                # mark larger boxes that contain either of the ends of the box 
                box_counting_index = int(filled_idx * epsilons[e_idx+start_index] / e)
                new_filled_boxes[box_counting_index] = 1
                if not filled_idx == len(filled_boxes)-1:
                    box_counting_index = int((filled_idx + .99) * epsilons[e_idx+start_index] / e)
                    new_filled_boxes[box_counting_index] = 1
        
        n_filled += [sum(new_filled_boxes)]
        filled_boxes = new_filled_boxes[:]
    
    dimensions=[]
    used_epsilons=[]
    for e_idx,e in enumerate(epsilons[start_index:-1]):
        if n_filled[e_idx] > 0 and n_filled[e_idx+1] > 0:
            used_epsilons += [e]
            dimensions += [-(log(n_filled[e_idx])- log(n_filled[e_idx+1])) / (log(e) - log(epsilons[e_idx + start_index + 1]))]
    
    return epsilons[start_index:], n_filled, used_epsilons, dimensions

def grassberger_procaccia_1D(points, n_samples, epsilons = np.logspace(-5,-1,10), norm = 1.):
    """
    points:      list of 1d coordinates making up the set under consideration
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)
    
    grassberger procaccia algorithm to calculate the correlation dimension.
    
    returns epsilons, n_filled, used_epsilons, dimensions
    """
    
    n_found = []
    n_points = len(points)
    for e in epsilons:
        close_points = 0
        set_size = 0
        for x_idx,x in enumerate(points):
            for j in xrange(1, n_points - x_idx):
                if abs(x - points[x_idx + j]) <= e:
                    close_points += 1
                else:
                    break
        n_found += [(close_points * 2 / (n_points * (n_points - 1)))]
    
    dimensions = []
    used_epsilons = []
    for e_idx, e in enumerate(epsilons[:-1]):
        if n_found[e_idx] > 0 and n_found[e_idx+1] > 0:
            used_epsilons += [e]
            dimensions += [(log(n_found[e_idx]) - log(n_found[e_idx + 1])) / (log(e) - log(epsilons[e_idx + 1]))]
    
    return epsilons, n_found, used_epsilons, dimensions
