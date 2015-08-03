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
import uuid
import pickle

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

def box_counting_1d(points, n_samples, t = None, epsilons = np.logspace(-5,-1,10), norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = "", save_input_data = True):
    """
    points:      list of 1d coordinates making up the set under consideration
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)

    most basic version of box-counting.
    It calculates the number of "epsilons"-sized boxes intesecting a set
    of points.
    
    returns epsilons, n_filled, used_epsilons, dimensions (and the path+name of
                 the save file if save_comparison_data is set to True)
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
    
    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "box_counting_1d---"+"data_string"+"---n_samples__{}".format(n_samples)+"---"+identifier+".p"

        if not save_input_data:
            points = None

        stuff = {"data"          : points,
                 "epsilons"      : epsilons,
                 "data_string"   : data_string,
                 "norm"          : norm,
                 "uuid"          : identifier,
                 "n_samples"     : n_samples,
                 "n"             : n_filled,
                 "t"             : t,
                 "used_epsilons" : used_epsilons,
                 "dimensions"    : dimensions}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons, n_filled, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons, n_filled, used_epsilons, dimensions

def box_counting_1d_raster(boxes, t = None, n_samples=None, epsilons = np.logspace(-5,-1,10), norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = "", save_input_data = True):
    """
    boxes:       list indicating which box of size norm / len(boxes) is full (1) 
                 or empty (0)
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)

    version of box-counting for when the set is given as a list of boxes.
    
    returns epsilons, n_filled, used_epsilons, dimensions (and the path+name of
                 the save file if save_comparison_data is set to True)
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
                        box_counting_index = int((filled_idx + .999) * input_box_size / e)
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
                    box_counting_index = int((filled_idx + .999) * epsilons[e_idx+start_index] / e)
                    new_filled_boxes[box_counting_index] = 1
        
        n_filled += [sum(new_filled_boxes)]
        filled_boxes = new_filled_boxes[:]
    
    dimensions=[]
    used_epsilons=[]
    for e_idx,e in enumerate(epsilons[start_index:-1]):
        if n_filled[e_idx] > 0 and n_filled[e_idx+1] > 0:
            used_epsilons += [e]
            dimensions += [-(log(n_filled[e_idx])- log(n_filled[e_idx+1])) / (log(e) - log(epsilons[e_idx + start_index + 1]))]

    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "box_counting_1d_raster---"+"data_string"+"---n_samples__{}".format(n_samples)+"---"+identifier+".p"
        data_file_name = identifier+".npy"
        
        if not save_input_data:
            boxes = None

        stuff = {"data"         : boxes,
                 "epsilons"     : epsilons[start_index:],
                 "data_string"  : data_string,
                 "norm"         : norm,
                 "uuid"         : identifier,
                 "n_samples"    : n_samples,
                 "n"            : n_filled,
                 "t"            : t,
                 "used_epsilons": used_epsilons,
                 "dimensions"   : dimensions}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons[start_index:], n_filled, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons[start_index:], n_filled, used_epsilons, dimensions

def grassberger_procaccia_1d(points, n_samples, t = None, epsilons = np.logspace(-5,-1,10), norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = "", save_input_data = True):
    """
    points:      list of 1d coordinates making up the set under consideration
    n_samples:   number of samples that were necessary to create "points". 
                 Example: if 'points' is all randomly chosen initial conditions 
                 that lead to trajectories staying longer than some t* then 
                 n_samples is the number of all tested trajectories to find
                 'points'.
    epsilons:    list of box-sizes to consider (default: np.logspace(-5,-1,10))
    norm:        normalisation constant for coordinates (default : 1)
    
    grassberger procaccia algorithm to calculate the correlation dimension.
    
    returns epsilons, n_filled, used_epsilons, dimensions (and the path+name of
                 the save file if save_comparison_data is set to True)
    """
    
    n_found = []
    n_points = len(points)
    skip = False
    for e in epsilons:
        close_points = 0
        set_size = 0
        if len(n_found) > 0:
            if n_found[-1] == 0:
                n_found += [0]
                skip = True

        if not skip:
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
    
    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "grassberger_procaccia_1d---"+"data_string"+"---n_samples__{}".format(n_samples)+"---"+identifier+".p"

        if not save_input_data:
            points = None

        stuff = {"data"         : points,
                 "epsilons"     : epsilons,
                 "data_string"  : data_string,
                 "norm"         : norm,
                 "uuid"         : identifier,
                 "n_samples"    : n_samples,
                 "n"            : n_found,
                 "t"            : t,
                 "used_epsilons": used_epsilons,
                 "dimensions"   : dimensions}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons, n_found, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons, n_found, used_epsilons, dimensions

# ============================================================================
# algorithms for creating the set:
# ============================================================================

def output_function_evaluation_1d(output_function, min_step = pow(10,-5), alpha = 3, t = 100, max_step_factor =2, delta = .000005):
    """
    implementation of the method from 
    http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.2778

    for a detailed desription of the parameters please heck out said paper.
    """
    max_step = max_step_factor * min_step
    x = min_step
    step = x
    X = [0,x]
    data = [output_function(x) for x in X]
    i = 1

    #array to track which step [min, adaptive, max] was used: to check if 
    #parameter choice was ok
    used = [0,0,0]
    
    temps = []
    steps = []
    slopes = []
    
    while x < 1:
        slope = (data[i] - data[i - 1]) / step
        slopes += [slope]
        if slope == 0:
            step = max_step
            used[2] += 1
        else:
            temp = min([delta / abs(slope), alpha * step])
            temps += [delta / abs(slope)]
            if temp < min_step:
                step = min_step
                used[0] += 1
            elif temp > max_step:
                step = max_step
                used[2] += 1
            else:
                step = temp
                used[1] += 1
        x=min([x + step,1])
        X += [x]
        data += [output_function(x)]
        i += 1
        steps+=[step]
    return

def gaio_stable_manifold_1d(pre_image, max_n_checked = pow(10, 5), required_resolution = pow(10,-5)):
    """
    use method described here:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.27.27
    to approximate the set of boxes at the required resolution intersecting
    the stable manifold
    """

    n_checked = 0

    l_max = -int(log(required_resolution, 2)) - 1
    levels = [l for l in xrange(5, l_max+1)]
    
    check_per_box = 100

    #set all as overlapping, so that alg. doesn't termonate right away
    overlapping_last_level = np.ones(pow(2, levels[0] - 1))

    for l in levels:
        box_size = pow(2., -l)
        overlapping = np.zeros(pow(2, l))
        for box_index in xrange(pow(2, l)):
            if overlapping_last_level[int(box_index /2)] == 1:
                for i in xrange(check_per_box):
                    pi = pre_image(box_index * box_size + np.random.random() * box_size)
                    n_checked += 1
                    if int(pi[0] / box_size) == box_index or int(pi[1] / box_size) == box_index:
                        #this box contains its preimage.
                        overlapping[box_index] = 1
                        break
        overlapping_last_level = overlapping[:]
    
    #getting stable manifold:
    n_ic = 10
    pts = np.array([])
    
    stable = overlapping[:]
    
    for idx, fixed in enumerate(overlapping):
        if fixed == 1:
            pts = np.append(pts, idx * box_size + np.random.random(n_ic) * box_size)

    new_pts = pts

    old_n_found = sum(stable)
    
    while n_checked <= max_n_checked:
        new_pts = [pre_img for pt in pts for pre_img in pre_image(pt)]
        n_checked += len(new_pts)
        new_pts = [pt for pt in new_pts if 0 <= pt < 1]
        pts=np.append(pts, new_pts)
        for pt in new_pts:
            stable[int(pt / box_size)] = 1
        n_found = sum(stable)
        if n_found == old_n_found:
            break

    return
            
# ============================================================================
# on the fly algorithms:
# ============================================================================

def uncertainty_method_1d(indicator, epsilons = np.logspace(-5,-1,10), n_required=100, n_max=10000, norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = ""):
    """
    indicator:   a point x is considered uncertain if 
                 indicator(x)!=indicator((x + e)%1). e is the current epsilon
    n_required:  the number of uncertain points the algorithm tryes to reach 
                 before going to the next epsilon.
    n_max:       the maximum number of attempts before going to the next 
                 epsilon.
                 
    calculates the scaling of the ratio of points in an epsilon environment of
    a point that lead to opposite results (one stays in the system, the other
    leaves). this scaling allows us to calculate a fractal dimension of the set.

    returns epsilons, uncertainty, used_epsilons, dimensions (and the path+name of
                 the save file if save_comparison_data is set to True)
                 uncertainty contains the fraction of uncertain points at each epsilon.
    """
    n_computed = 0
    uncertainty = []
    for e in epsilons:
        n_uncertain = 0
        n_tried = 0
        while n_uncertain < n_required and n_tried < n_max:
            n_tried += 2
            x = np.random.random()
            event_1 = indicator(x)
            event_2 = indicator((x + e) % 1)
            n_computed += 2
            if not (event_1 == event_2 ):
                n_uncertain += 1
        uncertainty += [n_uncertain / n_tried]
    dimensions = []
    used_epsilons = []
    for e_idx, e in enumerate(epsilons[:-1]):
        if uncertainty[e_idx] > 0 and uncertainty[e_idx + 1] > 0:
            used_epsilons += [e]     
            dimensions += [1 - (log(uncertainty[e_idx]) - log(uncertainty[e_idx + 1])) / (log(e) - log(epsilons[e_idx + 1]))]

    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "uncertainty_method_1d---"+"data_string"+"---n_computed__{}".format(n_computed)+"---"+identifier+".p"

        stuff = {"epsilons"     : list(epsilons),
                 "data_string"  : data_string,
                 "norm"         : norm,
                 "uuid"         : identifier,
                 "n_samples"    : n_computed,
                 "n"            : uncertainty,
                 "used_epsilons": used_epsilons,
                 "dimensions"   : dimensions,
                 "n_required"   : n_required,
                 "n_max"        : n_max}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons, uncertainty, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons, uncertainty, used_epsilons, dimensions

def tree_bottom_up_1d(time_function, n_samples = 10000, t = 10, epsilons = np.logspace(-5,-1,10), norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = ""):
    """
    time_function: returns the escape time for a initial condition
    n_samples:     maximum number of points sampled. 
    t:             minimum escape time for an initial condition to be part of
                   the set. Or more general: we accept a sample x if 
                   time_function(x)>=t

    Sample at most a given number of points in each box at the smallest epsilon:

    returns epsilons, n_filled, used_epsilons, dimensions (and the path+name of
                   the save file if save_comparison_data is set to True)
    """
    n_tested = 0
    n_filled = np.zeros(len(epsilons))

    for e_idx, e in enumerate(epsilons):
        if e_idx == 0:
            n_boxes = int(1 / e)
            if n_boxes * e < norm:
                n_boxes += 1
            boxes = np.zeros(n_boxes)

            all_filled = False
            while n_tested < n_samples and all_filled == False:
                #distribute samples on empty boxes:
                n_empty = n_boxes - int(sum(boxes))
                if n_empty > 0:
                    max_points = max([int((n_samples - n_tested) / n_empty), 1])
                else:
                    max_points = 0
                    all_filled = True

                for box_index, filled in enumerate(boxes):
                    if filled < 1:
                        for i in xrange(max_points):
                            x = (box_index + np.random.random()) * e
                            n_tested += 1
                            time = time_function(x)
                            if time >= t:
                                boxes[box_index] = 1
                                break
            n_filled[e_idx] = sum(boxes)
        else:
            larger_boxes = np.zeros(int( 1 / e ) + 1)
            for box_index in xrange(len(larger_boxes)):
                lower_idx = int(box_index       * e / epsilons[e_idx-1])
                upper_idx = int((box_index + 1) * e / epsilons[e_idx-1])
                if sum(boxes[lower_idx:upper_idx + 2]) > 0:
                    larger_boxes[box_index] = 1
            n_filled[e_idx] = sum(larger_boxes)
            boxes = larger_boxes
    
    dimensions = []
    used_epsilons = []
    for e_idx, e in enumerate(epsilons[:-1]):
        if n_filled[e_idx] > 0 and n_filled[e_idx+1] > 0:
            used_epsilons += [e]
            dimensions+=[-(log(n_filled[e_idx]) - log(n_filled[e_idx + 1])) / (log(e) - log(epsilons[e_idx + 1]))]

    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "tree_bottom_up_1d---"+"data_string"+"---n_samples__{}".format(n_samples)+"---"+identifier+".p"

        stuff = {"epsilons"     : epsilons,
                 "data_string"  : data_string,
                 "t"            : t,
                 "norm"         : norm,
                 "uuid"         : identifier,
                 "n_samples"    : n_samples,
                 "n_tested"     : n_tested,
                 "n"            : n_filled,
                 "used_epsilons": used_epsilons,
                 "dimensions"   : dimensions}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons, n_filled, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons, n_filled, used_epsilons, dimensions


def tree_top_down_1d(time_function, n_samples = 10000, t = 10, epsilons = np.logspace(-5,-1,10), norm = 1., save_comparison_data = False, data_dir = "/tmp/", data_string = ""):
    """
    time_function: returns the escape time for a initial condition
    n_samples:     maximum number of points sampled. 
    t:             minimum escape time for an initial condition to be part of
                   the set. Or more general: we accept a sample x if 
                   time_function(x)>=t

    Sample at most a given number of points in each box at the largest epsilon and refine only filled boxes:

    returns epsilons, n_filled, used_epsilons, dimensions (and the path+name of
                   the save file if save_comparison_data is set to True)
    """

    epsilons = epsilons[::-1]
    n_tested = 0
    n_filled = np.zeros(len(epsilons))
    
    found_points = []

    for e_idx, e in enumerate(epsilons):

        max_points = max([int(e * (n_samples-n_tested)),1])

        if e_idx == 0:
            n_boxes = int(1 / e)
            if n_boxes * e < norm:
                n_boxes += 1
            boxes = np.zeros(n_boxes)

            for box_index in xrange(n_boxes):
                lower_bound = box_index * e
                upper_bound = lower_bound + e
                
                #make sure we don't sample outside [0,1]
                delta=min([e, 1. - lower_bound])
                
                for i in xrange(max_points):
                    x = lower_bound + delta * np.random.random()
                    n_tested += 1
                    time = time_function(x)
                    if time >= t:
                        boxes[box_index] = 1
                        found_points += [x]
                        break
        else:
            n_boxes = int(1 / e)
            if n_boxes * e < norm:
                n_boxes += 1
            smaller_boxes = np.zeros(n_boxes)

            for x in found_points:
                smaller_boxes[int(x/e)] = 1

            for box_index, filled in enumerate(smaller_boxes):
                if filled < 1:
                    lower_bound = box_index * e
                    upper_bound = lower_bound + e
                    lower_idx = int(box_index * e / epsilons[e_idx - 1])
                    upper_idx = min([int((box_index + 1) * e / epsilons[e_idx - 1]), len(boxes) - 1])
                    if boxes[lower_idx] > 0 or boxes[upper_idx] > 0:
                        for i in xrange(max_points):
                            x = lower_bound + e * np.random.random()
                            n_tested += 1
                            time = time_function(x)
                            if time >= t:
                                smaller_boxes[box_index] = 1
                                found_points += [x]
                                smaller_boxes[box_index] = 1
                                break
            boxes = smaller_boxes[:]
              
        n_filled[e_idx] = sum(boxes)

    dimensions = []
    used_epsilons = []
    for e_idx, e in enumerate(epsilons[:-1]):
        if n_filled[e_idx] > 0 and n_filled[e_idx + 1] > 0:
            used_epsilons += [epsilons[e_idx + 1]]
            dimensions+=[-(log(n_filled[e_idx]) - log(n_filled[e_idx + 1])) / (log(e) - log(epsilons[e_idx + 1]))]

    if save_comparison_data:
        identifier = uuid.uuid1().hex

        file_name = "tree_top_down_1d---"+"data_string"+"---n_samples__{}".format(n_samples)+"---"+identifier+".p"

        stuff = {"epsilons"     : epsilons,
                 "data_string"  : data_string,
                 "t"            : t,
                 "norm"         : norm,
                 "uuid"         : identifier,
                 "n_samples"    : n_samples,
                 "n_tested"     : n_tested,
                 "n"            : n_filled,
                 "used_epsilons": used_epsilons,
                 "dimensions"   : dimensions}
                        
        pickle.dump(stuff, open(data_dir+file_name, "w"))

        return epsilons, n_filled, used_epsilons, dimensions, data_dir+file_name
    else:
        return epsilons, n_filled, used_epsilons, dimensions

