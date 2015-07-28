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
from nose.tools import assert_equals, assert_true, assert_almost_equals
import dimension_algorithms
import maps
import numpy as np
import pickle

def test_binning():
    """ Testing the conversion points->filledarray"""
    #1D-tests:
    points=np.array([[.5],[.75]])
    raster=dimension_algorithms.raster(points,0.5)[1]
    np.testing.assert_almost_equal([0,1],raster)
    raster=dimension_algorithms.raster(points,0.25)[1]
    np.testing.assert_almost_equal([0,0,1,1],raster)
    #2D-tests:
    #to_do

def test_box_counting():
    test_n_filled = pickle.load(open('./test_data/test_data_box_counting_n_filled.p', 'r'))
    test_used_epsilons = pickle.load(open('./test_data/test_data_box_counting_used_epsilons.p', 'r'))
    test_X = pickle.load(open('./test_data/test_data_box_counting_X.p', 'r'))
    test_dims = pickle.load(open('./test_data/test_data_box_counting_dims.p', 'r'))
    
    epsilons = np.logspace(-5,-1,10)

    es, nfs, ues, dims = dimension_algorithms.box_counting_1d(test_X, epsilons)

    np.testing.assert_almost_equal(epsilons, es)
    np.testing.assert_almost_equal(test_used_epsilons, ues)
    test_relative_filled = [test_n_filled[i] * e for i,e in enumerate(epsilons)]
    rfs = [nfs[i] * e for i, e in enumerate(epsilons)]
    np.testing.assert_almost_equal(test_relative_filled, rfs)
    np.testing.assert_almost_equal(test_dims, dims)

def test_GPA():
    test_n_found = pickle.load(open('./test_data/test_data_gpa_n_found.p', 'r'))
    test_used_epsilons = pickle.load(open('./test_data/test_data_gpa_used_epsilons.p', 'r'))
    test_X = pickle.load(open('./test_data/test_data_gpa_X.p', 'r'))
    test_dims = pickle.load(open('./test_data/test_data_gpa_dims.p', 'r'))
    
    epsilons = np.logspace(-5,-1,10)

    es, nfs, ues, dims = dimension_algorithms.grassberger_procaccia_1d(test_X, 10**4,epsilons)

    np.testing.assert_almost_equal(epsilons, es)
    np.testing.assert_almost_equal(test_used_epsilons, ues)
    test_relative_found = [test_n_found[i] * e for i, e in enumerate(epsilons)]
    rfs=[nfs[i] * e for i,e in enumerate(epsilons)]
    np.testing.assert_almost_equal(test_relative_found, rfs)
    np.testing.assert_almost_equal(test_dims, dims)

def test_other():
    testmap = maps.TentMap(.62, 3)
    dimension_algorithms.tree_bottom_up_1d(lambda x: testmap.time_until_hole(x)[0])
    dimension_algorithms.tree_top_down_1d(lambda x: testmap.time_until_hole(x)[0])
    dimension_algorithms.output_function_evaluation_1d(lambda x: testmap.time_until_hole(x)[0])
    dimension_algorithms.uncertainty_method_1d(lambda x: testmap.time_until_hole(x)[0])

def test_gaio():
    asymmetry_parameter = .62
    stretching_parameter = 3
    dimension_algorithms.gaio_stable_manifold_1d(lambda x: [1 - 2 * (1 - asymmetry_parameter) * x / stretching_parameter, 2 * asymmetry_parameter * x / stretching_parameter])

def test_io():
    test_X = pickle.load(open('./test_data/test_data_box_counting_X.p', 'r'))
    testmap = maps.TentMap(.62, 3)

    e, n, ue, d, f = dimension_algorithms.box_counting_1d(test_X, save_comparison_data = True, data_dir = "/tmp/", data_string = "*T*E*S*T*")
    saved_data = pickle.load(open(f, "r"))
    np.testing.assert_equal(saved_data["data"],test_X)
    np.testing.assert_equal(saved_data["epsilons"],e)
    np.testing.assert_equal(saved_data["n"],n)
    np.testing.assert_equal(saved_data["used_epsilons"],ue)
    np.testing.assert_equal(saved_data["dimensions"],d)

    e, n, ue, d, f = dimension_algorithms.grassberger_procaccia_1d(test_X, 10**4, save_comparison_data = True, data_dir = "/tmp/", data_string = "*T*E*S*T*")
    saved_data = pickle.load(open(f, "r"))
    np.testing.assert_equal(saved_data["data"],test_X)
    np.testing.assert_equal(saved_data["epsilons"],e)
    np.testing.assert_equal(saved_data["n"],n)
    np.testing.assert_equal(saved_data["used_epsilons"],ue)
    np.testing.assert_equal(saved_data["dimensions"],d)

    e, n, ue, d, f = dimension_algorithms.tree_bottom_up_1d(lambda x: testmap.time_until_hole(x)[0], save_comparison_data = True, data_dir = "/tmp/", data_string = "*T*E*S*T*")
    saved_data = pickle.load(open(f, "r"))
    np.testing.assert_equal(saved_data["epsilons"],e)
    np.testing.assert_equal(saved_data["n"],n)
    np.testing.assert_equal(saved_data["used_epsilons"],ue)
    np.testing.assert_equal(saved_data["dimensions"],d)

    e, n, ue, d, f = dimension_algorithms.tree_top_down_1d(lambda x: testmap.time_until_hole(x)[0], save_comparison_data = True, data_dir = "/tmp/", data_string = "*T*E*S*T*")
    saved_data = pickle.load(open(f, "r"))
    np.testing.assert_equal(saved_data["epsilons"],e)
    np.testing.assert_equal(saved_data["n"],n)
    np.testing.assert_equal(saved_data["used_epsilons"],ue)
    np.testing.assert_equal(saved_data["dimensions"],d)

    e, n, ue, d, f = dimension_algorithms.uncertainty_method_1d(lambda x: testmap.time_until_hole(x)[0], save_comparison_data = True, data_dir = "/tmp/", data_string = "*T*E*S*T*")
    saved_data = pickle.load(open(f, "r"))
    np.testing.assert_equal(saved_data["epsilons"],e)
    np.testing.assert_equal(saved_data["n"],n)
    np.testing.assert_equal(saved_data["used_epsilons"],ue)
    np.testing.assert_equal(saved_data["dimensions"],d)


