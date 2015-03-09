# -*- coding: utf-8 -*-
from __future__ import division, print_function
"""
    tests for quantum_maps module
    
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
import quantum_maps

# EPSILON is the maximum distance between two floats that they are still considered equal:
EPSILON = 1.e-15


def test_saving_and_loading():
    """check if loading saving alters the values in any way."""
    test_bakermap = quantum_maps.TernaryBaker(27, .1)
    eval_only_before_saving = test_bakermap.eigenvalues()
    eval_before_saving, evec_before_saving = test_bakermap.eigensystem()

    assert_true((eval_before_saving == eval_only_before_saving).all())

    test_bakermap.save_eigenvalues("test_eva_only")
    test_bakermap.save_eigensystem("test_eva", "test_eve")

    test_bakermap.load_eigenvalues("test_eva_only")
    eval_only_after_saving = test_bakermap.eigenvalues()
    assert_true((eval_before_saving == eval_only_after_saving).all())

    test_bakermap.load_eigensystem("test_eva", "test_eve")
    eval_after_saving, evec_after_saving = test_bakermap.eigensystem()
    assert_true((eval_before_saving == eval_after_saving).all())
    assert_true((evec_before_saving == evec_after_saving).all())


def test_moduli_sorting():
    """make sure things get sorted from largest to smallest |\nu|."""
    test_bakermap = quantum_maps.TernaryBaker(27, .1)
    to_be_husimid = test_bakermap.eigensystem()[1][:10]
    print(test_bakermap.husimi_distribution(to_be_husimid))
