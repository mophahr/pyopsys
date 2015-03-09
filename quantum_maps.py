# -*- coding: utf-8 -*-
from __future__ import division
"""
    quantum_maps module
    
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
import cmath as cmt
import math as mt
import errno
import sys
from math import pi

import numpy as np
from mpmath import jtheta


class QuantumMap:
    def __init__(self, M):
        self.M = M

        # placeholders eigenvalues/eigenvectors/etc.
        self.eva = None
        self.eva_calculated = False
        self.eve = None
        self.eve_calculated = False
        self.propagator_matrix = None
        self.propagator_matrix_filled = False
        self.moduli = None
        self.sorted_moduli = None
        self.sorted_eva = None
        self.sorted_exists = False

    def propagator(self):
        """
        To be defined in subclasses.
        """
        raise NotImplementedError

    # ============================================================================
    # eigensystem aquisition:
    # ============================================================================
    def eigenvalues(self):
        """calculate eigenvalues of self.propagator"""
        if not self.eva_calculated:
            self.eva = np.linalg.eigvals(self.propagator())
            self.eva_calculated = True
        return self.eva

    def eigensystem(self):
        """calculate right eigensystem of self.propagator"""
        if not self.eva_calculated or not self.eve_calculated:
            self.eva, self.eve = np.linalg.eig(self.propagator())
            self.eva_calculated = True
            self.eve_calculated = True
        return self.eva, self.eve

    def load_eigensystem(self, eva_file_name="eigenvalues",
                         eve_file_name="eigenvectors"):
        """load right eigensystem from <eve_file_name>.npy and <eva_file_name>.npy"""
        self.eve = np.load(eve_file_name + ".npy")
        self.eva = np.load(eva_file_name + ".npy")

    def load_eigenvalues(self, eva_file_name="eigenvalues"):
        """load eigenvalues from <eva_file_name>.npy """
        self.eva = np.load(eva_file_name + ".npy")

    def save_eigensystem(self, eva_file_name="eigenvalues",
                         eve_file_name="eigenvectors", ):
        """save eigensystem to <eve_file_name>.npy and <eva_file_name>.npy"""
        try:
            self.eve
        except AttributeError:
            print("(!)failed. No or incomplete eigensystem present.")
            sys.exit(errno.ENODATA)
        np.save(eve_file_name, self.eve)
        np.save(eva_file_name, self.eva)

    def save_eigenvalues(self, eva_file_name="eigenvalues"):
        """save eigenvalues to <eva_file_name>.npy" """
        try:
            self.eva
        except AttributeError:
            print("(!)failed. No eigensystem present.")
            sys.exit(errno.ENODATA)
        np.save(eva_file_name, self.eva)

    # ============================================================================
    # eigensystem refining:
    # ============================================================================
    def convert_to_energies(self):
        """convert radial eigenvalues to Re(E) Im(E) style."""
        energies = []
        n_finite = 0
        for i in range(self.M):
            absmu = cmt.polar(self.eva[i])[0]
            if absmu > 0.0:
                ImE = mt.log(absmu)
                ReE = cmt.log(self.eva[i] / absmu) * 1j
                energies.extend([ReE + 1j * ImE])
                n_finite += 1
        writelist = [[energies[i].imag, energies[i].real] for i in
                     range(n_finite)]
        np.savetxt("complex_energies" + self.idString + ".dat", writelist)

    def eigenvalues_sorted_by_modulus(self):
        """sorting by |\nu| in descending order"""
        eigenvalues = self.eigenvalues()
        if not self.sorted_exists:
            self.moduli = np.array(
                [cmt.polar(eigenvalue)[0] for eigenvalue in eigenvalues])
            sort_perm = self.moduli.argsort()[::-1]
            self.sorted_eva = eigenvalues[sort_perm]
            self.sorted_moduli = self.moduli[sort_perm]
            self.sorted_exists = True
        return self.sorted_eva

    # ============================================================================
    # phase_space visualisation of selected states:
    # ============================================================================
    def husimi_distribution(self, eigenstates, n_coherent_states=10,
                            epsilon=1.e-1):
        """
        Returns the husimi-distribution of the selected states.
        For a definition check out Arnd Bäcker's chapter on
        "Numerical Aspects of Eigenvalue and Eigenfunction Computations for
        Chaotic Quantum Systems" in "The Mathematical Aspects of Quantum Maps",
        Lecture Notes in Physics Volume 618, 2003, pp 91-144,
        http://link.springer.com/chapter/10.1007/3-540-37045-5_4

        We set \theta_1=\theta_2+1/2 in (38)
            q = m/n_coherent_states
            p = m/n_coherent_states
        """
        #jtheta(3,Z,t)

        mean_husimi = np.zeros([n_coherent_states, n_coherent_states])
        summed_husimi = np.zeros([n_coherent_states, n_coherent_states])

        n_eigenstates = len(eigenstates)

        for m in range(n_coherent_states):
            p = m / n_coherent_states
            for n in range(n_coherent_states):
                q = n / n_coherent_states

                for i in range(n_eigenstates):
                    temp_sum_value = 0
                    for j in range(self.M):
                        # j-th coefficient of i-th eigenstate:
                        c_j = eigenstates[i, j]
                        temp_summand_value = c_j
                        if abs(temp_summand_value) < epsilon:
                            continue
                        else:
                            temp_summand_value *= pow(2 * self.M, 1 / 4)

                        if abs(temp_summand_value) < epsilon:
                            continue
                        else:
                            temp_summand_value *= cmt.exp(
                                -pi * self.M * (q ** 2 - 1j * p * q))

                        if abs(temp_summand_value) < epsilon:
                            continue
                        else:

                            temp_summand_value *= jtheta(
                                3, 1j * pi * self.M * ((j + 1 / 2) / self.M - 1j /
                                                       (2 * self.M) - q + 1j * p),
                                cmt.exp(-pi * self.M))

                        temp_sum_value += temp_summand_value

                    husimi = abs(temp_sum_value) ** 2
                    mean_husimi[m, n] += husimi
                    summed_husimi[m, n] += husimi

        return summed_husimi, mean_husimi

    #            icenter = int( n/npo *self.M )
    #            hus=np.array( [cmt.exp(-self.M*mt.pi*pow(i/self.M-n/npo,2) - 2*mt.pi*self.M*1j*m/npo*i/self.M) for i in range()] )
    #            meanval=0
    #            for ns in range(self.minSelectedIndex, self.maxSelectedIndex):
    #                val=self.eve_sorted[:,ns][imin:imax].dot( hus )
    #                meanval= meanval+pow( abs( val ), 2)
    #            husimi[m,n]=meanval/ns


class TernaryBaker(QuantumMap):
    """
    For a definition check out M. S. and Eduardo G. Altmann
    'Quantum signatures of classical multifractal measures',
    Phys. Rev. E 91, 012919 (2015)
    http://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.012919
    """

    def __init__(self, M, reflectivity_center, reflectivity_left=1,
                 reflectivity_right=1):
        QuantumMap.__init__(self, M)
        self.R1 = reflectivity_left
        self.R2 = reflectivity_center
        self.R3 = reflectivity_right

    def propagator(self):
        if not self.propagator_matrix_filled:
            def f(n, m):
                if n < self.M / 3 and m < self.M / 3:
                    return mt.sqrt(self.R1) / mt.sqrt(self.M / 3) * cmt.exp(
                        -2 * pi * 1j / (self.M / 3) * (n + 1 / 2) * (m + 1 / 2))
                elif (self.M / 3 <= n < 2 / 3 * self.M and
                      self.M / 3 <= m < 2 / 3 * self.M):
                    return mt.sqrt(self.R2) / mt.sqrt(self.M / 3) * cmt.exp(
                        -2 * pi * 1j / (self.M / 3) * (
                            (n - self.M / 3) + 1 / 2) * ((m - self.M / 3) + 1 / 2))
                elif n >= self.M * 2 / 3 and m >= self.M * 2 / 3:
                    return mt.sqrt(self.R3) / mt.sqrt(self.M / 3) * cmt.exp(
                        -2 * pi * 1j / (self.M / 3) * (
                            (n - 2 * self.M / 3) + 1 / 2) * (
                            (m - 2 * self.M / 3) + 1 / 2))
                else:
                    return 0

            self.propagator_matrix = np.array(
                [[f(n, m) for n in range(self.M)] for m in range(self.M)])
            inverse_fourier_matrix = np.array([[1 / mt.sqrt(self.M) * cmt.exp(
                2 * pi * 1j / self.M * (n + 1 / 2) * (m + 1 / 2)) for n in
                                                range(self.M)] for m in
                                               range(self.M)])
            self.propagator_matrix = inverse_fourier_matrix.dot(
                self.propagator_matrix)
            self.propagator_matrix_filled = True
        return self.propagator_matrix


class CatMap(QuantumMap):
    """from S.P. Kuznetsov
        Disheveled Arnold's cat and the problem of quantum-classic correspondence
        Physica D: Nonlinear Phenomena, Volume 137, Issues 3-4, 15 March 2000, Pages 205-227
        http://www.sciencedirect.com/science/article/pii/S0167278999001827
        """

    def __init__(self, M, reflectivity_center, reflectivity_left=1,
                 reflectivity_right=1, hole=(1 / 6, 4 / 6)):
        QuantumMap.__init__(self, M)
        if not M % 2 == 0:
            print("For cat-map M must be even!")
            sys.exit(errno.EINVAL)
        self.R1 = reflectivity_left
        self.R2 = reflectivity_center
        self.R3 = reflectivity_right
        self.R2_region_lower_bound = int(self.M * hole[0])
        self.R2_region_upper_bound = int(self.M * hole[1])

    def propagator(self):
        if not self.propagator_matrix_filled:
            def f(n, m):
                tmp = 1 / mt.sqrt(self.M) * cmt.exp(
                    2 * pi * 1j / self.M * ( m * m - m * n + n * n))
                if n < self.R2_region_lower_bound:
                    return mt.sqrt(self.R1) * tmp
                if self.R2_region_lower_bound <= n < self.R2_region_upper_bound:
                    return mt.sqrt(self.R2) * tmp
                else:
                    return mt.sqrt(self.R3) * tmp

            self.propagator_matrix = np.array(
                [[f(n, m) for n in range(self.M)] for m in range(self.M)])
            self.propagator_matrix_filled = True
        return self.propagator_matrix

# ===============================================================================
# NOT USED AT THE MOMENT:
# ===============================================================================
#def fMiddleThird3R_arbitraryPhases( M, n, m, ref1, ref2, ref3, chi_p=.5, chi_q=.5 ):
#    """this map is not yet implemented. code is kept here for future reference """'
#    if (n<M/3 and m<M/3):
#        return mt.sqrt(ref1)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*(n+chi_q)*(m+chi_p))
#    elif (n>=M/3 and n<2/3*M and m>=M/3 and m<2/3*M):
#        return mt.sqrt(ref2)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-M/3)+chi_q)*((m-M/3)+chi_p))
#    elif (n>=M*2/3 and m>=M*2/3):
#        return mt.sqrt(ref3)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-2*M/3)+chi_q)*((m-2*M/3)+chi_p))
#    else:
#        return 0
#   
