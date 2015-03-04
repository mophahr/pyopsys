# -*- coding: utf-8 -*-
from __future__ import division
'''
	quantum maps module
	
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
import numpy as np
import cmath as cmt
import math as mt
from scipy.optimize import curve_fit
import errno, sys
import time
import os

class quantumMap:
	'''The Quantum map class; provide a fuction f(M,n,m) with M being the matrix
	dimension, n the position-like index and m the 'momentum': A hole can be 
	either introduced directly in f or by multipication with the projector matrix 
	in setReflectivities( self,r[,...] ) where r(M,n,m) are the matrix-elements; 
	standard here is element-wise multiplication, so if a matrix-product is 
	necessary use ElementByElement=False as a third argument when calling 
	setReflectivities(...).
	'''
	
	def __init__( self, M ):
	self.M=M

	#these strings are used for naming files
	self.id_string="--M_"+str(M)
	self.select_string=""

	#placeholders eigenvalues/eigenvectors/etc.
	self.eva=None
	self.eve=None
	self.propagator=None

	#============================================================================
	# eigen system generation/manipulation:
	#============================================================================
	def eigenvalues( self ):
		self.eva = self.eva or np.linalg.eigvals( self.propagator )
		return self.eva
	
	def eigensystem( self ):
		if not self.eva or not self.eve:
			self.eva, self.eve = np.linalg.eig( self.propagator )
		return self.eva, self.eve
	
	def load_eigensystem( self, eve_file_name="eigenvectors", eva_file_name="eigenvalues" ):
		''' load right eigensystem from <eve_file_name>.npy and <eva_file_name>.npy '''
		self.eve = np.load( eveFileName+".npy" )
		self.eva = np.load( evaFileName+".npy" )
	
	def load_eigenvalues( self, eva_file_name="eigenvalues" ):
		''' load eigenvalues from <eva_file_name>.npy '''
		self.eva = np.load( evaFileName+".npy" )

	def save_eigensystem( self, eve_file_name="eigenvectors", eva_file_name="eigenvalues" ):
		''' save eigensystem to <eve_file_name>.npy and <eva_file_name>.npy" '''
		try:
			self.eve
		except AttributeError:
			print("(!)failed. No or incomplete eigensystem present.")
			sys.exit(errno.ENODATA)
		np.save( eve_file_name, self.eve )
		np.save( eva_file_name, self.eva )

	def save_eigenvalues( self,  eva_file_name="eigenvalues" ):
		''' save eigenvalues to <eva_file_name>.npy" '''
		try:
			self.eva
		except AttributeError:
			print("(!)failed. No eigensystem present.")
			sys.exit(errno.ENODATA)
		np.save( eva_file_name, self.eva )
	
	def convert_to_energies( self, print_human_readable=True ):
		''' convert radial eigenvalues to Re(E) Im(E) style '''
		self.energies=[]
		n_finite=0
		for i in range(self.M):
			absmu=cmt.polar( self.eva[i] )[0]
			if absmu>0.0:
				ImE=mt.log( absmu )
				ReE=cmt.log( self.eva[i]/absmu )*1j
				self.energies.extend( [ReE+1j*ImE] )
				n_finite += 1
		writelist=[[self.energies[i].imag, self.energies[i].real] for i in range(n_finite)]
		np.savetxt( "complex_energies"+self.idString+".dat", writelist )
	
	
	#============================================================================
	# eigenvalue-plots: 
	#============================================================================
	def moduliDistribution( self, useLog=False, writeLimits=False, saveHists=False, plotHists=True ):
		print("plotting |mu| distribution")
		try:
			self.moduli_sorted
		except AttributeError:
			self.sortModuli()
		
		binSize=0.005; minEdge=0; maxEdge=1
		N=(maxEdge-minEdge)/binSize; Nplus1=N+1
		binList=np.linspace(minEdge, maxEdge, Nplus1)

		if plotHists:
			plt.subplot(2,1,1)
			plt.hist(self.moduli_sorted, bins=binList, log=useLog,histtype="stepfilled",alpha=0.75)
			plt.hist(self.moduli_sorted[:int(self.M/3)], facecolor='yellow', bins=binList, log=useLog,histtype="stepfilled",alpha=0.75)
			plt.hist(self.moduli_sorted[2*int(self.M/3):], facecolor='red', bins=binList, log=useLog,histtype="stepfilled",alpha=0.75)
			plt.subplot(2,1,2)
			plt.hist(self.moduli,cumulative=True, normed=True, bins=binList,histtype="stepfilled",alpha=0.75)
			plt.savefig("moduli_histogram"+self.idString+".pdf")
			plt.close()
		if writeLimits:
			limitsFile = open( "borders"+self.idString, "w" )
			limitsFile.write("%d %f %f\n" %( self.M, self.moduli_sorted[int(self.M/3)], self.moduli_sorted[2*int(self.M/3)] ) )
			limitsFile.close()

		if saveHists:
			histFile = open( "AbsMuHistogram"+self.idString, "w" )
			writeData = np.histogram( self.moduli, bins=binList, normed=True)[0]
			for i in range(int(N)):
				histFile.write( "%f %f\n" %(binList[i],writeData[i]) )
			histFile.close()

	def ImEDistribution( self, useLog=False ):
		self.convertToEnergies( ) 
		histFile = open( "ImEHistogram"+self.idString, "w" )
		ImEList = [ e.imag for e in self.energies ]
		
		minEdge=min(ImEList); maxEdge=max(ImEList)
		N=200
		Nplus1=N+1
		binList=np.linspace(minEdge, maxEdge, Nplus1)
		
		writeData = np.histogram( ImEList, bins=binList, normed=True)[0]
		for i in range(int(N)):
			histFile.write( "%f %f\n" %(binList[i],writeData[i]) )
		histFile.close()

	def plotEigenvalues( self ):
		print("plotting eigenvalues")
		try:
			self.eva
		except AttributeError:
			print("(!)failed")
			sys.exit(errno.ENODATA)
		plt.subplot(111,aspect='equal')
		plt.scatter([a.real for a in self.eva.tolist()],[a.imag for a in self.eva.tolist()])
		plt.savefig("moduli"+self.idString+".pdf")
		plt.close()



	#============================================================================
	# moduli:
	#============================================================================
	def sortModuli( self ):
		print("sorting by |mu|")
		try:
			self.eva
		except AttributeError:
			print("(!)failed. No eigensystem present.")
			sys.exit(errno.ENODATA)
		   
		self.moduli = np.array([cmt.polar( self.eva[i] )[0] for i in range(self.M)])
		self.sort_perm = self.moduli.argsort()[::-1]
		self.eva_sorted = self.eva[self.sort_perm]
		self.moduli_sorted = self.moduli[self.sort_perm]
		return self.moduli_sorted 

	def sortEigenvectors( self ):
		print("sorting by |mu|")
		try:
			self.eva
		except AttributeError:
			print("(!)failed. No eigensystem present.")
			sys.exit(errno.ENODATA)
		   
		self.moduli = np.array([cmt.polar( self.eva[i] )[0] for i in range(self.M)])
		self.sort_perm = self.moduli.argsort()[::-1]
		self.eva_sorted = self.eva[self.sort_perm]
		self.eve_sorted = self.eve[:,self.sort_perm]
		self.moduli_sorted = self.moduli[self.sort_perm]
		return self.moduli_sorted 

	def writeModuliHumanReadable( self ):
		print("writing human readable data to (complex_)eigenvalues_%d.dat" % self.M)
		try:
			self.moduli_sorted
		except AttributeError:
			print("(!)failed")
			self.sortModuli()
			print("(!)retry: writing human readable data to (complex_)eigenvalues_%d.dat" % self.M)
		writelist = [[i/self.M,self.moduli_sorted[i]] for i in range(self.M)]
		np.savetxt( "eigenvalues_%d.dat" % self.M, writelist )
		writelist=[[self.eva_sorted[i].real, self.eva_sorted[i].imag] for i in range(self.M)]
		np.savetxt( "complex_eigenvalues_%d.dat" % self.M, writelist )

	#============================================================================
	# selection of states:
	#============================================================================
	def selectNLongestLiving( self, N ):
		print( "select %d longest living eigenstates" % N )
		self.selectString="--"+str(N)+"_longestLiving"
		try:
			self.eve_sorted
		except AttributeError:
			print("(!)failed")
			self.sortEigenvectors()
		self.selectedStates = self.eve_sorted[:,:N]
		self.selectedEigenValues = self.eva_sorted[:N]
		self.minSelectedIndex=0
		self.maxSelectedIndex=N

	def selectRelativerange( self, minQuotient, maxQuotient, ignoreStates=False ):
		self.selectString="--fraction_from_"+str(minQuotient)+"to"+str(maxQuotient)+"_ofLongestLiving"
		print( "select a fraction of the eigensystem" )
		if ignoreStates:
			try:
				self.moduli_sorted
			except AttributeError:
				print("(!)failed")
				self.sortModuli()
		else:
			try:
				self.eve_sorted
			except AttributeError:
				print("(!)failed")
				self.sortEigenvectors()
		if minQuotient==0:
			self.minSelectedIndex=1
		else:
			self.minSelectedIndex=int(minQuotient*self.M)+1
		if maxQuotient==1:
			self.maxSelectedIndex=self.M
		else:
			self.maxSelectedIndex=int(maxQuotient*self.M)
		if not ignoreStates:
			self.selectedStates = self.eve_sorted[:,self.minSelectedIndex:self.maxSelectedIndex]
		self.selectedEigenvalues = self.eva_sorted[self.minSelectedIndex:self.maxSelectedIndex]

	def selectMuRange( self, minMu, maxMu, ignoreStates=False ):
		self.selectString="--mu_from_"+str(minMu)+"to"+str(maxMu)+"_ofLongestLiving"
		print( "select eigenstates with %f<=|mu|<%f" % (minMu, maxMu) )
		if ignoreStates:
			try:
				self.moduli_sorted
			except AttributeError:
				print("(!)failed")
				self.sortModuli()
		else:
			try:
				self.eve_sorted
			except AttributeError:
				print("(!)failed")
				self.sortEigenvectors()
		self.minSelectedIndex=self.M
		self.maxSelectedIndex=0
		for i in range(self.M):
			if minMu<=self.moduli_sorted[i] and self.moduli_sorted[i]<maxMu:
				self.minSelectedIndex=min([i,self.minSelectedIndex])
				self.maxSelectedIndex=max([i,self.maxSelectedIndex+1])
		if self.minSelectedIndex==self.M:
			self.minSelectedIndex=0
		if not ignoreStates:
			self.selectedStates = self.eve_sorted[:,self.minSelectedIndex:self.maxSelectedIndex]
		self.selectedEigenValues = self.eva_sorted[self.minSelectedIndex:self.maxSelectedIndex]

	def selectImETreshold( self, treshold ):
		print( "select eigenstates with |Im(E)| > %f" % treshold )
		self.selectString="--ImE_lt_"+str(treshold)
		try:
			self.eva
		except AttributeError:
			print("(!)failed. No eigensystem present.")
			sys.exit(errno.ENODATA)
		if treshold>1.0 or treshold<0.0:
			print("(!)failed. 0<=C<1 required.")
			sys.exit(errno.EINVAL)

	#============================================================================
	# visualisation of the selected states:
	#============================================================================
	def writeSelected( self, filename="selected_eigenvalues" ):
		writelist=[[element.real, element.imag] for element in self.selectedEigenValues.tolist()]
		print(self.selectString)
		print(self.idString)
		np.savetxt( filename+self.idString+self.selectString+".dat" , writelist )


	def plotSelected( self ):
		try:
			self.moduli_sorted
		except AttributeError:
			print("(!)failed")
			self.sortModuli()
		means = []
		for i in range(self.M):
			val = 0.
			for j in range(self.minSelectedIndex,self.maxSelectedIndex):
				val = val + pow(cmt.polar( self.eve_sorted[i][j] )[0],2)
			means.append([i/self.M,val/(self.maxSelectedIndex-self.minSelectedIndex)])
		means=np.array(means)
		print(means)
		plt.plot(means[:,0],means[:,1],linestyle="-",marker="")
		plt.xlabel(r"$q$" ,fontdict={'fontsize':20})
		plt.ylabel(r"$\left|\psi\right|^2$",fontdict={'fontsize':20})
		plt.savefig("meanwave"+self.idString+self.selectString+".pdf")
		plt.close()

	def husimi( self, NHusimi, saveData=True ):
		print("creating a husimi-plot of the selected states.")
		npo=NHusimi
		try:
			self.selectedStates
		except AttributeError:
			print("(!)failed")
			sys.exit(errno.ENODATA)
		ns=self.selectedStates.shape[1]
		husimi=np.zeros((npo,npo))
		width=int(0.03*self.M)
		for m in range( npo ): #go line by line
			for n in range( npo ):
				icenter = int( n/npo *self.M )
				imin = max([icenter-width,0])
				imax = min([icenter+width,self.M] )
				hus=np.array( [cmt.exp(-self.M*mt.pi*pow(i/self.M-n/npo,2) - 2*mt.pi*self.M*1j*m/npo*i/self.M) for i in range(imin,imax)] )
				meanval=0
				for ns in range(self.minSelectedIndex, self.maxSelectedIndex):
					val=self.eve_sorted[:,ns][imin:imax].dot( hus )
					meanval= meanval+pow( abs( val ), 2)
				husimi[m,n]=meanval/ns
		if saveData:
			np.savetxt("husimi--NHu_%d" % (npo) +self.idString+self.selectString+".dat", husimi)
			np.save( "husimi", husimi )
		plt.imshow(husimi.tolist(), cmap=cm.copper,origin='lower',interpolation='nearest')
		plt.colorbar(shrink=0.8,format="%.1e")
		plt.xlabel(r"$x$")
		plt.ylabel(r"$y$")
		plt.axis([0,NHusimi,0,NHusimi])
		plt.xticks([0,NHusimi],[r"0",r"1"])
		plt.yticks([0,NHusimi],[r"0",r"1"])
		plt.savefig("husimi--NHu_%d" % (npo) +self.idString+self.selectString+".pdf")
		plt.close()



#===============================================================================
'''Definition of the matrices of different maps:
'''
def fMiddleThird( M, n, m, ref ):
	if (n<M/3 and m<M/3):
		return 1/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*(n+1/2)*(m+1/2))
	elif (n>=M/3 and n<2/3*M and m>=M/3 and m<2/3*M):
		return mt.sqrt(ref)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-M/3)+1/2)*((m-M/3)+1/2))
	elif (n>=M*2/3 and m>=M*2/3):
		return 1/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-2*M/3)+1/2)*((m-2*M/3)+1/2))
	else:
		return 0

def fMiddleThird3R_arbitraryPhases( M, n, m, ref1, ref2, ref3, chi_p=.5, chi_q=.5 ):
	if (n<M/3 and m<M/3):
		return mt.sqrt(ref1)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*(n+chi_q)*(m+chi_p))
	elif (n>=M/3 and n<2/3*M and m>=M/3 and m<2/3*M):
		return mt.sqrt(ref2)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-M/3)+chi_q)*((m-M/3)+chi_p))
	elif (n>=M*2/3 and m>=M*2/3):
		return mt.sqrt(ref3)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-2*M/3)+chi_q)*((m-2*M/3)+chi_p))
	else:
		return 0

def fMiddleThird3R( M, n, m, ref1, ref2, ref3 ):
	if (n<M/3 and m<M/3):
		return mt.sqrt(ref1)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*(n+1/2)*(m+1/2))
	elif (n>=M/3 and n<2/3*M and m>=M/3 and m<2/3*M):
		return mt.sqrt(ref2)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-M/3)+1/2)*((m-M/3)+1/2))
	elif (n>=M*2/3 and m>=M*2/3):
		return mt.sqrt(ref3)/mt.sqrt(M/3)*cmt.exp(-2*mt.pi*1j/(M/3)*((n-2*M/3)+1/2)*((m-2*M/3)+1/2))
	else:
		return 0

def bakerProjector( M, n, m, ref ):
	if (n==m):
		if (n>=M/3 and n<2*M/3):
			return mt.sqrt(ref)
		else:
			return 1
	else:
		return 0

def bakerProjector3R( M, n, m, ref1, ref2, ref3 ):
	if (n==m):
		if (n<M/3):
			return mt.sqrt(ref1)
		if (n>=M/3 and n<2*M/3):
			return mt.sqrt(ref2)
		else:
			return mt.sqrt(ref3)
	else:
		return 0

def fCatMap( M, n, m, ref ):
	''' from S.P. Kuznetsov
		Disheveled Arnold's cat and the problem of quantum-classic correspondence
		Physica D: Nonlinear Phenomena, Volume 137, Issues 3-4, 15 March 2000, Pages 205-227
		http://www.sciencedirect.com/science/article/pii/S0167278999001827
	'''
	if M%2!=0:
		print("For cat-map M must be even!")
		sys.exit(errno.EINVAL)
	tmp = 1/mt.sqrt(M)*cmt.exp( 2*mt.pi*1j/M* ( m*m -m*n + n*n))# -mt.pi*1j/4 )
	if (n>=M*1/6 and n<4/6*M):
		return mt.sqrt(ref)*tmp
	else:
		return tmp

	def makePropagator ( self, f, mapName, ref, inverseFourierNeeded=True):
		print("\n\nsetting up the basic propagator")
		self.f=f
		self.ref=ref
		self.mapName=mapName
		self.idString="--"+self.mapName+"--ref_"+str(ref)+self.idString
		print(self.idString)
		print(" ->filling propagation matrix")
		self.propagator=np.array([[self.f(self.M,n,m,self.ref) for n in range(self.M)] for m in range(self.M)])
		if inverseFourierNeeded:
			print(" ->filling inverse fourier matrix")
			self.inv=np.array([[1/mt.sqrt(self.M)*cmt.exp(2*mt.pi*1j/self.M*(n+1/2)*(m+1/2)) for n in range(self.M)] for m in range(self.M)])      
			print(" ->matrix product")
			self.propagator=self.inv.dot(self.propagator)

	def makePropagator3R ( self, f, mapName, ref1, ref2, ref3, inverseFourierNeeded=True):
		print("\n\nsetting up the basic propagator")
		self.f=f
		self.ref1=ref1
		self.ref2=ref2
		self.ref3=ref3
		self.mapName=mapName
		self.idString="--"+self.mapName+"--ref_"+str(ref1)+"_"+str(ref2)+"_"+str(ref3)+self.idString
		print(self.idString)
		print(" ->filling propagation matrix")
		self.propagator=np.array([[self.f(self.M,n,m,self.ref1,self.ref2,self.ref3) for n in range(self.M)] for m in range(self.M)])
		if inverseFourierNeeded:
			print(" ->filling inverse fourier matrix")
			self.inv=np.array([[1/mt.sqrt(self.M)*cmt.exp(2*mt.pi*1j/self.M*(n+1/2)*(m+1/2)) for n in range(self.M)] for m in range(self.M)])      
			print(" ->matrix product")
			self.propagator=self.inv.dot(self.propagator)

	def makePropagator3R_arbitraryPhases ( self, f, mapName, ref1, ref2, ref3, chi_q, chi_p, inverseFourierNeeded=True):
		print("\n\nsetting up the basic propagator")
		self.f=f
		self.ref1=ref1
		self.ref2=ref2
		self.ref3=ref3
		self.mapName=mapName
		self.idString="--"+self.mapName+"--ref_"+str(ref1)+"_"+str(ref2)+"_"+str(ref3)+self.idString
		self.idString="--chi_q_"+str(chi_q)+"--chi_p_"+str(chi_p)+self.idString
		print(self.idString)
		print(" ->filling propagation matrix")
		self.propagator=np.array([[self.f(self.M,n,m,self.ref1,self.ref2,self.ref3,chi_q,chi_p) for n in range(self.M)] for m in range(self.M)])
		if inverseFourierNeeded:
			print(" ->filling inverse fourier matrix")
			self.inv=np.array([[1/mt.sqrt(self.M)*cmt.exp(2*mt.pi*1j/self.M*(n+chi_q)*(m+chi_p)) for n in range(self.M)] for m in range(self.M)])      
			print(" ->matrix product")
			self.propagator=self.inv.dot(self.propagator)

	def setReflectivities( self, r, ElementByElement=True ):
		'''sometimes setting up the (partial) holes outside the definition of the map is better.
		'''
		self.r=r
		print("filling projection matrix")
		self.projector=np.array([[self.r(self.M,n,m,self.ref) for n in range(self.M)] for m in range(self.M)])
		if ElementByElement:
			print(" ->element-by-element product")
			self.propagator=self.propagator*self.projector
		else:
			print(" ->matrix product")
			self.propagator=self.propagator.dot(self.projector)

	def setReflectivities3R( self, r, ElementByElement=True ):
		'''sometimes setting up the (partial) holes outside the definition of the map is better.
		'''
		self.r=r
		print("filling projection matrix")
		self.projector=np.array([[self.r(self.M,n,m,self.ref1, self.ref2, self.ref3) for n in range(self.M)] for m in range(self.M)])
		if ElementByElement:
			print(" ->element-by-element product")
			self.propagator=self.propagator*self.projector
		else:
			print(" ->matrix product")
			self.propagator=self.propagator.dot(self.projector)
   
