#!/usr/bin/env python
import sys
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def checkSpectra():
    """Function that reads in spectra and and checks its consisency."""

    # parsing the input arguments
    parser = argparse.ArgumentParser(description='Postprocessed spectra checking and plotting.')
    parser.add_argument('-p','--plot', action='store_true',
                        help='plot the initial and final spectra (default: no plotting)')
    parser.add_argument('-c','--case', type=int, default=1,
                        help='simulation case:1,2,3,4 to be read in (default: 1)')
    args = parser.parse_args()
    print 'Input arguments are: ', args

    # simulation directories
    #fileF2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
    #    str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'
    #fileW2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
    #    str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'
    #<debug>
    fileF2D = 'F2D.bin'
    fileW2D = 'W2D.bin'
    #</debug>

    case = {1:'data_beta_0p35',
            2:'data_eps_0p08_beta_0p35',
            3:'data_beta_0p7',
            4:'data_eps_0p13_beta_0p7'}
    whichCase = args.case

    # dimensions
    Nx = 256
    Ny = 256
    numOfSnapshots = 16
    contentSize = Nx*Ny*numOfSnapshots
    spectrumF2D = np.zeros((Nx,Ny,numOfSnapshots))
    spectrumW2D = np.zeros((Nx,Ny,numOfSnapshots))

     # printing without '\n'
    sys.stdout.write('Reading & Unpacking Data Files ')

    # reading in the spectra
    finF2D = open(fileF2D, 'rb')
    finW2D = open(fileW2D, 'rb')
    contentF2D = finF2D.read(8*contentSize) # 8, because doubles are 8 bytes
    contentW2D = finW2D.read(8*contentSize) # 8, because doubles are 8 bytes
    dataF2D = struct.unpack('d'*contentSize,contentF2D)
    dataW2D = struct.unpack('d'*contentSize,contentW2D)
    sys.stdout.write('...')
    sys.stdout.flush()  # flush the stdout buffer
    finF2D.close()
    finW2D.close()
    spectrumF2D = sliceSpectrum(dataF2D,Nx,Ny,numOfSnapshots)
    spectrumW2D = sliceSpectrum(dataW2D,Nx,Ny,numOfSnapshots)
    print "DONE"

    x=np.arange(Nx)
    y=np.arange(-Ny/2,Ny/2)

    # plotting ensemble average spectra in recti-linear coordinates
    if args.plot == True:
        sys.stdout.write('Plotting...')
        sys.stdout.flush() #flush stdout buffer
        plotSpectrum(spectrumF2D, 
                     x,
                     y,
                     numOfSnapshots,
                     model = 'Fully 2D')
        plotSpectrum(spectrumW2D, 
                     x,
                     y,
                     numOfSnapshots,
                     model = 'Weakly 2D')
        print('DONE')


def sliceSpectrum(data, Nx, Ny, nSnaps):
    """ Function that converts 1D array into full 3D of [Nx,Ny,numOfSnapshots. """
    spectrum = np.zeros((Nx,Ny,nSnaps))
    temp = np.zeros((Nx,Ny))
    chunkSize = Nx*Ny

    for k in range(nSnaps):
        for j in range(Ny):
            spectrum[:,j,k] = data[(k*chunkSize)+(j*Nx):(k*chunkSize)+(j+1)*Nx]
    return spectrum


def plotSpectrum(spec, kx, ky, nSnaps, model):
    """ Function that plots the spectra with Matplotlib. """
    Nx = kx.size
    Ny = ky.size

    [kxx,kyy] = np.meshgrid(kx,ky)
    maxScale = spec[:,:,0].max()
    z_init = np.transpose(spec[:,:,0]/maxScale)
    z_final = np.transpose(spec[:,:,-1]/maxScale) 

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(kxx,kyy, z_init, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(kxx,kyy, z_init, zdir='z', offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel('Kx')
    ax.set_ylabel('Ky')
    ax.set_zlabel('Initial Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(kxx,kyy, z_final, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(kxx,kyy, z_final, zdir='z', offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel('Kx')
    ax.set_ylabel('Ky')
    ax.set_zlabel('Final Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)
    
    plt.show()


if __name__=="__main__":
    checkSpectra()
