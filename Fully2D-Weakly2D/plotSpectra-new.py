#!/usr/bin/env python
import sys
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import RectBivariateSpline


def readSpectra():
    """Function that read in spectra and computes an ensemble average."""

    # parsing the input arguments
    parser = argparse.ArgumentParser(description='Wave spectra plotting and saving from surface elevations.')
    parser.add_argument('-p','--plot', action='store_true',
                        help='plot the initial and final spectra (default: no plotting)')
    parser.add_argument('-s','--save', action='store_true',
                         help='save the rectilinear half spectrum to binary file (default: no saving)')
    parser.add_argument('-f','--files', type=int, default=1, 
                        help='number of input files to be read in (default: 1)')
    parser.add_argument('-c','--case', type=int, default=1,
                        help='simulation case:1,2,3,4 to be read in (default: 1)')
    args = parser.parse_args()
    print 'Input arguments are: ', args

    # simulation directories
    if args.case in (1,3):
        dirFully2D = '/Volumes/1TB_HD_2/Simulations_Backup/Fully2D_200_sim/' # for cases 1,3
        dirWeakly2D = '/Volumes/1TB_HD_2/Simulations_Backup/Weakly2D_200_sim/' # for cases, 1,3
    else:
        dirFully2D = '/Volumes/1TB_HD_2/Simulations_Backup/Fully2D_100_sim/' # for cases 2,4
        dirWeakly2D = '/Volumes/1TB_HD_2/Simulations_Backup/Weakly2D_100_sim/' # for cases 2,4

    case = {1:'data_beta_0p35',
            2:'data_eps_0p08_beta_0p35',
            3:'data_beta_0p7',
            4:'data_eps_0p13_beta_0p7',
            5:'data_eps_0p13_beta_0p7_900sec'}

    whichCase = args.case
    numOfDataFiles = args.files

    # dimensions
    Nx = 512
    Ny = 256

    if whichCase == 5:
        numOfSnapshots = 91
    else:
        numOfSnapshots = 16

    contentSize = Nx*Ny*numOfSnapshots
    spectrumF2D = np.zeros((Nx,Ny,numOfSnapshots))
    spectrumW2D = np.zeros((Nx,Ny,numOfSnapshots))

    # printing without '\n'
    sys.stdout.write('Reading & Unpacking Data Files ')

    for i in range(numOfDataFiles):
        finF2D = open(dirFully2D+case[whichCase]+'/'+'surface'+str(i+1)+'.bin', 'rb')
        finW2D = open(dirWeakly2D+case[whichCase]+'/'+'surface'+str(i+1)+'.bin', 'rb')
        contentF2D = finF2D.read(8*contentSize) # 8, because doubles are 8 bytes
        contentW2D = finW2D.read(8*contentSize) # 8, because doubles are 8 bytes
        dataF2D = struct.unpack('d'*contentSize,contentF2D)
        dataW2D = struct.unpack('d'*contentSize,contentW2D)
        sys.stdout.write('.')
        sys.stdout.flush()  # flush the stdout buffer
        finF2D.close()
        finW2D.close()
        spectrumF2D = spectrumF2D + extractSpectrum(dataF2D,Nx,Ny,numOfSnapshots)
        spectrumW2D = spectrumW2D + extractSpectrum(dataW2D,Nx,Ny,numOfSnapshots)
    print "DONE"

    sys.stdout.write('Processing ...')
    sys.stdout.flush()  # flush the stdout buffer

    spectrumF2D = spectrumF2D/numOfDataFiles # for ensemble average
    spectrumW2D = spectrumW2D/numOfDataFiles

    rotatedHalfSpectrumF2D = np.zeros((Nx/2,Ny,numOfSnapshots))
    rotatedHalfSpectrumW2D = np.zeros((Nx/2,Ny,numOfSnapshots))
    rotatedHalfSpectrumF2D[0:Nx/2,0:Ny/2,:] = spectrumF2D[0:Nx/2,Ny/2:Ny,:]
    rotatedHalfSpectrumW2D[0:Nx/2,0:Ny/2,:] = spectrumW2D[0:Nx/2,Ny/2:Ny,:]
    rotatedHalfSpectrumF2D[0:Nx/2,Ny/2:Ny,:] = spectrumF2D[0:Nx/2,0:Ny/2,:]
    rotatedHalfSpectrumW2D[0:Nx/2,Ny/2:Ny,:] = spectrumW2D[0:Nx/2,0:Ny/2,:]

    x_for_interp = np.arange(0,Nx/2)
    y_for_interp = np.arange(-Ny/2,Ny/2)

    [r,theta,spectrumPolarF2D] = cart2polar(x = x_for_interp,
                                            y = y_for_interp,
                                            z = rotatedHalfSpectrumF2D,
                                            nSnaps = numOfSnapshots)
    [r,theta,spectrumPolarW2D] = cart2polar(x = x_for_interp,
                                            y = y_for_interp,
                                            z = rotatedHalfSpectrumW2D,
                                            nSnaps = numOfSnapshots)
    x=np.arange(Nx)
    y=np.arange(Ny)
    print('DONE')

    # plotting ensemble average spectra in recti-linear coordinates
    if args.plot:
        sys.stdout.write('Plotting...')
        sys.stdout.flush() #flush stdout buffer
        plotSpectrum(rotatedHalfSpectrumF2D, 
                     x_for_interp,
                     y_for_interp, 
                     numOfSnapshots,
                     model = 'Fully 2D')
        plotSpectrum(rotatedHalfSpectrumW2D, 
                     x_for_interp,
                     y_for_interp, 
                     numOfSnapshots,
                     model = 'Weakly 2D')
        plotPolarSpectrum(spectrumPolarF2D, r, theta, numOfSnapshots, model = 'Fully 2D')
        plotPolarSpectrum(spectrumPolarW2D, r, theta, numOfSnapshots, model = 'Weakly 2D')
        print('DONE')

    # saving ensemble average spectra to binary data file
    if args.save:
        sys.stdout.write('Saving spectrum to binary data file .')
        saveSpectrum(rotatedHalfSpectrumF2D,
                     rotatedHalfSpectrumW2D,
                     dirs={'Fully2D':dirFully2D+case[whichCase]+'/',
                           'Weakly2D':dirWeakly2D+case[whichCase]+'/'},
                     grid='Cartesian',
                     numSims = numOfDataFiles)
        saveSpectrum(spectrumPolarF2D,
                     spectrumPolarW2D,
                     dirs={'Fully2D':dirFully2D+case[whichCase]+'/',
                           'Weakly2D':dirWeakly2D+case[whichCase]+'/'},
                     grid='Polar',
                     numSims=numOfDataFiles)
        print('DONE')


def saveSpectrum(spectrumF2D, spectrumW2D, dirs, grid, numSims):
    """Module that saves abs(spectrum) to a binary file in [Nx x Ny x nSnaps] 
    format. It is scaled max value of all spectra snapshots."""
    [Nx,Ny,nSnaps] = spectrumF2D.shape
    
    fout = [open(dirs['Fully2D']+'averagedHalfSpectrum_from'+str(numSims)+'sims_'+grid+'_F2D.bin', 'wb'),
            open(dirs['Weakly2D']+'averagedHalfSpectrum_from'+str(numSims)+'sims_'+grid+'_W2D.bin', 'wb')]

    # to optimize recall that Python is row-major
    for k in range(nSnaps):
        sys.stdout.write('.')
        sys.stdout.flush()   #flush the stdout buffer
        for j in range(Ny):
            fout[0].write(abs(spectrumF2D[:,j,k]))
            fout[1].write(abs(spectrumW2D[:,j,k]))
            # NOTE: the modulus somehow is not taken over 3D or 2D array
            #       (termwise), hence we're doing it here in the loop!

    # close output file buffers
    for fileHandle in fout:
        fileHandle.close()


def cart2polar(x,y,z,nSnaps):
    """ Function that interpolates a surface/spectrum in (x,y,z(x,y))
    using Scipy's RectBivariateSpling to polar coordinate (r,theta,z(r,theta)).

    INPUT:  x, y, z(x,y), number-of-time-snapshots-to-interpolate
    RETURNS: r,theta, z(r,theta)"""

    rSize = thetaSize = y.size #x.size is 2*y.size - prefer a squre grid
    rInit = 0
    rFinal = np.sqrt(x[-1]**2+y[-1]**2)
    thetaInit = -np.pi/2
    thetaFinal = np.pi/2
    r = np.linspace(rInit,rFinal,rSize)
    theta = np.linspace(thetaInit,thetaFinal,thetaSize)

    # inerpolation
    newZ = np.zeros((rSize,thetaSize,nSnaps))
    for i in range(nSnaps):
        sys.stdout.write('.')
        sys.stdout.flush() # flushing the output buffer
        zInterpolant = RectBivariateSpline(x,y,z[:,:,i]) # using only square grid
        for j in range(rSize):
            for k in range(thetaSize):
                newZ[j,k,i] = zInterpolant(r[j]*np.cos(theta[k]),
                                           r[j]*np.sin(theta[k]))

    return r,theta,newZ

def extractSpectrum(data3D, Nx, Ny, nSnaps):
    """ Function that converts surface profile to its spectra ==> abs(A)
    from a full 3D array [Nx,Ny,nSnaps]"""
    spectrum = np.zeros((Nx,Ny,nSnaps))
    for k in range(nSnaps):
        spectrum[:,:,k] = np.abs(np.fft.fft2(data3D[:,:,k]))
    return spectrum

#def extractSpectrum(data, Nx, Ny, nSnaps):
#    """ Function that converts surface profile to its spectra ==> abs(A). """
#    spectrum = np.zeros((Nx,Ny,nSnaps))
#    temp = np.zeros((Nx,Ny))
#    chunkSize = Nx*Ny
#
#    for k in range(nSnaps):
#        for j in range(Ny): # convert into 2d array
#            temp[:,j] = data[(chunkSize*k)+(j*Nx):(chunkSize*k)+(j+1)*Nx]
#        spectrum[:,:,k] = np.abs(np.fft.fft2(temp))        
#    return spectrum


def plotPolarSpectrum(halfSpectrum, r, theta, nSnaps, model):
    """ Function that plot half of the interpolated polar spectrum. """
    # spectrum = R*exp[i*theta]
    spec = np.abs(halfSpectrum)
    rSize = r.size
    thetaSize = theta.size

    [rr,tt] = np.meshgrid(r,theta)
    maxScale = spec[:,:,0].max()
    z_init = np.transpose(spec[:,:,0]/maxScale)
    z_final = np.transpose(spec[:,:,-1]/maxScale)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(rr, tt, z_init, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(rr, tt, z_init, rstride=8, offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel(r'$r = \sqrt{x^2+y^2}$')
    ax.set_ylabel(r'$\theta = \tan^{-1}(y/x)$')
    ax.set_zlabel('Polar Initial Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(rr, tt, z_final, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(rr, tt, z_final, offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel(r'$r = \sqrt{x^2+y^2}$')
    ax.set_ylabel(r'$\theta = \tan^{-1}(y/x)$')
    ax.set_zlabel('Polar Final Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)
   
    plt.show()


def plotSpectrum(spectrum, x, y, nSnaps, model):
    """ Function that plots the spectra with Matplotlib. """
    # spectrum = A*exp[i*(kx*x+ky*y)]
    spec = np.abs(spectrum)
    Nx = x.size
    Ny = y.size

    [xx,yy] = np.meshgrid(x,y)
    maxScale = spec[:,:,0].max()
    z_init = np.transpose(spec[:,:,0]/maxScale)
    z_final = np.transpose(spec[:,:,-1]/maxScale) 

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(xx,yy, z_init, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(xx,yy, z_init, zdir='z', offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel('Kx')
    ax.set_ylabel('Ky')
    ax.set_zlabel('Initial Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)

    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.plot_surface(xx,yy, z_final, rstride=8, cstride=8, alpha=0.3)
    ax.contourf(xx,yy, z_final, zdir='z', offset=-0.5, cmap=cm.coolwarm)
    ax.set_xlabel('Kx')
    ax.set_ylabel('Ky')
    ax.set_zlabel('Final Spectrum')
    ax.set_title(model)
    ax.set_zlim(-0.5,1.0)
    
    plt.show()


if __name__=="__main__":
    readSpectra()
