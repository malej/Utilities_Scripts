#!/usr/bin/env python
import sys
import os
import struct
import argparse
import numpy as np
import matplotlib
#matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from plot_spectra import extractSpectrum 

def parseInput():
    """Function that parses input from command line."""
    parser = argparse.ArgumentParser(description='Postprocessed spectra checking and plotting.')
    parser.add_argument('-pa','--printArgs', action='store_true',
                        help='print to screen the list of input arguments and state (default: no printing)')
    parser.add_argument('-p','--plot', action='store_true',
                        help='plot the initial and final spectra (default: no plotting)')
    parser.add_argument('-c','--case', type=int, default=1,
                        help='simulation case:1,2,3,4 to be read in (default: 1)')
    parser.add_argument('-x','--Nx', type=int, default=512,
                        help='number of points in the main x-direction (default: 512)')
    parser.add_argument('-y','--Ny', type=int, default=256,
                        help='number of points in the transverse y-direction (default: 256)')
    parser.add_argument('-s','--snaps', type=int, default=151,
                        help='number of snapshots to be read in (default: 151)')
    parser.add_argument('-d','--dir', type=str, default=os.environ['PWD'],
                        help='file path directory (default: current directory)')
    parser.add_argument('-f','--filePrefix', type=str, default='snapshots_surface',
                        help='file name prefix; e.g., surface_snapshots if file is surface_snapshots99.bin (default: snapshots_surface)')
    args = parser.parse_args()

    if args.printArgs:
        print 'Input arguments are: ', args
    return args


def readSurface(args):
    """Function that read in surface elevation from Fully/Weakly 2D run."""
    path = args.dir
    fileName = args.filePrefix+'.bin'
    
    # dimensions
    Nx = args.Nx
    Ny = args.Ny
    numOfSnapshots = args.snaps
    contentSize = Nx*Ny*numOfSnapshots

     # printing without '\n'
    sys.stdout.write('Reading & Unpacking SURFACE ELEVATION Data Files ')

    # reading in the spectra
    fileHandle = open(fileName, 'rb')
    content = fileHandle.read(8*contentSize) # 8, because doubles are 8 bytes
    data = struct.unpack('d'*contentSize,content)
    sys.stdout.write('...')
    sys.stdout.flush()  # flush the stdout buffer
    fileHandle.close()

    freeSurface = sliceData(data,Nx,Ny,numOfSnapshots)
    print ('DONE')
    return freeSurface


def readPotential(args):
    """Function that read in velocity potential from Fully/Weakly 2D run."""
    path = args.dir
    fileName = 'snapshots_potential.bin'
    
    # dimensions
    Nx = args.Nx
    Ny = args.Ny
    numOfSnapshots = args.snaps
    contentSize = Nx*Ny*numOfSnapshots

     # printing without '\n'
    sys.stdout.write('Reading & Unpacking VELOCITY POTENTIAL Data Files ')

    # reading in the spectra
    fileHandle = open(fileName, 'rb')
    content = fileHandle.read(8*contentSize) # 8, because doubles are 8 bytes
    data = struct.unpack('d'*contentSize,content)
    sys.stdout.write('...')
    sys.stdout.flush()  # flush the stdout buffer
    fileHandle.close()

    velocityPotential = sliceData(data,Nx,Ny,numOfSnapshots)
    print ('DONE')
    return velocityPotential


def readRHS1(args):
    """Function that read in RHS1 (right-hand-side of d\zeta/dt from Fully/Weakly 2D run."""
    path = args.dir
    fileName = 'snapshots_RHS1.bin'
    
    # dimensions
    Nx = args.Nx
    Ny = args.Ny
    numOfSnapshots = args.snaps
    contentSize = Nx*Ny*numOfSnapshots

     # printing without '\n'
    sys.stdout.write('Reading & Unpacking RHS1 (right-hand-side of d/dt{zeta}) Data Files ')

    # reading in the spectra
    fileHandle = open(fileName, 'rb')
    content = fileHandle.read(8*contentSize) # 8, because doubles are 8 bytes
    data = struct.unpack('d'*contentSize,content)
    sys.stdout.write('...')
    sys.stdout.flush()  # flush the stdout buffer
    fileHandle.close()

    RHS1 = sliceData(data,Nx,Ny,numOfSnapshots)
    print ('DONE')
    return RHS1


def setWaveNumbers(args):
    """Function which sets the appropriate range of wavenumber in Fourier space."""
    Nx = args.Nx
    Ny = args.Ny
    kx = np.arange(Nx)
    ky = np.arange(-Ny/2,Ny/2)
    return [kx,ky] 


def checkSpectra(args):
    """Function that reads in spectra and and checks its consisency."""

    # simulation directories
    fileF2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
        str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'
    fileW2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
        str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'

    case = {1:'data_beta_0p35',
            2:'data_eps_0p08_beta_0p35',
            3:'data_beta_0p7',
            4:'data_eps_0p13_beta_0p7',
            #5:'/Volumes/1TB_HD_2/Simulations_Backup/Fully2D_200_sim/'}
            0:'/Users/mattmalej/test-RHS1-conserv-code'}
    whichCase = args.case
    #fileAux = case[args.case]+'data_beta_0p35/'+'surface1.bin' #needs 512x256
    fileAux = case[args.case]+'snapshots_surface.bin'

    # dimensions
    Nx = 256#512
    Ny = 256
    numOfSnapshots = 151#16
    contentSize = Nx*Ny*numOfSnapshots
    spectrumF2D = np.zeros((Nx,Ny,numOfSnapshots))
    spectrumW2D = np.zeros((Nx,Ny,numOfSnapshots))

     # printing without '\n'
    sys.stdout.write('Reading & Unpacking Data Files ')

    # reading in the spectra
    if args.case != 0:
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
    else:
        finAux = open(fileAux, 'rb')
        contentAux = finAux.read(8*contentSize) # 8, because doubles are 8 bytes
        dataAux = struct.unpack('d'*contentSize,contentAux)
        sys.stdout.write('...')
        sys.stdout.flush()  # flush the stdout buffer
        finAux.close()
        freeSurface = sliceSpectrum(dataAux,Nx,Ny,numOfSnapshots)
        plotSurface(freeSurface,Nx,Ny,numOfSnapshots)
        spectrumAux = extractSpectrum(dataAux,Nx,Ny,numOfSnapshots)
    print "DONE"

    x=np.arange(Nx)
    y=np.arange(-Ny/2,Ny/2)

    # plotting ensemble average spectra in recti-linear coordinates
    if args.plot == True:
        sys.stdout.write('Plotting...')
        sys.stdout.flush() #flush stdout buffer
        if args.case != 0:
            plotSpectrum(spectrumF2D, 
                         x,
                         y,
                         numOfSnapshots,
                         whichCase,
                         model = 'Fully 2D')                         
            plotSpectrum(spectrumW2D, 
                         x,
                         y,
                         numOfSnapshots,
                         whichCase,
                         model = 'Weakly 2D')
        else:
            plotSpectrum(spectrumAux,
                         x,
                         y,
                         numOfSnapshots,
                         whichCase,
                         model = 'F2D with Integrating Factor')
        print('DONE')

def plotSurface(surface, Nx, Ny, nSnaps):
    """ Function that plots a free-surface. """
    Lx = Ly = 46.8392997519448
    [xx,yy] = np.meshgrid(np.linspace(0.0,Lx,Nx),np.linspace(0.0,Ly,Ny))
    #maxScale = surface[:,:,0].max()
    z_init = surface[:,:,0].transpose()
    
    fig = plt.figure()#(figsize=plt.figaspect(0.5))
    #ax = fig.add_subplot(1,2,1, projection='3d')
    ax = Axes3D(fig)#fig.gca(projection='3d')
    rstride=Nx/32
    cstride=Ny/32
    surf = ax.plot_wireframe(xx[0::rstride,0::cstride],yy[0::rstride,0::cstride],z_init[0::rstride,0::cstride])
    #surf = ax.plot_wireframe(xx,yy,z_init, rstride=32,cstride=64)#, alpha=0.9, cmap=cm.gray_r,
                           #linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    contourLevels = np.arange(-0.12,0.12,0.05)
    contour = ax.contourf(xx,yy, z_init, zdir='z', offset=-0.3, cmap=cm.coolwarm,
                          contour_levels=contourLevels)
    v = np.linspace(-0.16, 0.16, 9, endpoint=True)
    cb = fig.colorbar(contour, shrink=0.4, aspect=5, ticks=v)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    #ax.set_zlabel('Initial Surface Elevation')
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    ax.set_title("Free Surface at ==> t = 0 sec")
    ax.set_zlim(-0.3,0.3)
    fig.set_size_inches(7,4)
    plt.savefig("surface0000.png", transparent = False, dpi=200)

    print "Saving Figures (free-surface) ..."
    for i in range(1,nSnaps):
        z_final = surface[:,:,i].transpose()

        fig = plt.figure()#(figsize=plt.figaspect(0.5))
        #ax = fig.add_subplot(1,2,2, projection='3d')
        ax = Axes3D(fig)#fig.gca(projection='3d')
        rstride=Nx/32
        cstride=Ny/32
        surf = ax.plot_wireframe(xx[0::rstride,0::cstride],yy[0::rstride,0::cstride],z_final[0::rstride,0::cstride])
        #surf = ax.plot_wireframe(xx,yy,z_init, rstride=32,cstride=64)#, alpha=0.9, cmap=cm.gray_r,
                           #linewidth=0, antialiased=True)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        contour = ax.contourf(xx,yy, z_final, zdir='z', offset=-0.3, cmap=cm.coolwarm,
                              countour_levels=contourLevels)
        cb = fig.colorbar(contour, shrink=0.4, aspect=5, ticks=v)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.set_zlabel('Final Surface Elevation')
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        ax.set_title('Free Surface at ==> t = '+str(i)+' sec')
        ax.set_zlim(-0.3,0.3)
        fig.set_size_inches(7,4)
        fileName = 'surface%04d.png' %i
        plt.savefig(fileName, transparent = False, dpi=200)   
        #plt.show()


def sliceData(data, Nx, Ny, nSnaps):
    """ Function that converts 1D array into full 3D of [Nx,Ny,numOfSnapshots]. """
    data3D = np.zeros((Nx,Ny,nSnaps))
    temp = np.zeros((Nx,Ny))
    chunkSize = Nx*Ny

    for k in range(nSnaps):
        for j in range(Ny):
            data3D[:,j,k] = data[(k*chunkSize)+(j*Nx):(k*chunkSize)+(j+1)*Nx]
    return data3D


def plotSpectrum(spec, kx, ky, args, model):
    """ Function that plots the spectra with Matplotlib. """
    case = args.case
    nSnaps = args.snaps

    Nx = kx.size
    Ny = ky.size

    [kxx,kyy] = np.meshgrid(kx,ky)
    maxScale = spec[:,:,0].max()
    z_init = np.transpose(spec[:,:,0]/maxScale)
    z_final = np.transpose(spec[:,:,-1]/maxScale)
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.plot_surface(kxx,kyy, z_init, rstride=4, cstride=4, alpha=0.3)
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
    
    if args.plot:
        plt.show()


def trap2D(data,args):
    """ Function that returns a 2D trapezoidal integral array of snapshots."""
    Lx = Ly =  46.8392997519448
    x = np.linspace(0.0, Lx, args.Nx)
    y = np.linspace(0.0, Ly, args.Ny)
    dx = Lx/args.Nx
    dy = Ly/args.Ny

    # Creating Trapzoidal weight matrix
    #  |---------------------|
    #  | 1 2 2 2 ... 2 2 2 1 |
    #  | 2 4 4 4 ... 4 4 4 2 |
    #  | 2 4 4 4 ... 4 4 4 2 |
    #  | . . . . ... . . . . |
    #  | 2 4 4 4 ... 4 4 4 2 |
    #  | 2 4 4 4 ... 4 4 4 2 |
    #  | 1 2 2 2 ... 2 2 2 1 |
    #  |---------------------|
    weights = np.ones((args.Nx,args.Ny))
    # set perimeter 2's and interior weights 4's
    weights[0,1:-1] = 2.0
    weights[-1,1:-1] = 2.0
    weights[1:-1,0] = 2.0
    weights[1:-1,-1] = 2.0
    weights[1:-1,1:-1] = 4.0

    integrand = weights*data
    sum = 0.0
    for i in range(args.Nx):
        for j in range(args.Ny):
            sum = sum + integrand[i,j]

    integral = dx*dy/4.0 * sum 
    # returning trapezoidal 2D integral of input 'data' 
    return integral


def computeRealEnergy(surface,potential,RHS1,args):
    """Function that computes real energy from theory."""
    nSnaps = args.snaps
    gv = 9.8 # gravitational acceleration

    print "\n ENERGY COMPUTATION FROM => phi*RHS1 + gv*zeta^2 "
    print "---------------------------------------------------"
    # buffers
    energySnaps = np.zeros(nSnaps)
    for i in range(nSnaps):
        energyIntegrand =  potential[:,:,i]*RHS1[:,:,i] + gv*surface[:,:,i]**2
        energySnaps[i] = trap2D(energyIntegrand,args)
        print "time t = ", i, " sec ==> Energy is ", energySnaps[i]
    
    relDiff = np.abs(energySnaps[-1] - energySnaps[0])/ energySnaps[-1]
    print " ENERGY RELATIVE DIFFERENCE abs(final-initial)/final is: ", relDiff
    return energySnaps


if __name__=="__main__":
    args = parseInput()
    surface = readSurface(args)
    potential = readPotential(args)
    RHS1 = readRHS1(args)

    realEnergy = computeRealEnergy(surface,potential,RHS1,args)

    spectrum = extractSpectrum(surface,
                               args.Nx,
                               args.Ny,
                               args.snaps)

    model = 'Fully2D w/o integrating factor'
    [kx,ky] = setWaveNumbers(args)
    plotSpectrum(spectrum, 
                 kx, 
                 ky, 
                 args, 
                 model) 

    #checkSpectra(inputArguments, surface)
