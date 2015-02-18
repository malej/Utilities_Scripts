#!/usr/bin/env python
import sys
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
    fileF2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
        str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'
    fileW2D = '/Users/mattmalej/Dropbox/AveragedData_for_WC/case-'+\
        str(args.case)+'/averagedHalfSpectrum_from100sims_Cartesian_F2D.bin'

    case = {1:'data_beta_0p35',
            2:'data_eps_0p08_beta_0p35',
            3:'data_beta_0p7',
            4:'data_eps_0p13_beta_0p7',
            #0:'/Volumes/1TB_HD_2/Simulations_Backup/Fully2D_200_sim/'}
            0:'/Users/mattmalej/phaseWave/hosmModule/temp/'}
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


def sliceSpectrum(data, Nx, Ny, nSnaps):
    """ Function that converts 1D array into full 3D of [Nx,Ny,numOfSnapshots. """
    spectrum = np.zeros((Nx,Ny,nSnaps))
    temp = np.zeros((Nx,Ny))
    chunkSize = Nx*Ny

    for k in range(nSnaps):
        for j in range(Ny):
            spectrum[:,j,k] = data[(k*chunkSize)+(j*Nx):(k*chunkSize)+(j+1)*Nx]
    return spectrum


def plotSpectrum(spec, kx, ky, nSnaps, case, model):
    """ Function that plots the spectra with Matplotlib. """
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
    
    #plt.show()


if __name__=="__main__":
    checkSpectra()
