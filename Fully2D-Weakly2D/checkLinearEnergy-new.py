# Checking linearized energy ... to mimic WY results
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def readFile():
    #path      = '/Users/mattmalej/Desktop/Check_WY_Energy/'
    path      = '/Users/mattmalej/test-RHS1-conserv-code/'
    inputFile = 'snapshots_surface.bin'
    fullPath  = path+inputFile

    Kx = 512
    Ky = 256
    Snaps = 151
    chunkSize = Kx*Ky*Snaps

    spectrum = np.zeros([Kx,Ky,Snaps],dtype=np.float64)

    try:
        fin = open(fullPath, "rb")
        temp = np.fromfile(fin, dtype=np.float64, count=chunkSize)
        fin.close()
        print "File ", inputFile, " READ IN ... OK "
    except:
        print " Cannot READ the file: ", inputFile, " ... EXCEPTION"

    # Populate spectrum 
    count = 0
    for k in range(Snaps):
        for j in range(Ky):
            for i in range(Kx):
                spectrum[i,j,k] = temp[count]
                count +=1

    return spectrum


def calculateEnergy(spectrum_F2D):
    [kx,ky,nSnaps] = spectrum_F2D.shape

    initEnergy = np.sum( (spectrum_F2D[:,:,0]**2) )

    print "Initial Energy is: ", initEnergy, '\n'

    for i in range(nSnaps):
        energy =  np.sum( (spectrum_F2D[:,:,i]**2) ) 
        relativeDiff = (initEnergy - energy)/initEnergy
        print "Energy at ", i*100, " seconds is: ", energy
        print "Relative Difference in Energy is: ", relativeDiff, '\n'

#####################################
# Main... 
spectrum_F2D = readFile()
specMax = np.max(np.max(np.max(spectrum_F2D[:,:,0])))
spectrum_F2D = spectrum_F2D/specMax
print specMax

calculateEnergy(spectrum_F2D)

#streamlines()

# Plot 1D Spectrum for check
Kx = 129
Ky = 129
Snaps = 10

fig = plt.figure()
#ax = fig.gca(projection='3d')
x = np.arange(Kx)
y = np.arange(Ky)
[xx,yy] = np.meshgrid(x,y)

plt.plot(x,spectrum_F2D[:,Ky-1,0],x,spectrum_F2D[:,Ky-1,1],x,spectrum_F2D[:,Ky-1,9])
#surf = ax.plot_surface(xx,yy,z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

#ax.set_zlim(0,specMax)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
