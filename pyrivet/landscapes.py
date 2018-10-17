import numpy as np
import matplotlib.pyplot as plt
from pyrivet import barcode


"""
Landscape classes
"""

class Landscape(object):
    """ A single landscape for a chosen k """

    def __init__(self, index, critical_points):
        self.index = index
        self.critical_points = critical_points # an nx2 array

    def __repr__(self):

        return "Landscape(%d,%s)" % (self.index, self.critical_points)

    def plot_landscapes(self):
        """ Plots a single landscape"""
        n = np.shape(self.critical_points)[0]
        x = self.critical_points[1:n,0]
        y = self.critical_points[1:n,1]
        plt.plot(x,y)
        plt.show()

    def evaluate(self, xvalue):
        """ Returns the landscape value at a queried x value """

        return np.interp(xvalue,self.critical_points[1:n,0],self.critical_points[1:n,1])


class Landscapes(object):
    """ Collection of non zero landscapes """

    def __init__(self, landscapes=None):
        if landscapes is None:
            landscapes = []
        self.landscapes = landscapes

    def __repr__(self):

        return "Landscapes(%s)" % self.landscapes

    def plot_landscapes(self):
        """ Plots the landscapes in the collection to a single axes"""
        for k in range(len(self.landscapes)):
            n = np.shape(self.landscapes[k].critical_points)[0]
            x = self.landscapes[k].critical_points[1:n,0]
            y = self.landscapes[k].critical_points[1:n,1]
            plt.plot(x,y)
        plt.show()


class Multiparameter_Landscape(object):
    """ A single multiparameter landscape """

    def __init__(self, index, bounds, zvalues):
        self.index = index # Landscape index
        self.bounds = bounds # Parameter subspace over which zvalues belong, in the form [[xmin,xmax],[ymin,ymax]]
        self.zvalues = zvalues # Landscape values

    def __repr__(self):
        return "Multiparameter Landscape(%d,%s)" % (self.index, self.zvalues)


def compute_landscapes(barcode,maxind=None):
    """ Computes the collection of persistence landscapes associated to a barcode up to index maxind
        using the algorithm set out in Bubenik + Dlotko.
    :param barcode: A barcode object
    :param maxind: The maximum index landscape to calculate
    """
    L = []
    barcode = barcode.expand()
    barcode = barcode.to_array()
    # sort by increasing birth and decreasing death
    sortedbarcode = barcode[np.lexsort((-barcode[:,1],barcode[:,0]))]
    k = 1 # initialise index for landscape
    if maxind is None:
        while(np.sum(sortedbarcode[:,2])>0):
            p = 0 # pointer to position in barcode
            [b,d,m],sortedbarcode = pop(sortedbarcode,p)
            critical_points = np.array([[float("-inf"),0],[b,0],[(b+d)/2,(d-b)/2]])
            while(critical_points[-1,0]!=float("inf")): # check last row is not trivial
                if (np.shape(sortedbarcode)[0]==0):
                    critical_points = np.vstack([critical_points,[[d,0],[float("inf"),0]]])
                    L.append(Landscape(k,critical_points))
                elif(d >= np.max(sortedbarcode[:,1])):
                    critical_points = np.vstack([critical_points,[[d,0],[float("inf"),0]]])
                    L.append(Landscape(k,critical_points))
                else:
                    # find (b',d') the first bar with d'>d
                    p = np.min(np.nonzero(sortedbarcode[:,1]>d)) # returns min arg of row with death larger than d
                    [bnew,dnew,m],sortedbarcode = pop(sortedbarcode,p)
                    if(bnew > d):
                        critical_points = np.vstack([critical_points, [d,0]])
                    if(bnew >= d):
                        critical_points = np.vstack([critical_points, [bnew,0]])
                    else:
                        critical_points = np.vstack([critical_points, [(bnew+d)/2,(d-bnew)/2]])
                        sortedbarcode = np.vstack([sortedbarcode,[bnew,d,1]])
                        sortedbarcode = sortedbarcode[np.lexsort((-sortedbarcode[:,1],sortedbarcode[:,0]))]
                        p+=1
                    critical_points = np.vstack([critical_points,[(bnew+dnew)/2,(dnew-bnew)/2]])
                    b,d = [bnew,dnew]
            k+=1

    return Landscapes(L)


def compute_multiparameter_landscape(computed_data, bounds, resolution , index, weight = [1,1]):
    """ Returns a multiparameter landscape object, the weighted multiparameter landscape calculated in parameter
        range specified by 'bounds'

    :param computed_data: byte array
        RIVET data from a compute_* function in the rivet module
    :param bounds: [[xmin,xmax],[ymin,ymax]]
        The parameter range over which to calculate the landscape
    :resolution: int
        The number of subdivisions of the parameter space for which the multiparameter landscape is to be computed
    :index: int
        The index of the landscape wanting to be calculated
    :weight: [w_1,w_2]
        A rescaling of the parameter space corresponding to alternative interleaving distances
    """
    # Discretise parameter space over which we calculate the landscape values
    x = np.linspace(bounds[0][0],bounds[0][1],resolution)
    y = np.linspace(bounds[1][0],bounds[1][1],resolution)
    # Find the slope offset parametrisations of the (slope = arctan(w1/w2))
    slopes = np.degrees(np.arctan(w1/w2))*np.ones(2*resolution-1)
    points = np.zeros((2*resolution-1,2))
    points[:resolution,0] = bounds[0][0]
    points[resolution-1:,0] = x
    points[resolution:,1] = bounds[1][0]
    points[:resolution,1] = np.flip(y,0)
    offsets = find_offsets(slopes,points)
    # Compute the barcodes of these slices
    slices = np.stack((slopes,offsets),axis=-1)
    barcodes = rivet.barcodes(computed_data,slices)
    # Compute and store the landscapes for each barcode (can be parallelised)
    LandscapeSlices = []
    for i in range(2*resolution-1):
        LandscapeSlices.append(compute_landscapes(barcodes[i,1]))
    # Form the array of zvalues by querying the appropriate points of the landscapes (can be done more efficiently)
    zvalues = np.zeros((resolution,resolution))

    return zvalues

def pop(array, row):
    poppedrow = array[row,:]
    array = np.delete(array,row,0)
    return poppedrow , array
