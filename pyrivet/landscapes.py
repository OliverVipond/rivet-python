from math import floor
import copy
import numbers
import math

MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES = 50.0

class landscape(object):
    """ Collection of non zero landscapes """

    def __init__(self, barcode, x_values_range=None, x_value_step_size=None,y_value_max=None,maxind=None):
        """ Computes the collection of persistence landscapes associated to a barcode up to index maxind
        using the algorithm set out in Bubenik + Dlotko.
        :param barcode: A barcode object
        :param maxind: The maximum index landscape to calculate
        """

        barcode = barcode.expand()
        barcode = barcode.to_array()
        if maxind is None:
            maxind = len(barcode)
            
        self.maximum_landscape_depth = maxind

        # Apply y-value threshold
        if y_value_max is not None:
            def max_y_value(persistence_pair):
                birth,death,_ = persistence_pair
                return (death-birth)/2.0
            barcode = filter(max_y_value,barcode)

        # Determine the minimum and maximum x-values for the
        # landscape if none are specified
        # Using map semantics here in case we want to exchange it for something
        # parallelized later
        if x_values_range is None:
            def max_x_value_of_persistence_point(persistence_pair):
                _,death,_ = persistence_pair
                return death
            def min_x_value_of_persistence_point(persistence_pair):
                birth,_,_ = persistence_pair
                return birth
            death_vector = np.array(list(map(max_x_value_of_persistence_point,barcode)))
            birth_vector = np.array(list(map(min_x_value_of_persistence_point,barcode)))
            self.x_values_range = [np.amin(birth_vector),np.amax(death_vector)]
        else:
            self.x_values_range = x_values_range

        # This parameter value is recommended; if it's not provided,
        # this calculation tries to keep the total memory for the landscape under
        # the threshold number of MiB
        if x_value_step_size is None:
            self.x_value_step_size = maxind*(self.x_values_range[1]-self.x_values_range[0])*64.0/(MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES*pow(2,23))
        else:
            self.x_value_step_size = x_value_step_size

        def tent_function_for_pair(persistence_pair):
            birth,death,_ = persistence_pair
            def evaluate_tent_function(x):
                if x <= (birth+death)/2.0:
                    return max(0,x-birth)
                else:
                    return max(0,death-x)
            return evaluate_tent_function

        x_values_start,x_value_stop = self.x_values_range
        width_of_x_values = x_value_stop - x_values_start
        number_of_steps = int(round(width_of_x_values/self.x_value_step_size))
#         print('nbr of step='+str(number_of_steps))
        x_values = np.array(range(number_of_steps))
        x_values = x_values*self.x_value_step_size + x_values_start
        self.grid_values = x_values

        def x_value_to_slice(x_value):
            unsorted_slice_values = np.array(list(map(lambda pair : tent_function_for_pair(pair)(x_value),barcode)))
            return unsorted_slice_values

        landscape_slices = np.array(list(map(x_value_to_slice,x_values)))
        
        if (maxind>landscape_slices.shape[1]):
            padding = np.zeros((number_of_steps,maxind-landscape_slices.shape[1]))
            landscape_slices = np.hstack((landscape_slices,padding))
        
            self.landscape_matrix = np.empty([maxind,number_of_steps])
            for i in range(number_of_steps):
                self.landscape_matrix[:,i] = landscape_slices[i,:]
                
        if (maxind<=landscape_slices.shape[1]):      
            self.landscape_matrix = np.empty([landscape_slices.shape[1],number_of_steps])
            for i in range(number_of_steps):
                self.landscape_matrix[:,i] = landscape_slices[i,:]

        # sorts all the columns using numpy's sort
        self.landscape_matrix = -np.sort(-self.landscape_matrix,axis=0)
        self.landscape_matrix = self.landscape_matrix[:maxind,:]


    def __repr__(self):
        return "Landscapes(%s)" % self.landscape_matrix

    def plot_landscapes(self,landscapes_to_plot=None):
        """ Plots the landscapes in the collection to a single axes"""
        if landscapes_to_plot is None:
            landscapes_to_plot = range(self.maximum_landscape_depth)
        elif type(landscapes_to_plot) is int:
            landscapes_to_plot = range(landscapes_to_plot)
        for k in landscapes_to_plot:
            x = self.grid_values
            y = self.landscape_matrix[k,:]
            plt.plot(x,y)
        plt.show()

    def __add__(self,other_landscape):
        if np.shape(self.landscape_matrix) != np.shape(other_landscape.landscape_matrix):
            raise TypeError("Attempted to add two landscapes with different shapes.")
        if self.x_values_range != other.x_values_range:
            raise TypeError("Attempted to add two landscapes with different ranges of x-values.")
        added_landscapes = copy.deepcopy(self)
        added_landscapes.landscape_matrix = self.landscape_matrix + other.landscape_matrix
        return added_landscapes

    def __mul__(self,multiple):
        # Scalar multiplication
        if isinstance(multiple,numbers.Number):
            multiplied_landscape = copy.deepcopy(self)
            multiplied_landscape.landscape_matrix *= multiple
            return multiplied_landscape
        # Inner product, * is element-wise multiplication
        else:
            return np.sum(self.landscape_matrix*multiple.landscape_matrix)

    def __sub__(self,other):
        return self + (-1.0)*other

class multiparameter_landscape(object):
    """ Collection of non zero multiparameter landscapes """

    def __init__(self, computed_data, maxind=10, bounds=None, grid_step_size=None, weight = [1,1]):
        """ Returns a multiparameter landscape object, the weighted multiparameter landscape calculated in parameter
            range specified by 'bounds'

        :param computed_data: byte array
            RIVET data from a compute_* function in the rivet module
        :param bounds: [[xmin,ymin],[xmax,ymax]]
            The parameter range over which to calculate the landscape
        :resolution: int
            The number of subdivisions of the parameter space for which the multiparameter landscape is to be computed
        :index: int
            The index of the landscape wanting to be calculated
        :weight: [w_1,w_2]
            A rescaling of the parameter space corresponding to alternative interleaving distances - not yet implemented
        """

        if bounds is None:
            Bounds = rivet.bounds(computed_data)
            self.bounds = [[Bounds.lower_left[0],Bounds.lower_left[1]],[Bounds.upper_right[0],Bounds.upper_right[1]]]
        else:
            self.bounds = bounds

        self.maximum_landscape_depth = maxind

        if grid_step_size is None:
            self.grid_step_size = maxind*(self.bounds[0][1]-self.bounds[0][0])*(self.bounds[1][1]-self.bounds[1][0])*64.0/(MAXIMUM_NUMBER_OF_MEGABYTES_FOR_LANDSCAPES*pow(2,23))
        else:
            self.grid_step_size = grid_step_size

        # Discretise parameter space over which we calculate the landscape values
        x_steps = int(round((self.bounds[1][0]-self.bounds[0][0])/self.grid_step_size+1))
        y_steps = int(round((self.bounds[1][1]-self.bounds[0][1])/self.grid_step_size+1))
        x = np.linspace(self.bounds[0][0], self.bounds[1][0], x_steps)
        y = np.linspace(self.bounds[0][1], self.bounds[1][1], y_steps)
        # Find the slope offset parametrisations of the (slope = arctan(w1/w2))
        slopes = np.degrees(np.arctan(weight[0]/weight[1]))*np.ones(x_steps+y_steps-1)
        lower_points = np.zeros((x_steps+y_steps-1,2))
        upper_points = np.zeros((x_steps+y_steps-1,2))
        lower_points[:y_steps,0] = self.bounds[0][0]
        lower_points[y_steps-1:,0] = x
        lower_points[y_steps:,1] = self.bounds[0][1]
        lower_points[:y_steps,1] = np.flip(y,0)
        upper_points[:x_steps,0] = x
        upper_points[x_steps-1:,0] = self.bounds[1][0]
        upper_points[:x_steps,1] = self.bounds[1][1]
        upper_points[x_steps-1:,1] =np.flip(y,0)
        points = [lower_points,upper_points]
        offsets = matching_distance.find_offsets(slopes,lower_points)

        # Compute the barcodes of these slices
        slices = np.stack((slopes,offsets),axis=-1)
        barcodes = rivet.barcodes(computed_data,slices)
        # Produce function which computes appropriate single landscape for each diagonal slices
        # Antidiagonals of each matrix are labelled 0 to x_steps+y_steps-1 from top left to bottom right - same indexing as corresponding barcodes

        def find_parameter_of_point_on_line(sl, offset, pt):
            """Finds the RIVET parameter representation of point on the line
            (sl,offset).  recall that RIVET parameterizes by line length, and takes the
            point where the line intersects the positive x-axis or y-axis to be
            parameterized by 0.  If the line is itself the x-axis or y-axis, then the
            origin is parameterized by 0.  

            WARNING: Code assumes that the point lies on the line, and does not check
            this.  Relatedly, the function could be written using only slope or
            offset as input, not both.  """

            if sl == 90:
                return pt[1]

            if sl == 0:
                return pt[0]

            # Otherwise the line is neither horizontal or vertical.
            m = math.tan(math.radians(sl))

            # Find the point on the line parameterized by 0.

            # If offset is positive, this is a point on the y-axis, otherwise, it is
            # a point on the x-axis.

            # Actually, the above is what SHOULD be true, but in the current implementation of RIVET
            # 0 is the point on the line closest to the origin.

            if offset > 0:
                y_int = pt[1] - m * pt[0]
                dist = np.sqrt(pow(pt[1] - y_int, 2) + pow(pt[0], 2))
                if pt[0] > 0:
                    return dist
                else:
                    return -dist
            else:
                x_int = pt[0] - pt[1] / m
                dist = np.sqrt(pow(pt[1], 2) + pow(pt[0] - x_int, 2))
                if pt[1] > 0:
                    return dist
                else:
                    return -dist

        parameter_step_size = np.sqrt(2)*self.grid_step_size


        def landscape_inputs(points, slices, antidiagonal_index):
            slopes = slices[:,0]
            offsets = slices[:,1]
            lower_points, upper_points = points
            k = antidiagonal_index
            if (k< y_steps):
                x_values_range = [find_parameter_of_point_on_line(slopes[k],offsets[k],lower_points[k,:]),find_parameter_of_point_on_line(slopes[k],offsets[k],lower_points[k,:])+x_steps*parameter_step_size]
            elif (k >= y_steps) and (k < x_steps ):
                x_values_range = [find_parameter_of_point_on_line(slopes[k],offsets[k],lower_points[k,:])-(k+1-y_steps)*parameter_step_size,find_parameter_of_point_on_line(slopes[k],offsets[k],upper_points[k,:])+(x_steps-k)*parameter_step_size]
            elif (k>= max(x_steps,y_steps)):
                x_values_range = [find_parameter_of_point_on_line(slopes[k],offsets[k],upper_points[k,:])-x_steps*parameter_step_size,find_parameter_of_point_on_line(slopes[k],offsets[k],upper_points[k,:])]

            return x_values_range


        # Compute the multiparameter landscape matrix. Not yet parallelised
        self.landscape_matrix = np.empty([maxind,y_steps,x_steps])
        print('x_steps'+str(x_steps))
        print('y_steps'+str(y_steps))
        antidiagonal_landscape_slices = list(map(lambda index : landscape(barcodes[index][1], landscape_inputs(points,slices,index), parameter_step_size, maxind=maxind),range(x_steps+y_steps-1)))
        holder_matrix = np.zeros((maxind,x_steps+y_steps-1,x_steps))
        for index in range(x_steps+y_steps-1):
            holder_matrix[:,index,:] = antidiagonal_landscape_slices[index].landscape_matrix

        for index in range(y_steps):

            self.landscape_matrix[:,index,:] = holder_matrix.diagonal(index,2,1)

    def __repr__(self):
        return "Multiparameter Landscapes(%s)" % self.landscape_matrix
