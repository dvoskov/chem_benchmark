from darts.engines import *

"""
    Base physics class with common utility functions

"""


class PhysicsBase:
    """
        Create interpolator object according to specified parameters

        Parameters
        ----------
        evaluator : an operator_set_evaluator_iface object
            State operators to be interpolated. Evaluator object is used to generate supporting points
        n_dims : integer
            The number of dimensions for interpolation (parameter space dimensionality)
        n_ops : integer
            The number of operators to be interpolated. Should be consistent with evaluator.
        axes_n_points: an index_vector, pybind-type vector of integers
            The number of supporting points for each axis.
        axes_min : a value_vector, pybind-type vector of floats
            The minimum value for each axis.
        axes_max : a value_vector, pybind-type vector of floats
            The maximum value for each axis.
        type : string
            interpolator type:
            'multilinear' (default) - piecewise multilinear generalization of piecewise bilinear interpolation on
                                      rectangles
            'linear' - a piecewise linear generalization of piecewise linear interpolation on triangles
        type : string
            interpolator mode:
            'adaptive' (default) - only supporting points required to perform interpolation are evaluated on-the-fly
            'static' - all supporting points are evaluated during itor object construction
        platform : string
            platform used for interpolation calculations :
            'cpu' (default) - interpolation happens on CPU
            'gpu' - interpolation happens on GPU
        precision : string
            precision used in interpolation calculations:
            'd' (default) - supporting points are stored and interpolation is performed using double precision
            's' - supporting points are stored and interpolation is performed using single precision
    """

    def create_interpolator(self, evaluator: operator_set_evaluator_iface, n_dims: int, n_ops: int,
                            axes_n_points: index_vector, axes_min: value_vector, axes_max: value_vector,
                            type: str = 'multilinear', mode: str = 'adaptive',
                            platform: str = 'cpu', precision: str = 'd'):
        # verify then inputs are valid
        assert len(axes_n_points) == n_dims
        assert len(axes_min) == n_dims
        assert len(axes_max) == n_dims
        for n_p in axes_n_points:
            assert n_p > 1
        
        #temporary fix till adaptive gpu itor comes
        if platform == 'gpu':
            mode = 'static'
            
        # calculate object name using 32 bit index type (i)
        itor_name = "%s_%s_%s_interpolator_i_%s_%d_%d" % (type,
                                                          mode,
                                                          platform,
                                                          precision,
                                                          n_dims,
                                                          n_ops)
        itor = None

        # try to create itor with 32-bit index type first (kinda a bit faster)
        try:
            itor = eval(itor_name)(evaluator, axes_n_points, axes_min, axes_max)
        except (ValueError, NameError):
            # 32-bit index type did not succeed: either total amount of points is out of range or has not been compiled
            # try 64 bit now raising exception this time if goes wrong:
            itor_name_long = itor_name.replace('interpolator_i', 'interpolator_l')
            itor = eval(itor_name_long)(evaluator, axes_n_points, axes_min, axes_max)
        return itor

    """
            Create timers for interpolators.

            Parameters
            ----------
            itor : an operator_set_gradient_evaluator_iface object
                The object which performes evaluation of operator gradient (interpolators currently, AD-based in future) 
            timer_name: string
                Timer name to be used for the given interpolator
        """

    def create_itor_timers(self, itor, timer_name: str):

        try:
            # in case this is a subsequent call, create only timer node for the given timer
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()
        except:
            # in case this is first call, create first only timer nodes for jacobian assembly and interpolation
            self.timer.node["jacobian assembly"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"] = timer_node()
            self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name] = timer_node()

        # assign created timer to interpolator
        itor.init_timer_node(self.timer.node["jacobian assembly"].node["interpolation"].node[timer_name])
