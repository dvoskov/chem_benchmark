from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.physics_base import PhysicsBase

class Elasticity (PhysicsBase):
    """"
       Class to generate deadoil physics for poromechanical simulation, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, n_points, max_u, n_dim,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d'):
        """"
           Initialize DeadOil class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
                - n_dim: space dimension
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.n_components = 0
        self.n_dim = n_dim
        self.n_vars = self.n_dim + self.n_components
        self.vars = ['displacement']
        self.max_u = max_u
        self.n_ops = self.n_dim
        self.n_axes_points = index_vector([n_points, n_points, n_points])
        self.n_axes_min = value_vector([-max_u, -max_u, -max_u])
        self.n_axes_max = value_vector([max_u, max_u, max_u])

        # create property evaluators
        self.density = 2.E+3
        self.el_dens_ev = elasticity_string_density_evaluator(self.density)

        # create engine accumulation and flux operators evaluator
        self.engine = eval("engine_elasticity_cpu%d" % (self.n_dim))()
        self.acc_flux_etor = elasticity_flux_evaluator(self.el_dens_ev)
        self.acc_flux_itor = self.create_interpolator(evaluator=self.acc_flux_etor,
                                                      n_dims = self.n_vars,
                                                      n_ops = self.n_ops,
                                                      axes_n_points = self.n_axes_points,
                                                      axes_min = self.n_axes_min,
                                                      axes_max = self.n_axes_max,
                                                      type=itor_type,
                                                      mode=itor_mode,
                                                      platform=platform,
                                                      precision=itor_precision  )

        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
    def init_wells(self, wells):
        return 0
    def set_uniform_initial_conditions(self, mesh, uniform_displacement: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_displacement: uniform displacement setting
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        assert(self.n_dim == len(uniform_displacement))

        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacement[i]

