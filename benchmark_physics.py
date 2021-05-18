from benchmark_operator_evaluator import AccFluxKinDiffEvaluator, RateEvaluator, AccFluxKinDiffWellEvaluator
from benchmark_properties import *
from darts.engines import *
import numpy as np
from darts.models.physics.physics_base import PhysicsBase


# Define our own operator evaluator class
class CustomPhysicsClass(PhysicsBase):
    def __init__(self, timer, n_points, min_p, max_p, min_z, max_z, property_container,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d'):
        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.components = property_container.component_name
        self.n_components = property_container.nc
        self.phases = property_container.phase_name
        self.n_phases = property_container.n_phases
        self.n_vars = self.n_components
        self.n_rate_temp_ops = self.n_phases

        # Name of interpolation method and engine used for this physics:
        engine_name = eval("engine_nc_kin_dif_np_cpu%d_%d" % (self.n_components, self.n_phases))
        self.n_ops = 3 * self.n_components + self.n_phases * self.n_components + self.n_phases + 1

        # Initialize main evaluators
        self.acc_flux_etor = AccFluxKinDiffEvaluator(property_container)
        self.acc_flux_etor_well = AccFluxKinDiffWellEvaluator(property_container)

        # Set axis limits
        self.n_axes_points = index_vector([21] + [n_points] * (self.n_components - 1))
        self.n_axes_min = value_vector([min_p] + [min_z] * (self.n_components - 1))
        self.n_axes_max = value_vector([max_p] + [max_z] * (self.n_components - 1))

        # Create interpolators
        self.acc_flux_itor = self.create_interpolator(self.acc_flux_etor, self.n_vars, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform)

        self.acc_flux_itor_well = self.create_interpolator(self.acc_flux_etor_well, self.n_vars, self.n_ops,
                                                           self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                           platform=platform)

        # create rate operators evaluator
        self.rate_etor = RateEvaluator(property_container)

        # interpolator platform is 'cpu' since rates are always computed on cpu
        self.rate_itor = self.create_interpolator(self.rate_etor, self.n_vars, self.n_rate_temp_ops,
                                                  self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                  platform='cpu')

        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')
        self.create_itor_timers(self.acc_flux_itor_well, 'well interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # create engine according to physics selected
        self.engine = engine_name()

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_gas_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 0, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        self.new_rate_oil_inj = lambda rate, inj_stream: rate_inj_well_control(self.phases, 1, self.n_components,
                                                                               self.n_components, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_gas_prod = lambda rate: rate_prod_well_control(self.phases, 0, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)
        self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.phases, 1, self.n_components,
                                                                     self.n_components,
                                                                     rate, self.rate_itor)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = uniform_composition[c]
