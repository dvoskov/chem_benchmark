from math import fabs

from darts.engines import *
from darts.physics import *
from darts.tools.keyword_file_tools import *
from darts.models.physics.physics_base import PhysicsBase

class DeadOilPoromechanics (PhysicsBase):
    """"
       Class to generate physics for deadoil poromechanics, including
        Important definitions:
            - accumulation_flux_operator_evaluator
            - accumulation_flux_operator_interpolator
            - rate_evaluator
            - rate_interpolator
            - property_evaluator
            - well_control (rate, bhp)
    """
    def __init__(self, timer, physics_filename, n_points, min_p, max_p, max_u, min_z,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d'):
        """"
           Initialize DeadOil class.
           Arguments:
                - timer: time recording object
                - physics_filename: filename of the physical properties
                - n_points: number of interpolation points
                - min_p, max_p: minimum and maximum pressure
                - min_z: minimum composition
        """
        self.timer = timer.node["simulation"]
        self.n_points = n_points
        self.min_p = min_p
        self.max_p = max_p
        self.min_z = min_z
        self.max_u = max_u
        self.n_dim = 3
        self.n_components = 2
        self.n_vars = self.n_dim + self.n_components
        self.phases = ['water', 'oil']
        self.components = ['water', 'oil']
        self.rate_phases = ['water', 'oil', 'liquid']
        self.vars = ['pressure', 'water composition', 'ux', 'uy', 'uz']
        self.n_phases = len(self.phases)
        self.n_axes_points = index_vector([n_points] * self.n_components)
        self.n_axes_min = value_vector([min_p, min_z])
        self.n_axes_max = value_vector([max_p, 1 - min_z])
        self.n_ops = self.n_components + self.n_components + self.n_components

        # read keywords from physics file
        pvdo = get_table_keyword(physics_filename, 'PVDO')
        swof = get_table_keyword(physics_filename, 'SWOF')
        pvtw = get_table_keyword(physics_filename, 'PVTW')[0]
        dens = get_table_keyword(physics_filename, 'DENSITY')[0]
        rock = get_table_keyword(physics_filename, 'ROCK')

        swof_well = []
        swof_well.append(value_vector([swof[0][0],  swof[0][1],  swof[0][2],  0.0]))
        swof_well.append(value_vector([swof[-1][0], swof[-1][1], swof[-1][2], 0.0]))

        surface_oil_dens = dens[0]
        surface_water_dens = dens[1]

        # create property evaluators
        self.do_oil_dens_ev = dead_oil_table_density_evaluator(pvdo, surface_oil_dens)
        self.do_wat_dens_ev = dead_oil_string_density_evaluator(pvtw, surface_water_dens)
        self.do_oil_visco_ev = dead_oil_table_viscosity_evaluator(pvdo)
        self.do_water_visco_ev = dead_oil_string_viscosity_evaluator(pvtw)
        self.do_water_sat_ev = dead_oil_water_saturation_evaluator(self.do_wat_dens_ev, self.do_oil_dens_ev)
        self.do_oil_oil_relperm_ev = table_phase2_relative_permeability_evaluator(self.do_water_sat_ev, swof)
        self.do_oil_wat_relperm_ev = table_phase1_relative_permeability_evaluator(self.do_water_sat_ev, swof)
        self.density = 2.E+3
        self.el_dens_ev = elasticity_string_density_evaluator(self.density)
        #self.do_pcow_ev = table_phase_capillary_pressure_evaluator(self.do_water_sat_ev, swof)

        # evaluate names of required classes depending on amount of components, self.phases, and selected physics
        self.engine = eval("engine_nc_pm_%s%d" % (platform, self.n_components))()
        self.acc_flux_etor = pm_dead_oil_acc_flux_evaluator(self.do_oil_dens_ev, self.do_oil_visco_ev,
                                                    self.do_oil_oil_relperm_ev, self.do_wat_dens_ev,
                                                    self.do_water_sat_ev, self.do_water_visco_ev,
                                                    self.do_oil_wat_relperm_ev,
                                                    self.el_dens_ev)
        self.acc_flux_itor = self.create_interpolator(  evaluator=self.acc_flux_etor,
                                                        n_dims = self.n_components,
                                                        n_ops = self.n_ops,
                                                        axes_n_points=self.n_axes_points,
                                                        axes_min=self.n_axes_min,
                                                        axes_max=self.n_axes_max,
                                                        type=itor_type,
                                                        mode=itor_mode,
                                                        platform=platform,
                                                        precision=itor_precision)
        # set up timers
        self.create_itor_timers(self.acc_flux_itor, 'reservoir interpolation')

        # create rate operators evaluator
        #self.rate_etor = dead_oil_rate_evaluator(self.do_oil_dens_ev, self.do_oil_visco_ev, self.do_oil_oil_relperm_ev,
        #                                         self.do_wat_dens_ev, self.do_water_sat_ev, self.do_water_visco_ev,
        #                                         self.do_oil_wat_relperm_ev)

        #try:
            # try first to create interpolator with 4-byte index type
        #    self.rate_itor = rate_interpolator_name(self.rate_etor, index_vector([n_points, n_points]),
        #                                            value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))
        #except RuntimeError:
            # on exception (assume too small integer range) create interpolator with long index type
        #    self.rate_itor = rate_interpolator_name_long(self.rate_etor, index_vector([n_points, n_points]),
        #                                            value_vector([min_p, min_z]), value_vector([max_p, 1 - min_z]))

        # set up timers
        #self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"] = timer_node()
        #self.rate_itor.init_timer_node(
        #    self.timer.node["jacobian assembly"].node["interpolation"].node["rate interpolation"])

        # create well controls
        # water stream
        # min_z is the minimum composition for interpolation
        # 2*min_z is the minimum composition for simulation
        # let`s take 3*min_z as the minimum composition for injection to be safely within both limits

        #self.water_inj_stream = value_vector([1 - 3 * min_z])
        #self.new_bhp_water_inj = lambda bhp: bhp_inj_well_control(bhp, self.water_inj_stream)
        #self.new_rate_water_inj = lambda rate: rate_inj_well_control(self.rate_phases, 0, self.n_components,
        #                                                             self.n_components,
        #                                                             rate,
        #                                                             self.water_inj_stream, self.rate_itor)
        #self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        #self.new_rate_water_prod = lambda rate: rate_prod_well_control(self.rate_phases, 0, self.n_components,
        #                                                               self.n_components,
        #                                                               rate, self.rate_itor)

        #self.new_rate_oil_prod = lambda rate: rate_prod_well_control(self.rate_phases, 1, self.n_components,
        #                                                               self.n_components,
        #                                                               rate, self.rate_itor)
        #self.new_rate_liq_prod = lambda rate: rate_prod_well_control(self.rate_phases, 2, self.n_components,
        #                                                               self.n_components,
        #                                                               rate, self.rate_itor)
        #self.new_acc_flux_itor = lambda new_acc_flux_etor: acc_flux_itor_name(new_acc_flux_etor,
        #                                                                      index_vector([n_points, n_points]),
        #                                                                      value_vector([min_p, min_z]),
        #                                                                      value_vector([max_p, 1 - min_z]))

    def init_wells(self, wells):
        """""
        Function to initialize the well rates for each well
        Arguments:
            -wells: well_object array
        """
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_components, self.rate_phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list, uniform_displacements: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
        """
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
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = uniform_displacements[i]

    def set_nonuniform_initial_conditions(self, mesh, initial_pressure, initial_composition: list, initial_displacement: list):
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure[:] = initial_pressure
        # set initial composition
        mesh.composition.resize(nb * (self.n_components - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.n_components - 1):
            composition[c::(self.n_components - 1)] = initial_composition[c]
        # set initial displacements
        displacement = np.array(mesh.displacement, copy=False)
        for i in range(self.n_dim):
            displacement[i::self.n_dim] = initial_displacement[i]
