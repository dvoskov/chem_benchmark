from darts.models.reservoirs.struct_reservoir import StructReservoir
from benchmark_physics import CustomPhysicsClass
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector
import numpy as np
from benchmark_operator_evaluator import *
from benchmark_properties import *

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


# Model class creation here!
class Model(DartsModel):
    def __init__(self):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir"""
        nx = 1000
        trans_exp = 3
        solid_init = 0.7
        perm = 100 / (1 - solid_init) ** trans_exp
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=1, dx=1, dy=1, dz=1, permx=perm, permy=perm,
                                         permz=perm/10, poro=1, depth=1000)

        """well location"""
        self.reservoir.add_well("I1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=1, j=1, k=1, multi_segment=False)

        self.reservoir.add_well("P1")
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=nx, j=1, k=1, multi_segment=False)

        self.zero = 1e-12
        self.solid_init = solid_init
        self.init_ions = 0.5
        self.solid_inject = self.zero

        """Physical properties"""
        # Create property containers:
        self.equi_prod = (self.init_ions / 2) ** 2
        # self.equi_prod = self.init_ions ** 2 * 55.508 ** 2 / ((1 - self.init_ions) ** 2)
        rock_compr = 1e-7
        self.property_container = property_container(phase_name=['gas', 'wat'],
                                                     component_name=['CO2', 'Ions', 'H2O', 'CaCO3'],
                                                     min_z=self.zero / 10,
                                                     diff_coef=1e-9,
                                                     rock_comp=rock_compr,
                                                     equi_prod=self.equi_prod,
                                                     kin_rate_cte=1e-0)
        self.components = self.property_container.component_name
        self.phases = self.property_container.phase_name

        """ properties correlations """
        self.property_container.flash_ev = FlashBasic([10, 1e-12, 1e-1], min_z=self.zero)
        # self.property_container.flash_ev = FlashBasic([10, 1e-1], min_z=self.zero)
        self.property_container.density_ev = dict([('wat', DensityBasic(1000, 1e-6)), ('gas', DensityBasic(100, 1e-4)),
                                                   ('sol', DensityBasic(2000, rock_compr))])
        self.property_container.viscosity_ev = dict([('wat', ViscosityBasic(1)), ('gas', ViscosityBasic(0.1))])
        self.property_container.rel_perm_ev = dict([('wat', RelPermBasic(2, 0.0)), ('gas', RelPermBasic(2, 0.0))])

        """ Activate physics """
        self.physics = CustomPhysicsClass(self.timer, n_points=1001, min_p=1, max_p=1000,
                                          min_z=self.zero/10, max_z=1-self.zero/10,
                                          property_container=self.property_container)

        self.params.trans_mult_exp = trans_exp

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
        self.params.max_ts = 0.1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-5
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2

        self.timer.node["initialization"].stop()

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        zc_fl_init = [self.zero / (1 - self.solid_init), self.init_ions]
        zc_fl_init = zc_fl_init + [1 - sum(zc_fl_init)]
        self.init_comp = [x * (1 - self.solid_init) for x in zc_fl_init]
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 95, self.init_comp)

    def set_boundary_conditions(self):
        zc_fl_inj_stream_gas = [1 - 2 * self.zero / (1 - self.solid_inject), self.zero / (1 - self.solid_inject)]
        zc_fl_inj_stream_gas = zc_fl_inj_stream_gas + [1 - sum(zc_fl_inj_stream_gas)]
        self.inj_stream_gas = [x * (1 - self.solid_inject) for x in zc_fl_inj_stream_gas]
        tot_inj = 0.2

        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_gas_inj(tot_inj, self.inj_stream_gas)
            else:
                w.control = self.physics.new_bhp_prod(50)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks
        self.op_num[n_res:] = 1
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well]


    def print_and_plot(self, filename):

        nc = self.property_container.nc
        Sg = np.zeros(self.reservoir.nb)
        Ss = np.zeros(self.reservoir.nb)
        X = np.zeros((self.reservoir.nb, nc - 1, 2))

        rel_perm = np.zeros((self.reservoir.nb, 2))
        visc = np.zeros((self.reservoir.nb, 2))
        density = np.zeros((self.reservoir.nb, 3))
        density_m = np.zeros((self.reservoir.nb, 3))

        Xn = np.array(self.physics.engine.X, copy=True)


        P = Xn[0:self.reservoir.nb * nc:nc]
        z_caco3 = 1 - (Xn[1:self.reservoir.nb * nc:nc] + Xn[2:self.reservoir.nb * nc:nc] + Xn[3:self.reservoir.nb * nc:nc])

        z_co2 = Xn[1:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_inert = Xn[2:self.reservoir.nb * nc:nc] / (1 - z_caco3)
        z_h2o = Xn[3:self.reservoir.nb * nc:nc] / (1 - z_caco3)

        for ii in range(self.reservoir.nb):
            x_list = Xn[ii*nc:(ii+1)*nc]
            state = value_vector(x_list)
            properties = properties_evaluator(self.property_container)
            (sg, ss, x, y, rho_aq, rho_g, rho_aq_m, rho_g_m, rho_s, rho_s_m) = properties.evaluate(state)

            kr_w = self.property_container.rel_perm_ev['wat'].evaluate(1 - sg)
            kr_g = self.property_container.rel_perm_ev['gas'].evaluate(sg)

            mu_w = self.property_container.viscosity_ev['wat'].evaluate(state)
            mu_g = self.property_container.viscosity_ev['gas'].evaluate(state)

            rel_perm[ii, :] = [kr_w, kr_g]
            visc[ii, :] = [mu_w, mu_g]
            density[ii, :] = [rho_aq, rho_g, rho_s]
            density_m[ii, :] = [rho_aq_m, rho_g_m, rho_s_m]

            X[ii, :, 0] = x
            X[ii, :, 1] = y
            Sg[ii] = sg
            Ss[ii] = ss

        # Write all output to a file:
        with open(filename, 'w+') as f:
            # Print headers:
            print('//Gridblock\t gas_sat\t pressure\t C_m\t poro\t co2_liq\t co2_vap\t h2o_liq\t h2o_vap\t ca_plus_co3_liq\t liq_dens\t vap_dens\t solid_dens\t liq_mole_dens\t vap_mole_dens\t solid_mole_dens\t rel_perm_liq\t rel_perm_gas\t visc_liq\t visc_gas', file=f)
            print('//[-]\t [-]\t [bar]\t [kmole/m3]\t [-]\t [-]\t [-]\t [-]\t [-]\t [-]\t [kg/m3]\t [kg/m3]\t [kg/m3]\t [kmole/m3]\t [kmole/m3]\t [kmole/m3]\t [-]\t [-]\t [cP]\t [cP]', file=f)
            for ii in range (self.reservoir.nb):
                print('{:d}\t {:6.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:8.5f}\t {:8.5f}\t {:8.5f}\t {:7.5f}\t {:7.5f}\t {:7.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}\t {:6.5f}'.format(
                    ii, Sg[ii], P[ii], Ss[ii] * density_m[ii, 2], 1 - Ss[ii], X[ii, 0, 0], X[ii, 0, 1], X[ii, 2, 0], X[ii, 2, 1], X[ii, 1, 0],
                    density[ii, 0], density[ii, 1], density[ii, 2], density_m[ii, 0], density_m[ii, 1], density_m[ii, 2],
                    rel_perm[ii, 0], rel_perm[ii, 1], visc[ii, 0], visc[ii, 1]), file=f)

        """ start plots """

        font_dict_title = {'family': 'sans-serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': 18,
                           }

        font_dict_axes = {'family': 'monospace',
                          'color': 'black',
                          'weight': 'normal',
                          'size': 18,
                          }

        fig, axs = plt.subplots(3, 3, figsize=(16, 14), dpi=200, facecolor='w', edgecolor='k')
        """ sg and x """
        axs[0][0].plot(z_co2, 'b')
        axs[0][0].set_xlabel('x [m]', font_dict_axes)
        axs[0][0].set_ylabel('$z_{CO_2}$ [-]', font_dict_axes)
        axs[0][0].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][1].plot(z_h2o, 'b')
        axs[0][1].set_xlabel('x [m]', font_dict_axes)
        axs[0][1].set_ylabel('$z_{H_2O}$ [-]', font_dict_axes)
        axs[0][1].set_title('Fluid composition', fontdict=font_dict_title)

        axs[0][2].plot(z_inert, 'b')
        axs[0][2].set_xlabel('x [m]', font_dict_axes)
        axs[0][2].set_ylabel('$z_{w, Ca+2} + z_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[0][2].set_title('Fluid composition', fontdict=font_dict_title)

        axs[1][0].plot(X[:, 0, 0], 'b')
        axs[1][0].set_xlabel('x [m]', font_dict_axes)
        axs[1][0].set_ylabel('$x_{w, CO_2}$ [-]', font_dict_axes)
        axs[1][0].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][1].plot(X[:, 2, 0], 'b')
        axs[1][1].set_xlabel('x [m]', font_dict_axes)
        axs[1][1].set_ylabel('$x_{w, H_2O}$ [-]', font_dict_axes)
        axs[1][1].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[1][2].plot(X[:, 1, 0], 'b')
        axs[1][2].set_xlabel('x [m]', font_dict_axes)
        axs[1][2].set_ylabel('$x_{w, Ca+2} + x_{w, CO_3-2}$ [-]', font_dict_axes)
        axs[1][2].set_title('Liquid mole fraction', fontdict=font_dict_title)

        axs[2][0].plot(P, 'b')
        axs[2][0].set_xlabel('x [m]', font_dict_axes)
        axs[2][0].set_ylabel('$p$ [bar]', font_dict_axes)
        axs[2][0].set_title('Pressure', fontdict=font_dict_title)

        axs[2][1].plot(Sg, 'b')
        axs[2][1].set_xlabel('x [m]', font_dict_axes)
        axs[2][1].set_ylabel('$s_g$ [-]', font_dict_axes)
        axs[2][1].set_title('Gas saturation', fontdict=font_dict_title)

        axs[2][2].plot(1 - Ss, 'b')
        axs[2][2].set_xlabel('x [m]', font_dict_axes)
        axs[2][2].set_ylabel('$\phi$ [-]', font_dict_axes)
        axs[2][2].set_title('Porosity', fontdict=font_dict_title)

        left = 0.05  # the left side of the subplots of the figure
        right = 0.95  # the right side of the subplots of the figure
        bottom = 0.05  # the bottom of the subplots of the figure
        top = 0.95  # the top of the subplots of the figure
        wspace = 0.25  # the amount of width reserved for blank space between subplots
        hspace = 0.25  # the amount of height reserved for white space between subplots
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        for ii in range(3):
            for jj in range(3):
                for tick in axs[ii][jj].xaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

                for tick in axs[ii][jj].yaxis.get_major_ticks():
                    tick.label.set_fontsize(20)

        plt.tight_layout()
        plt.savefig("results_kinetic_brief.pdf")
        plt.show()
