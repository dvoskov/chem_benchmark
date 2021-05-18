from benchmark_properties import *
import numpy as np
from darts.engines import *
from darts.physics import *


# Define our own operator evaluator class
class AccFluxKinDiffEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.property = property_container
        self.min_z = self.property.min_z

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        # Perform normalization over z_caco3 (which is zc[-1]!)
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        zc_sol = zc[-1]
        zc_fl = zc[:-1] / np.sum(zc[:-1])

        nc = len(vec_state_as_np)
        nc_fl = len(zc_fl)
        nump_fl = len(self.property.phase_name)

        properties = properties_evaluator(self.property)
        (sg, ss, x, y, rho_aq, rho_g, rho_aq_m, rho_g_m, rho_s, rho_s_m) = properties.evaluate(state)

        # Flow properties:
        kr_w = self.property.rel_perm_ev['wat'].evaluate(1 - sg)
        kr_g = self.property.rel_perm_ev['gas'].evaluate(sg)

        mu_w = self.property.viscosity_ev['wat'].evaluate(state)
        mu_g = self.property.viscosity_ev['gas'].evaluate(state)

        nump_fl = len(self.property.phase_name)

        '''Operator ids'''
        alpha_id = 0
        beta_id = nc
        gamma_id = 2 * nc
        delta_id = 3 * nc
        eta_id = 3 * nc + nc * nump_fl
        poro_id = 3 * nc + nc * nump_fl + nump_fl

        aq_id = 0
        gas_id = 1

        # Evaluate chemical properties:
        compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        solid_flux = 0

        phi = 1 - ss
        density_tot = sg * rho_g_m + (1 - sg) * rho_aq_m

        # Calculate some simple kinetic rate:
        kinetic_rate = np.zeros((nc,))
        # ion_prod = 55.508 ** 2 * (x[1] / 2) ** 2 / (x[2] ** 2)
        ion_prod = (x[1] / 2) ** 2

        kinetic_rate[1] = - self.property.kin_rate_cte * (1 - ion_prod / self.property.equi_prod) * ss
        # kinetic_rate[1] = - self.property.kin_rate_cte * (1 - ion_prod / self.property.equi_prod) * ss * x[0] * (x[2] - self.min_z) * x[1]
        kinetic_rate[-1] = - 0.5 * kinetic_rate[1]

        # print('ions: ', zc_fl[1], ', omega: ', zc_fl[1] / equi_prod, ', rate: ', kinetic_rate[1])

        for ii in range(nc - 1):
            '''Acc, flux, and diff operator:'''
            values[alpha_id + ii] = zc_fl[ii] * density_tot * compr * phi
            values[beta_id + ii] = y[ii] * rho_g_m * kr_g / mu_g + x[ii] * rho_aq_m * kr_w / mu_w
            values[delta_id + ii * nump_fl + aq_id] = self.property.diff_coef * x[ii]
            values[delta_id + ii * nump_fl + gas_id] = self.property.diff_coef * y[ii]

            '''Kinetic operator'''
            values[gamma_id + ii] = kinetic_rate[ii]

        ii = nc - 1
        '''Solid phase operators'''
        values[alpha_id + ii] = zc[ii] * rho_s_m
        values[beta_id + ii] = solid_flux
        values[delta_id + ii * nump_fl + aq_id] = 0
        values[delta_id + ii * nump_fl + gas_id] = 0
        values[gamma_id + ii] = kinetic_rate[ii]

        values[eta_id + aq_id] = (1 - sg) * rho_aq_m * (self.property.diff_coef > 0)
        values[eta_id + gas_id] = sg * rho_g_m * (self.property.diff_coef > 0)

        '''Porosity operator'''
        values[poro_id] = phi
        return 0


# Define our own operator evaluator class
class AccFluxKinDiffWellEvaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.property = property_container
        self.min_z = self.property.min_z

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        # Perform normalization over z_caco3 (which is zc[-1]!)
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        zc_fl = zc[:-1] / np.sum(zc[:-1])

        nc = len(vec_state_as_np)
        nc_fl = len(zc_fl)
        nump_fl = len(self.property.phase_name)

        properties = properties_evaluator(self.property)
        (sg, ss, x, y, rho_aq, rho_g, rho_aq_m, rho_g_m, rho_s, rho_s_m) = properties.evaluate(state)

        mu_w = self.property.viscosity_ev['wat'].evaluate(state)
        mu_g = self.property.viscosity_ev['gas'].evaluate(state)

        '''Operator ids'''
        alpha_id = 0
        beta_id = nc
        gamma_id = 2 * nc
        delta_id = 3 * nc
        eta_id = 3 * nc + nc * nump_fl
        poro_id = 3 * nc + nc * nump_fl + nump_fl

        aq_id = 0
        gas_id = 1

        # Evaluate chemical properties:
        kinetic_rate = np.zeros((nc,))
        solid_flux = 0
        phi = 1

        compr = (1 + self.property.rock_comp * (pressure - self.property.p_ref))  # compressible rock
        density_tot = sg * rho_g_m + (1 - sg) * rho_aq_m

        for ii in range(nc):
            if ii == (nc - 1):
                '''Solid phase operators'''
                values[alpha_id + ii] = zc[ii] * rho_s_m
                values[beta_id + ii] = solid_flux
                values[delta_id + ii * nump_fl + aq_id] = 0
                values[delta_id + ii * nump_fl + gas_id] = 0
                values[gamma_id + ii] = 0

            else:
                '''Acc, flux, and diff operator:'''
                values[alpha_id + ii] = zc_fl[ii] * density_tot * compr * phi
                values[beta_id + ii] = y[ii] * rho_g_m * sg / mu_g + x[ii] * rho_aq_m * (1 - sg) / mu_w
                values[delta_id + ii * nump_fl + aq_id] = 0
                values[delta_id + ii * nump_fl + gas_id] = 0

                '''Kinetic operator'''
                values[gamma_id + ii] = kinetic_rate[ii]

        values[eta_id + aq_id] = 0
        values[eta_id + gas_id] = 0

        '''Porosity operator'''
        values[poro_id] = 1
        return 0


class RateEvaluator(operator_set_evaluator_iface):
    # Simplest class existing to mankind:
    def __init__(self, property_container):
        # Initialize base-class
        super().__init__()
        self.property = property_container
        self.min_z = self.property.min_z

    def evaluate(self, state, values):
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        # Perform normalization over z_caco3 (which is zc[-1]!)
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))
        zc_fl = zc[:-1] / np.sum(zc[:-1])

        nc = len(vec_state_as_np)
        nc_fl = len(zc_fl)
        nump_fl = len(self.property.phase_name)

        properties = properties_evaluator(self.property)
        (sg, ss, x, y, rho_aq, rho_g, rho_aq_m, rho_g_m, rho_s, rho_s_m) = properties.evaluate(state)

        mu_w = self.property.viscosity_ev['wat'].evaluate(state)
        mu_g = self.property.viscosity_ev['gas'].evaluate(state)

        kr_w = self.property.rel_perm_ev['wat'].evaluate(1 - sg)
        kr_g = self.property.rel_perm_ev['gas'].evaluate(sg)

        density_tot = sg * rho_g_m + (1 - sg) * rho_aq_m  # kmole/m3

        flux = np.zeros(nc_fl)
        for ii in range(nc_fl):
            flux[ii] = rho_g_m * kr_g * y[ii] / mu_g + rho_aq_m * x[ii] * kr_w / mu_w  # kmole/(m3.cP)

        # step-2
        flux_sum = np.sum(flux)
        # step-4
        values[0] = sg * flux_sum / density_tot  # m3 (after dp*T)
        # values[0] = sg * flux_sum  # kmole
        values[1] = (1 - sg) * flux_sum / density_tot
        return 0


""" only used for plotting in main.py """
class properties_evaluator(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.num_comp = property_container.nc
        self.min_z = property_container.min_z
        self.components = property_container.component_name

        self.property = property_container

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                #print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        # Perform normalization over z_caco3 (which is zc[-1]!)
        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        # How to deal with unphysical point for normalization (for fluid components and solid components)?
        if zc[-1] < 0:
            zc = self.comp_out_of_bounds(zc)

        zc_sol = zc[-1]
        zc_fl = zc[:-1] / np.sum(zc[:-1])
        nc_fl = len(zc_fl)

        # Perform flash only for component that appear in both phases, for example, H2O or CO2 but not ions like Ca+2
        # Check if K-values for each fluid component:
        if len(self.property.flash_ev.K_values) == nc_fl:
            V, x, y = self.property.flash_ev.evaluate(zc_fl)

            if 0 < V < 1:
                liq_frac = x
                vap_frac = y
            else:
                liq_frac = zc_fl
                vap_frac = zc_fl

        else:
            # When using normalization instead of small k-values for components only in liquid phase
            inert_ids = 1
            mask = np.ones((nc_fl,), dtype=bool)
            mask[inert_ids] = 0
            zc_flash = zc_fl[mask] / np.sum(zc_fl[mask])
            # zc_flash = zc_fl[mask] / np.sum(zc_fl[mask])

            V_star, x, y = self.property.flash_ev.evaluate(zc_flash)
            V = V_star * (1 - np.sum(zc_fl[mask == 0]))

            liq_frac = np.zeros((nc_fl,))
            vap_frac = np.zeros((nc_fl,))
            vap_frac[mask] = y

            if 0 < V < 1:
                liq_frac[:] = (zc_fl - vap_frac * V) / (1 - V)
                vap_frac[mask] = y
            else:
                liq_frac[:] = zc_fl
                vap_frac[mask] = zc_fl[mask]

        Mt_aq = 0
        Mt_g = 0
        # molar weight of mixture
        for ii in range(nc_fl):

            Mt_aq = Mt_aq + props(self.components[ii], 'Mw') * liq_frac[ii]
            Mt_g = Mt_g + props(self.components[ii], 'Mw') * vap_frac[ii]

        rho_g = 0
        rho_aq = 0

        rho_g_m = 0
        rho_aq_m = 0

        """" PROPERTIES Evaluation """
        rho_s = self.property.density_ev['sol'].evaluate(state)
        rho_s_m = rho_s / props(self.property.component_name[-1], 'Mw')

        if V <= 0:
            sg = 0
            # single aqueous phase
            rho_aq = self.property.density_ev['wat'].evaluate(state)  # output in [kg/m3]
            rho_aq_m = rho_aq / Mt_aq  # [Kmol/m3]
        elif V >= 1:
            sg = 1
            # single vapor phase
            rho_g = self.property.density_ev['gas'].evaluate(state)  # in [kg/m3]
            rho_g_m = rho_g / Mt_g  # [Kmol/m3]
        else:
            # two phases
            rho_aq = self.property.density_ev['wat'].evaluate(state)  # output in [kg/m3]
            rho_g = self.property.density_ev['gas'].evaluate(state)  # in [kg/m3]
            rho_aq_m = rho_aq / Mt_aq  # [Kmol/m3]
            rho_g_m = rho_g / Mt_g  # [Kmol/m3]
            sg = rho_aq_m / (rho_g_m / V - rho_g_m + rho_aq_m)  # saturation using [Kmol/m3]

        ss = zc_sol  # C_m * M_w / rho_s = 1 - phi --> C_m = rho_s_m * z_caco3 --> phi = 1 - z_caco3 --> ss = z_caco3
        return sg, ss, liq_frac, vap_frac, rho_aq, rho_g, rho_aq_m, rho_g_m, rho_s, rho_s_m
