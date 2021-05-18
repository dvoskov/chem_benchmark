from darts.physics import *
# from rrsolver import RachfordRice, index_vector as i_v, value_vector as v_v
import numpy as np
# from numba import jit


#  Main container for custom properties
#  covers basic properties needed for multiphase flow with chemistry
class property_container(property_evaluator_iface):
    def __init__(self, phase_name, component_name, min_z=1e-11, diff_coef=0, rock_comp=1e-5,
                 kin_rate_cte=0, equi_prod=1):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.n_phases = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name
        self.min_z = min_z

        self.diff_coef = diff_coef
        self.rock_comp = rock_comp
        self.p_ref = 1
        self.kin_rate_cte = kin_rate_cte
        self.equi_prod = equi_prod

        # Allocate (empty) evaluators
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.rel_well_perm_ev = []
        self.flash_ev = 0


#  Density dependent on compressibility only
class DensityBasic(property_evaluator_iface):
    def __init__(self, density, compressibility=0, p_ref=1):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.density_rc = density
        self.cr = compressibility
        self.p_ref = p_ref

    def evaluate(self, state):
        pres = state[0]
        return self.density_rc * (1 + self.cr * (pres - self.p_ref))


class FracFlowWat(property_evaluator_iface):
    def __init__(self, visc_ratio):
        super().__init__()
        # Density evaluator class based on simple first order compressibility approximation (Taylor expansion)
        self.M = visc_ratio  # mu_wat / mu_oil, usually around 10?

    def evaluate(self, s_w):
        return s_w ** 2 / (s_w ** 2 + (1 - s_w) ** 2 * self.M)


#  Viscosity dependent on viscosibility only
class ViscosityBasic(property_evaluator_iface):
    def __init__(self, viscosity, viscosibility=0, p_ref=1):
        super().__init__()
        # Viscosity evaluator class based on simple first order approximation (Taylor expansion)
        self.viscosity_rc = viscosity
        self.dvis = viscosibility
        self.p_ref = p_ref

    def evaluate(self, state):
        pres = state[0]
        return self.viscosity_rc * (1 + self.dvis * (pres - self.p_ref))


#  Relative permeability based on Corey function
class RelPermBasic(property_evaluator_iface):
    def __init__(self, corey_exp, sr=0):
        super().__init__()
        self.exp = corey_exp
        self.sr = sr

    def evaluate(self, sat):
        return (sat - self.sr)**self.exp


#  Relative permeability based on Corey function
class RelPermBrooksCorey(property_evaluator_iface):
    def __init__(self, corey_exp, kr_end=1, sat_res=0):
        super().__init__()
        self.exp = corey_exp
        self.sr = sat_res
        self.kre = kr_end

    def evaluate(self, sat):
        return self.kre*(sat - self.sr)**self.exp


class RockCompressCoef(property_evaluator_iface):
    def __init__(self, compressibility=0, p_ref=1):
        super().__init__()
        # Rock compres. factor evaluator class based on simple first order
        # compressibility approximation (Taylor expansion)
        self.cr = compressibility
        self.p_ref = p_ref

    def evaluate(self, state):
        pres = state[0]
        return 1 + self.cr * (pres - self.p_ref)


from numba import jit
@jit(nopython=True)
def RR_func(zc, k, eps):
    a = 1 / (1 - np.max(k)) + eps
    b = 1 / (1 - np.min(k)) - eps

    max_iter = 200  # use enough iterations for V to converge
    for ii in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (k - 1) / (V * (k - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if ii >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (k - 1) + 1)
    y = k * x

    return (V, x, y)


#  Flash based on constant K-values
class FlashBasic(property_evaluator_iface):
    def __init__(self, K_values, min_z=1e-11):
        super().__init__()
        # Custom flash class based on simple k-values Rachford-Rice solution
        self.K_values = np.array(K_values)
        self.min_z = min_z

    def evaluate(self, composition):
        # return self.RR(composition)
        return RR_func(composition, self.K_values, self.min_z)
        #
        # return self.RR_cpp(composition, self.K_values)

    def RR(self, zc):
        eps = self.min_z
        a = 1 / (1 - np.max(self.K_values)) + eps
        b = 1 / (1 - np.min(self.K_values)) - eps

        max_iter = 200  # use enough iterations for V to converge
        for ii in range(1, max_iter):
            V = 0.5 * (a + b)
            r = np.sum(zc * (self.K_values - 1) / (V * (self.K_values - 1) + 1))
            if abs(r) < 1e-8:
                break

            if r > 0:
                a = V
            else:
                b = V

        if ii >= max_iter:
            print("Flash warning!!!")

        x = zc / (V * (self.K_values - 1) + 1)
        y = self.K_values * x
        return (V, x, y)

    # def RR_cpp(self, zc, k):
    #     ph_i = i_v([1, 0])
    #     nPhases = len(ph_i)
    #     nc = len(k)
    #     rr = RachfordRice(nPhases)
    #     eps = 1e-14
    #     nu = v_v([0] * nPhases)
    #     k_vals = v_v(k)
    #     zc_vals = v_v(zc)
    #     rr.solveRachfordRice(nPhases, ph_i, nc, k_vals, zc_vals, nu, eps)
    #
    #     V = nu[0]
    #     x = zc / (V * (k - 1) + 1)
    #     y = k * x
    #
    #     return V, x, y


def props(component, property):
    if 1:
        properties = [["CO2", "H2O", "Ca+2", "CO3-2", "CaCO3", "Ions"],  # component
                      [44.01, 18.015, 40.078, 60.008, 100.086, (40.078 + 60.008)/2]]  # molecular mass [g/mol]
    else:
        properties = [["CO2", "H2O", "Ca+2", "CO3-2", "CaCO3", "Ions"],  # component
                      [1, 1, 1, 1, 1, 1]]  # molecular mass [g/mol]

    prop = ["Mw"]
    index1 = prop.index(property) + 1
    index2 = properties[0][:].index(component)
    return properties[index1][index2]


#  Chemistry for simplest kinetic rate
class ChemistryBasic(property_evaluator_iface):
    def __init__(self, kin_rate, wat_molal, equi_prod, stoich_matrix, diff_coef_tensor, components,
                 min_surf_area=1, min_z=1e-8, order_react=1):
        super().__init__()
        # Simple kinetic rate evaluator class
        self.kin_rate = kin_rate
        self.min_surf_area = min_surf_area
        self.order_react = order_react
        self.wat_molal = wat_molal
        self.equi_prod = equi_prod
        self.stoich_matrix = np.array(stoich_matrix)
        self.z_min = min_z
        self.diff_coef_tensor = diff_coef_tensor
        self.components = components

    def evaluate(self, sat_sol, liq_frac):
        # Calculate kinetic rate
        ca_index = self.components.index('Ca+2')
        co3_index = self.components.index('CO3-2')
        co2_index = self.components.index('CO2')
        h2o_index = self.components.index('H2O')
        sol_index = self.components.index('CaCO3')

        sol_prod = self.wat_molal**2 * liq_frac[ca_index] * liq_frac[co3_index] / (liq_frac[h2o_index]**2)

        kinetic_rate = -self.min_surf_area * (sat_sol - self.z_min) * self.kin_rate * \
            (1 - (sol_prod/self.equi_prod)**self.order_react) * liq_frac[co2_index]

        kinetic_rate = -self.kin_rate * liq_frac[co2_index] * (sat_sol - self.z_min)
        return kinetic_rate * self.stoich_matrix
