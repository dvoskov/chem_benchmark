import numpy as np
import copy
from scipy.interpolate import interp1d
from property_evaluator_iface import property_evaluator_iface


#  Main container for custom properties
#  covers basic properties needed for multiphase flow with chemistry
class PropertyContainer(property_evaluator_iface):
    def __init__(self, phase_name, component_name, min_z=1e-8):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.n_phases = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name
        self.min_z = min_z

        # Allocate (empty) evaluators
        self.density_ev = None
        self.rock_compr_ev = None
        self.viscosity_ev = None
        self.rel_perm_ev = None
        self.enthalpy_ev = None
        self.kin_rate_ev = None
        self.flash_ev = None
        self.trans_multi = None


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

class DensityWaterWithCO2(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, molefrac_co2, rho_wat, rho_gas):
        return 1 / (molefrac_co2 / rho_gas + (1 - molefrac_co2) / rho_wat)


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
class RelPermBrooksCorey(property_evaluator_iface):
    def __init__(self, corey_exp, kr_end=1, sat_res=0):
        super().__init__()
        self.exp = corey_exp
        self.sr = sat_res
        self.kre = kr_end

    def evaluate(self, sat):
        return self.kre*(sat - self.sr)**self.exp


#  Flash based on constant K-values
class FlashBasic(property_evaluator_iface):
    def __init__(self, K_values):
        super().__init__()
        # Custom flash class based on simple k-values Rachford-Rice solution
        self.K_values = np.array(K_values)

    def evaluate(self, composition):
        zc = np.array(composition, copy=True)
        eps = 1e-12
        # zc += eps
        # zc /= sum(zc)

        beta_min = 1 / (1 - np.max(self.K_values)) + eps
        beta_max = 1 / (1 - np.min(self.K_values)) - eps
        beta = 0.5 * (beta_min + beta_max)
        tol = 1e-14
        max_iter = 1000
        beta_left = beta_min
        beta_right = beta_max
        converged = False

        for c in range(0, max_iter):
            if np.abs(self.rachford_rice(beta, zc)) < tol or np.abs(beta_left - beta_right) < tol:
                converged = True
                break

            if self.rachford_rice(beta_left, zc) * self.rachford_rice(beta, zc) < 0:
                beta_right = beta
            else:
                beta_left = beta

            beta = 0.5 * (beta_left + beta_right)

        if converged:
            if beta < 0 or beta > 1:
                x = zc
                y = zc
            else:
                x = zc / (1 + beta * (self.K_values - 1))
                y = self.K_values * x
        else:
            x = np.NaN
            y = np.NaN
            print('Flash does not converged!\n')

        return beta, x, y

    def rachford_rice(self, beta, zc):
        return np.sum(zc * (1 - self.K_values) / (1 + beta * (self.K_values - 1)))


class Flash3PhaseLiqVapSol(property_evaluator_iface):
    def __init__(self, K_val_fluid, K_val_chemistry, rate_annihilation_matrix, pres_range=None):
        super().__init__()

        # Custom three phase flash evaluator for equilibrium chemistry with liquid-vapor-solid equilibrium:
        self.K_val_fluid = np.array(K_val_fluid)

        # K-values can be vector (constant K-values) or matrix (pressure dependent K-values):
        self.pres_range = None
        if self.K_val_fluid.ndim > 1:
            self.constant_K = False
            try:
                self.pres_range = pres_range
            except ValueError:
                raise Exception('WARNING: Cannot input K-value matrix without pressure range')

        else:
            self.constant_K = True

        self.K_val_chemistry = np.array(K_val_chemistry)
        self.k_values = None
        self.rate_anni_mat = rate_annihilation_matrix
        self.sum_colums_anni_mat = np.sum(self.rate_anni_mat, axis=0)

    def evaluate(self, state, tol=1e-10, min_comp=1e-10):
        """
        Class method which computes the three-phase flash equilibrium (liq-vap-sol):
        :param state: vector with state related parameters [pressure, element_comp_0, ..., element_comp_N-1]
        :param tol: tolerance for convergence (norm(residual))
        :param min_comp: minimum composition (zero-point)
        :return liq_frac: converged liquid component mole frac
        :return vap_frac: converged vapor component mole frac
        :return sol_frac: converged solid component mole frac
        :return nu: vector with phase fractions (liq, vap, sol)
        """
        state_np = np.asarray(state)
        z_elem = np.append(state_np[1:], [(1 - np.sum(state_np[1:])) / 2, (1 - np.sum(state_np[1:])) / 2])
        z_elem_old = np.array(z_elem, copy=True)
        pres = state_np[0]

        # If constant K-values simply store values as is, otherwise interpolate K-values based on current pressure:
        if self.constant_K:
            self.k_values = self.K_val_fluid.flatten()
        else:
            self.k_values = np.zeros((self.K_val_fluid.shape[0]))
            for ii in range(len(self.k_values)):
                try:
                    self.k_values[ii] = interp1d(self.pres_range, self.K_val_fluid[ii, :])(pres)
                except ValueError:
                    raise Exception('WARNING: Pressure out of K-value range, check if injection rate is not to large')

            self.k_values.flatten()

        # Get more stable initial guess for three-phase flash:
        liq_frac, vap_frac = self.init_three_phase_flash(tol)

        # Check if element composition is physical:
        z_elem = self.get_physical_comp(z_elem, min_comp)

        # Initialize vector of nonlinear unknowns used for the three phase flash:
        nonlin_unkws = self.construct_nonlin_unkwns(liq_frac, vap_frac)

        # NOTE: nonlin_unknws_full = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Compute composition:
        z_comp = self.eval_comp(nonlin_unkws)

        # Compute residual
        res = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

        # Start Newton loop to find root for full system of equations:
        curr_iter = 0
        max_iter = 100
        temp_tol = copy.deepcopy(tol)

        while (np.linalg.norm(res) > temp_tol) and (curr_iter <= max_iter):
            # Compute Jacobian, used every Newton iteration to compute solution to nonlinear system:
            jac = self.compute_jac_full(z_elem, nonlin_unkws)

            # Solve linear system:
            nonlin_upd = -np.linalg.solve(jac, res)

            # Update non-linear unknowns:
            nonlin_unkws += nonlin_upd

            # Recompute composition:
            z_comp = self.eval_comp(nonlin_unkws)

            # Recompute residual equations (vec_residual --> 0 when converged):
            res = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

            # Increment iteration counter:
            curr_iter += 1

            # Relax tolerance if certain number of iterations is reached:
            if (curr_iter == 20) or (curr_iter == 40):
                temp_tol *= 10

        if curr_iter > max_iter:
            raise Exception(
                'WARNING: Three-phase equilibrium did not converge (res={:}, iters={:})'.format(res, curr_iter))

        # Check if in three- or two-phase:
        nonlin_unkws, phase_frac_zero = self.eval_bounds_nonlin_unknws(nonlin_unkws, min_comp)
        system_state = self.determine_sys_state(phase_frac_zero)

        # Perform two-phase flash (if in two-phase region):
        if system_state == '110':
            # Do two-phase vapor-solid flash:
            nonlin_unkws = self.two_phase_vap_sol(z_elem, nonlin_unkws, tol)
        elif system_state == '101':
            # Do two-phase liquid-solid flash:
            nonlin_unkws = self.two_phase_liq_sol(z_elem, nonlin_unkws, tol)
        elif system_state == '011':
            # Do two-phase liquid-vapor flash:
            nonlin_unkws = self.two_phase_liq_vap(z_elem, nonlin_unkws, tol)

        # Check if in two- or single-phase:
        nonlin_unkws, phase_frac_zero = self.eval_bounds_nonlin_unknws(nonlin_unkws, min_comp)
        system_state = self.determine_sys_state(phase_frac_zero)

        if system_state == '100':
            # No liquid and vapor phase present, therefore in state 100:
            # raise Exception('Only solid phase present, check element composition, must be an error somewhere!')
            print('Only solid phase present, check element composition, must be an error somewhere!')
        elif system_state == '001':
            # Single phase liquid, all elements are soluble in water so ze == zc(0:-1):
            nonlin_unkws[0:4] = z_elem[0:4]
        elif system_state == '010':
            # Single phase gas, only first two elements appear in gas phase ze == zc(0:-1):
            nonlin_unkws[4:6] = z_elem[0:2]
        return self.unpack_nonlin_unkws(nonlin_unkws)

    def init_three_phase_flash(self, tol=1e-10):
        """
        Class method which computes an initial guess for the three phase flash, using a two phase (liq-vap) flash
        :return: an intial guess for liquid and vapor comp molefrac
        """
        # Do two-phase flash to have more stable initial guess for three-phase flash:
        # Perform two phase flash to initialize liquid and vapor component mole fraction in the physical region:
        z_comp_dummy = np.array([0.5, 0.5])

        nu_min = 1 / (1 - np.max(self.k_values)) + tol
        nu_max = 1 / (1 - np.min(self.k_values)) - tol
        nu_new = (nu_min + nu_max) / 2

        curr_iter = 0
        max_iter = 100

        while (np.abs(self.rachford_rice(nu_new, z_comp_dummy)) > tol) and curr_iter < max_iter \
                and (np.abs(nu_new - nu_min) > tol) \
                and (np.abs(nu_new - nu_max) > tol):
            # Perform bisection iteration:
            # Check if function is monotonically increasing or decreasing in order to correctly set new interval:
            if (self.rachford_rice(nu_min, z_comp_dummy) *
                    self.rachford_rice(nu_new, z_comp_dummy) <= 0):
                nu_max = nu_new

            else:
                nu_min = nu_new

            # Update new interval center value:
            nu_new = (nu_max + nu_min) / 2

            # Increment iteration:
            curr_iter += 1

        # Set vapor and liquid component mole fractions based on physical region:
        if curr_iter < max_iter:
            liq_frac = z_comp_dummy / (nu_new * (self.k_values - 1) + 1)
            vap_frac = self.k_values * liq_frac
        else:
            print('No converged initial guess found!!!\n')
            liq_frac = np.array([0.7, 0.3])
            vap_frac = np.array([0.3, 0.7])

        return liq_frac, vap_frac

    def rachford_rice(self, nu, z_comp):
        return np.sum(z_comp * (1 - self.k_values) / (1 + (self.k_values - 1) * nu))

    def get_physical_comp(self, z_comp, min_comp=1e-10):
        """
        Class method which computes if element total composition is out of physical bounds
        :param z_comp: element composition (depends on state)
        :param min_comp: zero-point for composition
        :return: z_comp: element composition (depends on state)
        """
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_index = z_comp <= min_comp
        if temp_index.any():
            # At least one truth value in the boolean array:
            z_comp[temp_index] = min_comp

        return (z_comp / np.sum(z_comp))

    def two_phase_vap_sol(self, z_elem, nonlin_unkws, tol=1e-10):
        """
        Class method which solves a two-phase flash for vapor-solid equilibrium
        :param z_elem: element composition (depends on state)
        :param nonlin_unkws: full vector of nonlinear unknowns
        :return nonlin_unkws: (possibly) updated nonlinear unknowns and updates state denoter
        """
        # Do two-phase vapor-solid flash:
        # Reduced set of unknowns:
        #            4      5      7       8
        # --- X = [y_h2o, y_co2, nu_vap, nu_sol]        (PYTHON COUNTING!)
        # Map to :   0      1      2       3            --> size (4,)
        # Equations used: eq_0  eq_1  eq_7  eq_8
        # Map to:           0     1     2    3          --> size (4, 4)
        # Map full set of unknowns to reduced set:
        nonlin_unkws_red = np.zeros((4,))
        nonlin_unkws_red[0:2] = nonlin_unkws[4:6]
        nonlin_unkws_red[2:] = nonlin_unkws[7:]

        # Compute initial residual equations:
        z_comp = self.eval_comp(nonlin_unkws)
        dummy_residual = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

        # Map full residual to reduced residual:
        res_red = np.zeros((4,))
        res_red[0:2] = dummy_residual[0:2]
        res_red[2:] = dummy_residual[7:]

        curr_iter = 0
        max_iter = 100
        temp_tol = copy.deepcopy(tol)

        while (np.linalg.norm(res_red) > temp_tol) and (curr_iter < max_iter):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            jac_full = self.compute_jac_full(z_elem, nonlin_unkws)

            # Map full jacobian to reduced jacobian, from (9, 9) to (4, 4):
            jac_red = np.zeros((4, 4))
            jac_red[0:2, 0:2] = jac_full[0:2, 4:6]
            jac_red[0:2, 2:] = jac_full[0:2, 7:]
            jac_red[2:, 0:2] = jac_full[7:, 4:6]
            jac_red[2:, 2:] = jac_full[7:, 7:]

            # Solve linear system:
            nonlin_unkws_red_upd = -np.linalg.solve(jac_red, res_red)

            # Update non-linear unknowns:
            nonlin_unkws_red += nonlin_unkws_red_upd

            # Map back to full solution:
            nonlin_unkws[4:6] = nonlin_unkws_red[0:2]
            nonlin_unkws[7:] = nonlin_unkws_red[2:]

            # Recompute composition:
            z_comp = self.eval_comp(nonlin_unkws)

            # Recompute residual:
            res_full = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

            # Map full residual to reduced residual:
            res_red = np.zeros((4,))
            res_red[0:2] = res_full[0:2]
            res_red[2:] = res_full[7:]

            # Increment iteration counter:
            curr_iter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (curr_iter == 20) or (curr_iter == 40):
                temp_tol *= 10

        return nonlin_unkws

    def two_phase_liq_sol(self, z_elem, nonlin_unkws, tol=1e-10):
        """
        Class method which solves a two-phase flash for liquid-solid equilibrium
        :param z_elem: element composition (depends on state)
        :param nonlin_unkws: full vector of nonlinear unknowns
        :return nonlin_unkws: (possibly) updated nonlinear unknowns and updates state denoter
        """
        # Reduced set of unknowns:
        # -- X = [x_h2o, x_co2, x_co3, x_ca, nu_liq, nu_sol]    (PYTHON COUNTING!)
        # Map to:   0      1      2     3      6       8        --> size (6,)
        # Equations used: eq_0  eq_1  eq_2  eq_3  eq_6  eq_7
        # Map to:           0     1     2    3     4     5      --> size (6, 6)
        # Map full set of unknowns to reduced set:
        nonlin_unkws_red = np.zeros((6,))
        nonlin_unkws_red[0:4] = nonlin_unkws[0:4]
        nonlin_unkws_red[4] = nonlin_unkws[6]
        nonlin_unkws_red[5] = nonlin_unkws[8]

        # Compute full residual:
        z_comp = self.eval_comp(nonlin_unkws)
        res_full = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

        # Map full residual to reduced residual:
        res_red = np.zeros((6,))
        res_red[0:4] = res_full[0:4]
        res_red[4] = res_full[6]
        res_red[5] = res_full[7]

        curr_iter = 0
        max_iter = 100
        temp_tol = copy.deepcopy(tol)

        while (np.linalg.norm(res_red) > temp_tol) and (curr_iter < max_iter):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            jac_full = self.compute_jac_full(z_elem, nonlin_unkws)

            # Map full jacobian to reduced jacobian, from (9, 9) to (6, 6):
            jac_red = np.zeros((6, 6))
            jac_red[0:4, 0:4] = jac_full[0:4, 0:4]
            jac_red[0:4, 4] = jac_full[0:4, 6]
            jac_red[0:4, 5] = jac_full[0:4, 8]
            jac_red[4, 0:4] = jac_full[6, 0:4]
            jac_red[4, 4] = jac_full[6, 6]
            jac_red[4, 5] = jac_full[6, 8]
            jac_red[5, 0:4] = jac_full[7, 0:4]
            jac_red[5, 4] = jac_full[7, 6]
            jac_red[5, 5] = jac_full[7, 8]

            # Solve linear system:
            nonlin_unkws_red_upd = -np.linalg.solve(jac_red, res_red)

            # Update non-linear unknowns:
            nonlin_unkws_red += nonlin_unkws_red_upd

            # Map back to full solution:
            nonlin_unkws[0:4] = nonlin_unkws_red[0:4]
            nonlin_unkws[6] = nonlin_unkws_red[4]
            nonlin_unkws[8] = nonlin_unkws_red[5]

            # Recompute composition:
            z_comp = self.eval_comp(nonlin_unkws)

            # Recompute residual:
            res_full = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

            # Map full residual to reduced residual:
            res_red = np.zeros((6,))
            res_red[0:4] = res_full[0:4]
            res_red[4] = res_full[6]
            res_red[5] = res_full[7]

            # Increment iteration counter:
            curr_iter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (curr_iter == 20) or (curr_iter == 40):
                temp_tol *= 10

        return nonlin_unkws

    def two_phase_liq_vap(self, z_elem, nonlin_unkws, tol=1e-10):
        """
        Class method which solves a two-phase flash for liquid-vapor equilibrium
        :param z_elem: element composition (depends on state)
        :param nonlin_unkws: full vector of nonlinear unknowns
        :return nonlin_unkws: (possibly) updated nonlinear unknowns and updates state denoter
        """
        # Reduced set of unknowns:
        #            0      1      2       3    4      5      6       7
        # --- X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap]     (PYTHON COUNTING!)
        # Map to :   0      1      2       3    4      5      6       7         --> size (8,)
        # Equations used: eq_0  eq_1  eq_2  eq_3  eq_4  eq_5  eq_7  eq_8
        # Map to:           0     1     2    3     4     5      6     7         --> size (8, 8)
        # Map full set of unknowns to reduced set:
        nonlin_unkws_red = nonlin_unkws[:-1]

        # Compute full residual:
        z_comp = self.eval_comp(nonlin_unkws)
        res_full = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

        # Map full residual to reduced residual:
        res_red = np.zeros((8,))
        res_red[0:6] = res_full[0:6]
        res_red[6:] = res_full[7:]

        curr_iter = 0
        max_iter = 100
        temp_tol = copy.deepcopy(tol)

        while (np.linalg.norm(res_red) > temp_tol) and (curr_iter < max_iter):
            # Compute full Jacobian, used every Newton iteration to compute solution to nonlinear system:
            jac_full = self.compute_jac_full(z_elem, nonlin_unkws)

            # Map full jacobian to reduced jacobian, from (9, 9) to (6, 6):
            jac_red = np.zeros((8, 8))
            jac_red[0:6, 0:6] = jac_full[0:6, 0:6]
            jac_red[0:6, 6:] = jac_full[0:6, 6:-1]
            jac_red[6:, 0:6] = jac_full[7:, 0:6]
            jac_red[6:, 6:] = jac_full[7:, 6:-1]

            # Solve linear system:
            nonlin_unkws_red_upd = -np.linalg.solve(jac_red, res_red)

            # Update non-linear unknowns:
            nonlin_unkws_red += nonlin_unkws_red_upd

            # Map back to full solution:
            nonlin_unkws[:-1] = nonlin_unkws_red

            # Recompute composition:
            z_comp = self.eval_comp(nonlin_unkws)

            # Recompute residual:
            res_full = self.compute_res_full(z_elem, z_comp, nonlin_unkws)

            # Map full residual to reduced residual:
            res_red = np.zeros((8,))
            res_red[0:6] = res_full[0:6]
            res_red[6:] = res_full[7:]

            # Increment iteration counter:
            curr_iter += 1

            # Increase tolerance if certain number of iterations is reached:
            if (curr_iter == 20) or (curr_iter == 40):
                temp_tol *= 10

        return nonlin_unkws

    def compute_res_full(self, z_elem, z_comp, nonlin_unkws):
        """
        Class method which constucts the Residual equations for the full system, containing equations for:
        - component to element mole conservation
        - phase equilibrium equations (liq-vap-sol)
        - chemical equilibrium equations (dissolution CaCO3)
        :param vec_element_comp: element composition (depends on state)
        :param vec_component_comp: component composition (depends on state)
        :param vec_nonlin_unknowns: vector of nonlinear unknowns
        :return: set of residual equations for full system
        """
        # NOTE: nonlin_unknws_full = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Compute residual vector for full set of equations:
        residual = np.zeros((9,))

        # First three equation are z_e*sum{E*z_c} - E*z_c = 0
        residual[0:3] = z_elem[0:-1] * np.sum(np.dot(self.rate_anni_mat, z_comp)) - \
                        np.dot(self.rate_anni_mat[:-1, :], z_comp)

        # Fourth (Python==3) Equation for sum of composition is zero (1 - sum{z_c} = 0):
        residual[3] = 1 - np.sum(z_comp)

        # Fifth and Sixth Equation are fugacity constraints for water and co2 fractions (Kx - y = 0):
        residual[4:6] = self.k_values * nonlin_unkws[0:2] - nonlin_unkws[4:6]

        # Seventh Equation is chemical equilibrium (K - Q = 0)
        # NOTE: Only one chemical equilibrium reaction, involving Ca+2 and CO3-2 meaning that if initial and boundary
        # conditions for Ca+2 and CO3-2 are the same, they will remain through the simulation:
        # More general case would be:
        #   residual[6] = (55.508**2) * vec_liquid_molefrac_full[2] * vec_liquid_molefrac_full[3]  -
        #                       sca_k_caco3 * (vec_liquid_molefrac_full[0]**2)
        # residual[6] = 55.508 * nonlin_unkws[2] - np.sqrt(self.K_val_chemistry[0]) * nonlin_unkws[0]
        residual[6] = nonlin_unkws[2] * nonlin_unkws[3] - self.K_val_chemistry[0]

        # Eighth Equation is the sum of the phase fractions equal to 1 (1 - sum{nu_i} = 0):
        residual[7] = 1 - np.sum(nonlin_unkws[6:])

        # Ninth Equation is the sum of liquid - vapor component fraction should be zero (sum{x_i - y_i} = 0):
        residual[8] = np.sum(nonlin_unkws[0:4]) - np.sum(nonlin_unkws[4:6])
        return residual

    def compute_jac_full(self, z_elem, nonlin_unkws):
        """
        Class method which constucts the Jacobian matrix for the full system, containing equations for:
        - component to element mole conservation
        - phase equilibrium equations (liq-vap-sol)
        - chemical equilibrium equations (dissolution CaCO3)
        :param vec_element_comp: element composition (depends on state)
        :param vec_nonlin_unknowns: vector of nonlinear unknowns
        :return: Jacobian matrix of partial derivatives of residual w.r.t. nonlinear unknowns
        """
        # NOTE: nonlin_unknws_full = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        # Compute jacobian matrix for full set of equations:
        jacobian = np.zeros((9, 9))

        # -----------------------------------------------------------------------------------------
        # Compute first three rows of jacobian matrix (res: z_e*sum{E*z_c} - E*z_c = 0):
        # -----------------------------------------------------------------------------------------
        # First four columns w.r.t. liquid component mole fractions:
        for ithCol in range(0, 4):
            jacobian[0:3, ithCol] = z_elem[0:-1] * self.sum_colums_anni_mat[ithCol] * nonlin_unkws[6] - \
                                    self.rate_anni_mat[0:3, ithCol] * nonlin_unkws[6]
        # Following two columns w.r.t. vapor component mole fractions:
        for ithCol in range(4, 6):
            jacobian[0:3, ithCol] = z_elem[0:-1] * self.sum_colums_anni_mat[ithCol - 4] * \
                                    nonlin_unkws[7] - self.rate_anni_mat[0:3, ithCol - 4] * nonlin_unkws[7]

        # Following three columns w.r.t. phase mole fractions:
        jacobian[0:3, 6] = z_elem[0:-1] * np.dot(self.sum_colums_anni_mat[0:4], nonlin_unkws[0:4]) - \
                           np.dot(self.rate_anni_mat[0:3, 0:4], nonlin_unkws[0:4])  # w.r.t. liquid phase frac
        jacobian[0:3, 7] = z_elem[0:-1] * np.dot(self.sum_colums_anni_mat[0:2], nonlin_unkws[4:6]) - \
                           np.dot(self.rate_anni_mat[0:3, 0:2], nonlin_unkws[4:6])  # w.r.t vapor phase frac
        jacobian[0:3, 8] = z_elem[0:-1] * self.sum_colums_anni_mat[4] - self.rate_anni_mat[0:3, 4]  # solid frac

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for overall composition constraint (1 - sum{z_c} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[3, 0:4] = -nonlin_unkws[6]
        jacobian[3, 4:6] = -nonlin_unkws[7]
        jacobian[3, 6] = -np.sum(nonlin_unkws[0:4])
        jacobian[3, 7] = -np.sum(nonlin_unkws[4:6])
        jacobian[3, 8] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for fugacity constraints for water and co2 fractions (Kx - y = 0):
        # -----------------------------------------------------------------------------------------
        # k_h2o*x_h2o - y_h2o = 0:
        jacobian[4, 0] = self.k_values[0]
        jacobian[4, 4] = -1
        # k_co2*x_co2 - y_co2 = 0:
        jacobian[5, 1] = self.k_values[1]
        jacobian[5, 5] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for chemical equilibrium constraint (K - Q = 0):
        # -----------------------------------------------------------------------------------------
        # NOTE: See computation of residual, this is not the general case!!!!
        # jacobian[6, 0] = -np.sqrt(self.K_val_chemistry[0])
        # jacobian[6, 2] = 55.508
        jacobian[6, 2] = nonlin_unkws[3]
        jacobian[6, 3] = nonlin_unkws[2]

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for phase fractions equal to 1 (1 - sum{nu_i} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[7, 6:] = -1

        # -----------------------------------------------------------------------------------------
        # Compute jacobian row for sum of liquid - vapor component fraction should be zero (sum{x_i - y_i} = 0):
        # -----------------------------------------------------------------------------------------
        jacobian[8, 0:4] = 1
        jacobian[8, 4:6] = -1
        return jacobian

    def unpack_nonlin_unkws(self, nonlin_unkws):
        """
        Class method which unpacks the vector of nonlinear unkowns (liq-vap-sol):
        :param nonlin_unkws: vector with nonlinear unknowns used in nonlinear Newton loop
        :return liq_frac: converged liquid component mole frac
        :return vap_frac: converged vapor component mole frac
        :return sol_frac: converged solid component mole frac
        :return nu: vector with phase fractions (liq, vap, sol)
        """
        liq_frac = copy.deepcopy(nonlin_unkws[0:4])
        liq_frac = np.append(liq_frac, [0])

        vap_frac = copy.deepcopy(nonlin_unkws[4:6])
        vap_frac = np.append(vap_frac, [0, 0, 0])

        sol_frac = np.array([0, 0, 0, 0, 1])

        nu = copy.deepcopy(nonlin_unkws[6:])
        return liq_frac, vap_frac, sol_frac, nu

    def determine_sys_state(self, phase_frac_zero):
        """
        Class method which using information from state to update state denoter
        :param temp_index:
        :return str_state_denoter: updated state denoter on current state of system
        """
        # Determine if system is in three-phase or less:
        # Phase denotation system according to AD-GPRS logic for three-phase system:
        # Water = 0, Oil = 1, Gas = 2 || Liquid = 0, Vapor = 1, Solid = 2
        # system_state 000 001 010 100 011 101 110 111
        # phase_0       -   x   -   -   x   x   -   x
        # phase_1       -   -   x   -   x   -   x   x
        # phase_2       -   -   -   x   -   x   x   x
        # Since zero phases present is not possible in our system, there are 2^{n}-1 states possible!
        if phase_frac_zero[0] and phase_frac_zero[1]:
            # No liquid and vapor phase present, therefore in state 100:
            system_state = '100'
        elif phase_frac_zero[0] and phase_frac_zero[2]:
            # No liquid and solid phase present, therefore in state: 010:
            system_state = '010'
        elif phase_frac_zero[1] and phase_frac_zero[2]:
            # No vapor and solid phase present, therefore in state: 001:
            system_state = '001'
        elif phase_frac_zero[0]:
            # No liquid phase present, therefore at least in state 110:
            system_state = '110'
        elif phase_frac_zero[1]:
            # No vapor phase present, therefore at least in state 101:
            system_state = '101'
        elif phase_frac_zero[2]:
            # No solid phase present, therefore at least in state 011:
            system_state = '011'
        else:
            # Pure three phase system:
            system_state = '111'
        return system_state

    def eval_bounds_nonlin_unknws(self, nonlin_unkws, min_comp=1e-10):
        """
        Class method which evaluate if the nonlinear uknowns are out of physical bounds
        :param nonlin_unkws: vector with nonlinear unknowns for Newton-Loop
        :return nonlin_unkws: "                                          "
        :return temp_index: boolean vector containing true for each phase not present
        """
        # Check for negative values in the liquid and vapor component fractions as well as phase fractions:
        temp_index = nonlin_unkws <= min_comp
        if temp_index.any():
            # At least one truth value in the boolean array:
            nonlin_unkws[temp_index] = min_comp

            # Rescale all variables so that they sum to one:
            nonlin_unkws[0:4] = nonlin_unkws[0:4] / np.sum(nonlin_unkws[0:4])
            nonlin_unkws[4:6] = nonlin_unkws[4:6] / np.sum(nonlin_unkws[4:6])
            nonlin_unkws[6:] = nonlin_unkws[6:] / np.sum(nonlin_unkws[6:])
        return nonlin_unkws, temp_index[6:]

    def eval_comp(self, nonlin_unkws):
        """
        Class method which evaluates component total composition
        :return vec_component_comp: vector with component composition
        """
        z_comp = np.append(nonlin_unkws[0:4], [0]) * nonlin_unkws[6] + \
                           np.append(nonlin_unkws[4:6], [0, 0, 0]) * nonlin_unkws[7] + \
                           np.array([0, 0, 0, 0, 1]) * nonlin_unkws[8]
        return z_comp

    def construct_nonlin_unkwns(self, liq_frac, vap_frac, min_comp=1e-10):
        """
        Class methods which constructs the initial vector of nonlinear unknowns according:
        # NOTE: vec_nonlin_unknowns = X = [x_h2o, x_co2, x_co3, x_ca, y_h2o, y_co2, nu_liq, nu_vap, nu_sol]
        # ---------- Python Numbering:       0      1      2      3     4      5      6       7       8
        Based on initial guess!
        :param two_phase_liq_molefrac: initial guess in physical region for liquid component molefractions
        :param two_phase_vap_molefrac: initial guess in physical region for vapor component molefractions
        :return vec_nonlin_unknowns: vector with nonlinear unknowns used in nonlinear Newton loop
        """
        liq_frac = np.append(liq_frac, [min_comp, min_comp, 0])
        vap_frac = np.append(vap_frac, [0, 0, 0])
        phase_frac = np.array([0.5, 0.5, min_comp])
        nonlin_unkws = np.append(np.append(liq_frac[0:-1], vap_frac[0:2]), phase_frac)
        return nonlin_unkws
