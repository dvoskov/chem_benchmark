from custom_properties import PropertyContainer, DensityBasic, RockCompressCoef, ViscosityBasic, \
    RelPermBrooksCorey, Flash3PhaseLiqVapSol
import numpy as np
import sys


# Open output file:
sys.stdout = open("equi_properties.txt", "w")

# Define some convergence related parameters:
conv_tol = 1e-16
min_comp = 1e-16

# Set the equilibrium constant for the chemical equilibrium:
kval_caco3 = 3.36 * 10 ** (-9)

# Define rate-annihilation matrix E:
# Components: H20, CO2, Ca+2, CO3-2, CaCO3
# Elements: H20, CO2, Ca+2, CO3-2
mat_rate_annihilation = np.array([[1, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 1]])

# Flash container assuming Vapor-Liquid equilibrium
kval_wat = 0.1  # 0.1080
kval_co2 = 100  # 1149

"""Physical properties"""
prop_con_equi = PropertyContainer(phase_name=['Liquid', 'Vapor'], component_name=['H2O', 'CO2', 'Ca+2_and_CO3-2'],
                                  min_z=min_comp)

""" properties correlations """
prop_con_equi.density_ev = dict([('Liquid', DensityBasic(density=1000, compressibility=1e-6)),
                                 ('Vapor', DensityBasic(density=100, compressibility=1e-4)),
                                 ('Solid', DensityBasic(density=2000, compressibility=1e-7))])
prop_con_equi.viscosity_ev = dict([('Liquid', ViscosityBasic(viscosity=1)),
                                   ('Vapor', ViscosityBasic(viscosity=0.1)),
                                   ('Solid', ViscosityBasic(viscosity=0.0))])

""" flash procedure """
prop_con_equi.flash_ev = Flash3PhaseLiqVapSol(K_val_fluid=[kval_wat, kval_co2],
                                              K_val_chemistry=[kval_caco3],
                                              rate_annihilation_matrix=mat_rate_annihilation,
                                              pres_range=None)

component_list = ["H2O", "CO2", "Ca+2", "CO3-2", "CaCO3"]
phase_list = ['Liquid', 'Vapor', 'Solid']
mole_frac_list = ['Liquid MoleFrac', 'Vapor MoleFrac', 'Solid MoleFrac']

# Print phase split, for specific initial conditions:
print('Properties based on initial state: ')

# Given some element composition, determine what phase-split is using defined properties:
pressure = 95  # [bar]
composition = np.array([0.01, 0.01, 0.98])  # element composition (see description above)
state = np.append(pressure, composition[:-1])  # note that we simplify the model here and take the sum of Ca+2 and
                                               # CO3-2 as the last element, instead of them separately
print('Initial state = ', state)

liq_frac, vap_frac, sol_frac, phase_frac = prop_con_equi.flash_ev.evaluate(state, conv_tol, min_comp)
zc_ini = liq_frac * phase_frac[0] + vap_frac * phase_frac[1] + sol_frac * phase_frac[2]
phase_split_ini = np.array([liq_frac,
                            vap_frac,
                            sol_frac])

print('\t-----------------------------------------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(component_list) + 1)
row_format ="{:>20}" + "{:20.7e}" * (len(component_list))

print(row_header_format.format("", *component_list))
print(row_format.format("Composition, z_c", *zc_ini))

for phase, row in zip(mole_frac_list, phase_split_ini):
    print(row_format.format(phase, *row))

print('\t-----------------------------------------------------------------------------------------------------------------------')

# Print relevant properties (density, viscosity, etc.)
print('\t-------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(phase_list) + 1)
print(row_header_format.format("", *phase_list))

row_format ="{:>20}" + "{:20.7e}" * (len(phase_list))
density_list = []
viscosity_list = []

for phase in phase_list:
    density_list.append(prop_con_equi.density_ev[phase].evaluate(state))
    viscosity_list.append(prop_con_equi.viscosity_ev[phase].evaluate(state))

saturation_list = (phase_frac / density_list) / sum((phase_frac / density_list))
sat_list_mobile = np.append(saturation_list[:-1] / sum(saturation_list[:-1]), 0)
print(row_format.format('Phase MoleFrac', *phase_frac))
print(row_format.format('Density', *density_list))
print(row_format.format('Viscosity', *viscosity_list))
print(row_format.format('Sat. phi_tot', *saturation_list))
print(row_format.format('Sat. phi_fluid', *sat_list_mobile))
print('\t-------------------------------------------------------------------------------------')
print('\n')

# Print phase split, for specific initial conditions:
print('Properties based on injection state: ')

# Given some element composition, determine what phase-split is using defined properties:
pressure = 125  # [bar]
composition = np.array([0.99, 0.01, 0])  # element composition (see description above)
state = np.append(pressure, composition[:-1])  # note that we simplify the model here and take the sum of Ca+2 and
                                               # CO3-2 as the last element, instead of them separately
print('Initial state = ', state)

liq_frac, vap_frac, sol_frac, phase_frac = prop_con_equi.flash_ev.evaluate(state, conv_tol, min_comp)
zc_ini = liq_frac * phase_frac[0] + vap_frac * phase_frac[1] + sol_frac * phase_frac[2]
phase_split_ini = np.array([liq_frac,
                            vap_frac,
                            sol_frac])

print('\t-----------------------------------------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(component_list) + 1)
row_format ="{:>20}" + "{:20.7e}" * (len(component_list))

print(row_header_format.format("", *component_list))
print(row_format.format("Composition, z_c", *zc_ini))

for phase, row in zip(mole_frac_list, phase_split_ini):
    print(row_format.format(phase, *row))

print('\t-----------------------------------------------------------------------------------------------------------------------')

# Print relevant properties (density, viscosity, etc.)
print('\t-------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(phase_list) + 1)
print(row_header_format.format("", *phase_list))

row_format ="{:>20}" + "{:20.7e}" * (len(phase_list))
density_list = []
viscosity_list = []

for phase in phase_list:
    density_list.append(prop_con_equi.density_ev[phase].evaluate(state))
    viscosity_list.append(prop_con_equi.viscosity_ev[phase].evaluate(state))

saturation_list = (phase_frac / density_list) / sum((phase_frac / density_list))
sat_list_mobile = np.append(saturation_list[:-1] / sum(saturation_list[:-1]), 0)
print(row_format.format('Phase MoleFrac', *phase_frac))
print(row_format.format('Density', *density_list))
print(row_format.format('Viscosity', *viscosity_list))
print(row_format.format('Sat. phi_tot', *saturation_list))
print(row_format.format('Sat. phi_fluid', *sat_list_mobile))
print('\t-------------------------------------------------------------------------------------')
print('\n')

sys.stdout.close()
