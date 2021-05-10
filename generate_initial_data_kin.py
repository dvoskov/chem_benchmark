from custom_properties import PropertyContainer, DensityBasic, RockCompressCoef, ViscosityBasic, \
    RelPermBrooksCorey, FlashBasic
import numpy as np
import sys


# Open output file:
sys.stdout = open("kin_properties.txt", "w")

# Define some convergence related parameters:
min_comp = 1e-12

# Components: H20, CO2, Ca+2, CO3-2, CaCO3
# Flash container assuming Vapor-Liquid equilibrium for H2O and CO2
kval_wat = 0.1
kval_co2 = 10
kval_ca = 1e-12
kval_co3 = 1e-12
molar_weight = np.array([18.015, 44.01, 40.078, 60.008, 100.086])

"""Physical properties"""
prop_con_kin = PropertyContainer(phase_name=['wat', 'gas'], component_name=['H2O', 'CO2', 'Ca+2', 'CO3-2', 'CaCO3'],
                                  min_z=min_comp)

""" properties correlations """
prop_con_kin.density_ev = dict([('wat', DensityBasic(density=1000, compressibility=1e-6)),
                                ('gas', DensityBasic(density=100, compressibility=1e-4)),
                                ('sol', DensityBasic(density=2000, compressibility=1e-7))])
prop_con_kin.viscosity_ev = dict([('wat', ViscosityBasic(viscosity=1)),
                                  ('gas', ViscosityBasic(viscosity=0.1)),
                                  ('sol', ViscosityBasic(viscosity=0.0))])

""" flash procedure """
# Make sure K-values of Ca+2 and CO3-2 are very low (and consistent), since they don't vaporize
# (or take them out completely)
prop_con_kin.flash_ev = FlashBasic(K_values=[kval_wat, kval_co2, kval_ca, kval_co3])

component_list = ["H2O", "CO2", "Ca+2", "CO3-2", "CaCO3"]
phase_list = ['Liquid', 'Vapor', 'Solid']
phase_list_dict = ['wat', 'gas', 'sol']
mole_frac_list = ['Liquid MoleFrac', 'Vapor MoleFrac', 'Solid MoleFrac']
nc = len(component_list)

# Print phase split, for specific initial conditions:
print('Properties based on injection state: state = [P, z_h2o, z_co2, z_ca, z_co3]')

# Given some composition, determine what phase-split is using defined properties:
pressure = 165  # [bar]
composition = np.array([min_comp, 1 - 4 * min_comp, min_comp, min_comp, min_comp])
state = np.append(pressure, composition[:-1])

print('Injection state = ', state)

# Perform Flash procedure here:
zc_sol = composition[-1]
zc_fluid = composition[:-1] / (1 - zc_sol)
[V, x, y] = prop_con_kin.flash_ev.evaluate(zc_fluid)

liq_frac = np.zeros((nc,))
vap_frac = np.zeros((nc,))
sol_frac = np.array([0, 0, 0, 0, 1])

if 0 < V < 1:
    liq_frac[:-1] = x
    vap_frac[:-1] = y
else:
    liq_frac[:-1] = zc_fluid
    vap_frac[:-1] = zc_fluid

V = min(max(V, 0), 1)
M_aq = 0
M_g = 0
for ii in range(len(zc_fluid)):
    M_aq += x[ii] * molar_weight[ii]
    M_g += y[ii] * molar_weight[ii]
phase_frac = np.array([(1 - V) * (1 - zc_sol), V * (1 - zc_sol), zc_sol])
phase_split_ini = np.array([liq_frac,
                            vap_frac,
                            sol_frac])

print('\t-----------------------------------------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(component_list) + 1)
row_format ="{:>20}" + "{:20.7e}" * (len(component_list))

print(row_header_format.format("", *component_list))
print(row_format.format("Composition, z_c", *composition))

for phase, row in zip(mole_frac_list, phase_split_ini):
    print(row_format.format(phase, *row))

print('\t-----------------------------------------------------------------------------------------------------------------------')

# Print relevant properties (density, viscosity, etc.)
print('\t-------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(phase_list) + 1)
print(row_header_format.format("", *phase_list))

row_format ="{:>20}" + "{:20.7e}" * (len(phase_list))
density_list = np.array([])
viscosity_list = np.array([])

for phase in phase_list_dict:
    density_list = np.append(density_list, prop_con_kin.density_ev[phase].evaluate(state))
    viscosity_list = np.append(viscosity_list, prop_con_kin.viscosity_ev[phase].evaluate(state))

density_molar = np.array([density_list[0] / M_aq, density_list[1] / M_g])
saturation_list = np.append((phase_frac[:-1] / density_molar) / sum((phase_frac[:-1] / density_molar)) * (1 - zc_sol), zc_sol)
sat_list_mobile = np.append(saturation_list[:-1] / np.sum(saturation_list[:-1]), 0)
print(row_format.format('Phase MoleFrac', *phase_frac))
print(row_format.format('Mass Density', *density_list))
print(row_format.format('Viscosity', *viscosity_list))
print(row_format.format('Sat. phi_tot', *saturation_list))
print(row_format.format('Sat. phi_fluid', *sat_list_mobile))
print('\t-------------------------------------------------------------------------------------')
print('\n')

# Print phase split, for specific initial conditions:
print('Properties based on initial state: state = [P, z_h2o, z_co2, z_ca, z_co3]')

# Given some composition, determine what phase-split is using defined properties:
pressure = 95  # [bar]
composition = np.array([0.15, min_comp, 0.075, 0.075, 0.7 - min_comp])
state = np.append(pressure, composition[:-1])  # note that we simplify the model here and take the sum of Ca+2 and
                                               # CO3-2 as the last element, instead of them separately
print('Initial state = ', state)

# Perform Flash procedure here:
zc_sol = composition[-1]
zc_fluid = composition[:-1] / (1 - zc_sol)
[V, x, y] = prop_con_kin.flash_ev.evaluate(zc_fluid)

liq_frac = np.zeros((nc,))
vap_frac = np.zeros((nc,))
sol_frac = np.array([0, 0, 0, 0, 1])

if 0 < V < 1:
    liq_frac[:-1] = x
    vap_frac[:-1] = y
else:
    liq_frac[:-1] = zc_fluid
    vap_frac[:-1] = zc_fluid

V = min(max(V, 0), 1)
M_aq = 0
M_g = 0
for ii in range(len(zc_fluid)):
    M_aq += x[ii] * molar_weight[ii]
    M_g += y[ii] * molar_weight[ii]

phase_frac = np.array([(1 - V)*(1 - composition[-1]), V*(1 - composition[-1]), composition[-1]])
phase_split_ini = np.array([liq_frac,
                            vap_frac,
                            sol_frac])

print('\t-----------------------------------------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(component_list) + 1)
row_format ="{:>20}" + "{:20.7e}" * (len(component_list))

print(row_header_format.format("", *component_list))
print(row_format.format("Composition, z_c", *composition))

for phase, row in zip(mole_frac_list, phase_split_ini):
    print(row_format.format(phase, *row))

print('\t-----------------------------------------------------------------------------------------------------------------------')

# Print relevant properties (density, viscosity, etc.)
print('\t-------------------------------------------------------------------------------------')
row_header_format ="{:>20}" * (len(phase_list) + 1)
print(row_header_format.format("", *phase_list))

row_format ="{:>20}" + "{:20.7e}" * (len(phase_list))
density_list = np.array([])
viscosity_list = np.array([])

for phase in phase_list_dict:
    density_list = np.append(density_list, prop_con_kin.density_ev[phase].evaluate(state))
    viscosity_list = np.append(viscosity_list, prop_con_kin.viscosity_ev[phase].evaluate(state))

density_molar = np.array([density_list[0] / M_aq, density_list[1] / M_g])
saturation_list = np.append((phase_frac[:-1] / density_molar) / sum((phase_frac[:-1] / density_molar)) * (1 - zc_sol), zc_sol)
sat_list_mobile = np.append(saturation_list[:-1] / np.sum(saturation_list[:-1]), 0)
print(row_format.format('Phase MoleFrac', *phase_frac))
print(row_format.format('Mass Density', *density_list))
print(row_format.format('Viscosity', *viscosity_list))
print(row_format.format('Sat. phi_tot', *saturation_list))
print(row_format.format('Sat. phi_fluid', *sat_list_mobile))
print('\t-------------------------------------------------------------------------------------')
print('\n')

sys.stdout.close()
