from custom_properties import PropertyContainer, DensityBasic, RockCompressCoef, ViscosityBasic, \
    RelPermBrooksCorey, FlashBasic
import numpy as np
import sys


# Open output file:
sys.stdout = open("kin_properties.txt", "w")

# Define some convergence related parameters:
conv_tol = 1e-16
min_comp = 1e-16

# Components: H20, CO2, Ca+2, CO3-2, CaCO3
# Flash container assuming Vapor-Liquid equilibrium for H2O and CO2
kval_wat = 0.1  # 0.1080
kval_co2 = 100  # 1149

"""Physical properties"""
prop_con_equi = PropertyContainer(phase_name=['Liquid', 'Vapor'], component_name=['H2O', 'CO2', 'Ca+2', 'CO3-2'],
                                  min_z=min_comp)

""" properties correlations """
prop_con_equi.density_ev = dict([('Liquid', DensityBasic(density=1000, compressibility=1e-6)),
                                 ('Vapor', DensityBasic(density=100, compressibility=1e-4)),
                                 ('Solid', DensityBasic(density=2000, compressibility=1e-7))])
prop_con_equi.viscosity_ev = dict([('Liquid', ViscosityBasic(viscosity=1)),
                                   ('Vapor', ViscosityBasic(viscosity=0.1)),
                                   ('Solid', ViscosityBasic(viscosity=0.0))])

""" flash procedure """
composition = np.array([1.9607843e-02, 1.9607843e-02, 1.8241835e-08, 1.8241835e-08, 9.6078428e-01])
# Make sure K-values of Ca+2 and CO3-2 are very low (and consistent), since they don't vaporize
prop_con_equi.flash_ev = FlashBasic(K_values=[0.1, 100, min_comp/composition[2], min_comp/composition[3]])

component_list = ["H2O", "CO2", "Ca+2", "CO3-2", "CaCO3"]
phase_list = ['Liquid', 'Vapor', 'Solid']
mole_frac_list = ['Liquid MoleFrac', 'Vapor MoleFrac', 'Solid MoleFrac']

# Print phase split, for specific initial conditions:
print('Properties based on initial state: ')

# Given some element composition, determine what phase-split is using defined properties:
pressure = 95  # [bar]
composition = np.array([1.9607843e-02, 1.9607843e-02, 1.8241835e-08, 1.8241835e-08, 9.6078428e-01])
state = np.append(pressure, composition[:-1])

print('Initial state = ', state)

# Perform Flash procedure here:
zc_norm = composition[:-1] / sum(composition[:-1])
[v, liq_frac, vap_frac] = prop_con_equi.flash_ev.evaluate(zc_norm)

phase_frac = np.array([(1 - v)*(1 - composition[-1]), v*(1 - composition[-1]), composition[-1]])
liq_frac = np.append(liq_frac, 0)
vap_frac = np.append(vap_frac, 0)
sol_frac = np.array([0, 0, 0, 0, 1])
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
composition = np.array([9.9000000e-01, 1.0000000e-02, 9.9888889e-17, 9.9888889e-17, 1.0000000e-16])
state = np.append(pressure, composition[:-1])  # note that we simplify the model here and take the sum of Ca+2 and
                                               # CO3-2 as the last element, instead of them separately
print('Initial state = ', state)

# Perform Flash procedure here:
zc_norm = composition[:-1] / sum(composition[:-1])
[v, liq_frac, vap_frac] = prop_con_equi.flash_ev.evaluate(zc_norm)

phase_frac = np.array([(1 - v)*(1 - composition[-1]), v*(1 - composition[-1]), composition[-1]])
liq_frac = np.append(liq_frac, 0)
vap_frac = np.append(vap_frac, 0)
sol_frac = np.array([0, 0, 0, 0, 1])
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
