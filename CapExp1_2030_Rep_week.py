#%%

import numpy as np
from datetime import datetime
import xarray as xr
import sys
import core_func
import matplotlib as plt
import pandas as pd
import logging
import pypsa
import networkx as nx
logging.basicConfig(level=logging.INFO)


# %%
# Lets create a network
network = pypsa.Network()

# INPUTS - Fill in as desired
scenario = "BAU"
hours_in_opt = 672
peak_weighting = 5
offpeak_weighting = 45
# months_to_optimise = 6
solver_name = "gurobi"  # Can be gurobi or GLPK
candi_hourly_match_portion = 0.25 # portion of C&Is that hourly or annually match
investment_years = [2030,2035,2040] # Years to do capacity_output expansion
optimise_frequency = 1 # hours per capacity expansion time slice
r = 0.055 # discount rate
upscale_demand_factor = 1 
rpp = 0.1896
CFE_score = 1
folder_path = 'Results_csvs'

rep_weeks =["2030-01-21 00:00",
            "2030-04-22 00:00",
            "2030-06-17 00:00",
            "2030-11-11 00:00",
            "2035-01-29 00:00",
            "2035-04-16 00:00",
            "2035-07-23 00:00",
            "2035-11-19 00:00",
            "2040-02-13 00:00",
            "2040-04-09 00:00",
            "2040-06-18 00:00",
            "2040-11-19 00:00"]

rep_weeks_weighting = [peak_weighting,offpeak_weighting]*len(investment_years)*2

network = core_func.set_snapshots(network, rep_weeks, optimise_frequency, investment_years)
network = core_func.set_snapshot_weightings(network, rep_weeks_weighting)
network, T = core_func.set_discount_rate(network, r)
network = core_func.input_buses(network)
network = core_func.input_carriers(network)
network = core_func.input_links(network)
network = core_func.input_generators(network)
network = core_func.input_generators_t_p_max_pu(network, investment_years)
network = core_func.input_batteries(network)
network = core_func.input_dummy_extendable_generators(network)
network = core_func.input_dummy_vre_trace_p_max_pu(network, investment_years)
network = core_func.input_dummy_extendable_batteries(network)
candi_matching_loads, loads_less_candi_matching = core_func.input_loads(network, upscale_demand_factor, candi_hourly_match_portion)
# core_func.load_profile(network)
network.iplot(mapbox=True)
network.plot(bus_sizes = 2, color_geomap=True)
plt.show()
#
# # """CONSTRAINTS"""
# m = network.optimize.create_model(multi_investment_periods=True)
#
# network, m = core_func.hydro_constraint(network, m)
#
# initial_capacity = network.generators.p_nom.groupby(network.generators.carrier).sum()
# initial_battery_capacity = network.storage_units.p_nom.sum()
# initial_capacity = pd.concat([initial_capacity,pd.Series([initial_battery_capacity],index = ["Battery"])])
# plt.bar(initial_capacity.index, initial_capacity)
# plt.ylabel("Capacity (MW)")
# plt.title("Initial 2025 Capacity (MW)")
#
# network.optimize.solve_model(method = 2, crossover =0, MIPGap = 0.1, IntFeasTol = 1e-4, FeasabilityTol = 1e-4, FeasRelax =1, solver_name = solver_name)
#
# print("All done! Well done champion\n")
#
# print(network.statistics.curtailment(comps = None))
#
#
# network = core_func.capacity_results(network)
#
#
# core_func.emissions_results(network, scenario)
#
# core_func.generation_profile(network)
#
# core_func.generation_profile_no_batteries(network)
