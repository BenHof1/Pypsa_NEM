import pypsa as psa
import networkx as nx
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import json
sys.path.insert(1,'../LV_Net/')
import func as translate
#%%
json_Path = 'C:/Users/u6352049/Downloads/'
file = 'D_LV_network.json'
f = open(json_Path + file)
data = json.load(f)

net = translate.to_pypsa(data)
#%%
demand = pd.read_csv('E:/Shubhankar/2022-01-26.csv',usecols=['systemid','export_p','export_q','timestamp'])
demand.export_p =demand.export_p/1000
demand.export_q =demand.export_q/1000

#%%
dx = demand.groupby('systemid', as_index=False).count()
da = dx[dx.timestamp==288]
demand_2 = demand[demand.systemid.isin(da.systemid.unique())]
#%%

net.snapshots = sorted(demand_2.timestamp.unique())

#%%
count = 0
c_max = len(demand_2.systemid.unique())
load_samples = demand_2.systemid.unique()
for i in net.loads.index.to_list():
    ex_p = np.array(demand_2[demand_2.systemid==load_samples[count]].export_p.to_list(), dtype=np.float16)
    ex_q = np.array(demand_2[demand_2.systemid==load_samples[count]].export_q.to_list(), dtype=np.float16)
    count+=1
    if count>=c_max:
        count =0
    net.loads_t.p_set[i] = ex_p
    net.loads_t.q_set[i] = ex_q
#%%
net.generators.loc[0, 'bus'] = "My bus 57_0"
net.generators.loc[0,'p_nom'] = 0.1
net.generators.loc[0,'p_nom_min'] = 0.0
net.generators.loc[0,'p_nom_max'] = 10
net.generators.loc[0,'p_max_pu'] = 10
net.generators.loc[0,'p_min_pu'] = 0
# net.generators_t.p_max_pu['My gen 0'] = np.array(np.repeat(100,288), dtype=np.float16)
# net.generators_t.p_min_pu['My gen 0'] = np.array(np.repeat(0,288), dtype=np.float16)

net.generators.loc[0,'p_set'] = 0
net.generators.loc[0,'q_set'] = 0
net.generators.loc[0,'q_nom'] = 1.0
net.generators.loc[0,'marginal_cost'] = 1.0
net.generators.loc[0,'p_nom_extendable']=True
net.generators.loc[0,'control'] = 'Slack'


net.lpf(snapshots=net.snapshots[0])
now = net.snapshots[0]
angle_diff = pd.Series(net.buses_t.v_ang.loc[now,net.lines.bus0].values -
                       net.buses_t.v_ang.loc[now,net.lines.bus1].values,
                       index=net.lines.index)

print((angle_diff*180/np.pi).describe())
# psa.Network.consistency_check(net)


translate.plotly_net(net, title=None);

net.lpf()
net.pf(use_seed=True)


# net.lopf()
# net.lpf()
# net.pf()
