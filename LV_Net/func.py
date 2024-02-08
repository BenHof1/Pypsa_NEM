import pypsa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx


def add_active(name, net):
    # Places active nodes near their network connection
    x = net.buses[net.buses.index == "My bus {}".format(name)].x.tolist()[0]
    y = net.buses[net.buses.index == "My bus {}".format(name)].y.tolist()[0]

    # Add active node to assign load to. Index based on order of connection to network
    connected = (len(net.lines[net.lines.bus0 == "My bus {}".format(name)].index.to_list()) +
                 len(net.lines[net.lines.bus1 == "My bus {}".format(name)].index.to_list()))
    new_node = name + '_' + str(connected)
    net.add("Bus", "My bus {}".format(new_node), x=x, y=y)
    # Link active node to network
    line_count = len(net.lines.index.to_list())
    net.add("Line", "My line {}".format(line_count),
            bus0="My bus {}".format(new_node),
            bus1="My bus {}".format(name),
            x=0.00001, r=0.00001,
            length=0.001)
    return new_node


def add_bus(net, node_val, name):
    # Return if Node already exists
    if "My bus {}".format(name) in net.buses.index.to_list():
        return net
    else:
        net.add("Bus", "My bus {}".format(name))

    # Add features if available
    if 'v_base' in list(node_val.keys()):
        net.buses.loc[["My bus {}".format(name)], 'v_nom'] = node_val['v_base']

    if 'xy' in list(node_val.keys()):
        if node_val['xy'] is not None:
            net.buses.loc[["My bus {}".format(name)], 'x'] = node_val['xy'][0]
            net.buses.loc[["My bus {}".format(name)], 'y'] = node_val['xy'][1]
    return net


def add_load(net, bus_val):
    # v_nom= bus_val['v_nom'], could be added if relevant

    # Ensure network connection exists
    name = bus_val['cons'][0]['node']
    net = add_bus(net, bus_val, name)

    # Add connection
    new_node = add_active(name, net)

    # Check this is pointing correctly!
    # Add node features if available
    net.buses.loc[["My bus {}".format(new_node)], 'v_mag_pu_set'] = 1
    if 'v_max_pu' in list(bus_val.keys()):
        net.buses.loc[["My bus {}".format(new_node)], 'v_mag_pu_max'] = bus_val['v_max_pu']
    if 'v_min_pu' in list(bus_val.keys()):
        net.buses.loc[["My bus {}".format(new_node)], 'v_mag_pu_min'] = bus_val['v_min_pu']
    load_count = len(net.loads.index.to_list())

    # Define load character. Can be extended
    net.add("Load", "My load {}".format(load_count),
            bus="My bus {}".format(new_node)
            )
    return net


def add_gen(net, bus_val):
    # Ensure network connection exists
    name = bus_val['cons'][0]['node']
    net = add_bus(net, bus_val, name)

    # Add active node to assign load to. Index based on order of connection to network
    new_node = add_active(name, net)

    # Define Control type: default gen is PQ
    if 'is_slack' in list(bus_val.keys()):
        if bus_val['is_slack']:
            control = "Slack"
        else:
            control = 'PQ'
    else:
        control = 'PQ'

    gen_count = len(net.generators.index.to_list())
    net.add("Generator", "My gen {}".format(gen_count),
            bus="My bus {}".format(new_node),
            control=control
            )
    return net


def add_line(net, line):
    resistivity = line['z'][0]
    r_var = line['z0'][0]
    impedance = line['z'][1]
    x_var = line['z0'][1]

    line_count = len(net.lines.index.to_list())
    net.add("Line", "My line {}".format(line_count),
            bus0="My bus {}".format(line['cons'][0]['node']),
            bus1="My bus {}".format(line['cons'][1]['node']),
            x=impedance, r=resistivity,
            x_pu=x_var,
            r_pu=r_var,
            s_nom_max=line['i_max'],
            s_nom_min=-line['i_max'],
            length=line['length'])
    return net


def add_transformer(net, x):
    resistivity = x['z'][0][0]
    r_var = x['z'][0][1]
    impedance = x['z'][1][0]
    x_var = x['z'][1][1]

    net = add_bus(net, x, x['cons'][0]['node'])
    net = add_bus(net, x, x['cons'][1]['node'])

    net.add('Transformer', 'My Transformer {}'.format(1),
            bus0="My bus {}".format(x['cons'][0]['node']),
            bus1="My bus {}".format(x['cons'][1]['node']),
            x=impedance, r=resistivity,
            x_pu=x_var,
            r_pu=r_var,
            s_nom_max=x['s_max'],
            tap_ratio=x['nom_turns_ratio']
            )
    return net


def to_pypsa(data):
    net = pypsa.Network()

    for i in sorted(data['components'].keys()):
        # If Transformer
        if 'Transformer' in data['components'][i].keys():
            net = add_transformer(net, data['components'][i]['Transformer'])
        # Draw Nodes
        if 'Node' in data['components'][i].keys():
            net = add_bus(net, data['components'][i], name=str(i))

        # Manage Loads
        if 'Load' in data['components'][i].keys():
            net = add_load(net, data['components'][i]['Load'])
        # Manage Gen
        if 'Infeeder' in data['components'][i].keys():
            net = add_gen(net, data['components'][i]['Infeeder'])
            # Alternatively use fd_michael node for source.

        # Setup Lines
        if 'Line' in data['components'][i].keys():
            net = add_line(net, data['components'][i]['Line'])

    return net


def plotly_net(net, title=None):
    gen = net.generators.bus.to_list()
    load = net.loads.bus.to_list()
    passive = net.buses.index.to_list()
    ls = []
    for i in passive:
        if i in gen:
            ls.append((i, 1))
        elif i in load:
            ls.append((i, -1))
        else:
            ls.append((i, -0.5))
    ls2 = [i[1] for i in ls]

    fig = go.Figure(net.iplot(bus_text=net.buses.index.to_list(), title=title,
                              bus_colors=ls2, jitter=0,
                              layouter=nx.kamada_kawai_layout)
                    )
    # kamada_kawai_layout - best
    # spring - ok
    # Spectral - low potential
    return fig


def plt_net(net, title=None, bus_cmap='RdYlGn_r', jitter=0.001, bus_size=0.001, figsize=(6.4, 4.8)):
    gen = net.generators.bus.to_list()
    load = net.loads.bus.to_list()
    passive = net.buses.index.to_list()
    ls = []
    for i in passive:
        if i in gen:
            ls.append((i, -1))
        elif i in load:
            ls.append((i, 1))
        else:
            ls.append((i, 0.1))
    bs_color = [i[1] for i in ls]
    fig = plt.figure(figsize=figsize)
    net.plot(title=title, bus_colors=bs_color,
             bus_cmap=bus_cmap, bus_sizes=bus_size, jitter=jitter,
             margin=0.2, geomap=False,
             layouter=nx.kamada_kawai_layout)
    plt.tight_layout()
    plt.text(0.1, -0.5, 'Generators', ha='center', color='limegreen', weight='bold')
    plt.text(0.5, -0.5, 'Loads', ha='center', color='lightcoral', weight='bold')
    plt.text(0.9, -0.5, 'Passive', ha='center', color='darkgoldenrod', weight='bold')
    # plt.show()
    return fig.get_figure()
