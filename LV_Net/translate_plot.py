import json
import func

json_Path = 'C:/Users/u6352049/Downloads/'
file = 'S_LV_network.json'
f = open(json_Path + file)
data = json.load(f)

net = func.to_pypsa(data)
plot_a = func.plotly_net(net, title=file.split('.')[0].replace('_', ' '))
# plot_a.show()
plot_b = func.plt_net(net)
plot_b.show()
plot_a.write_image("images/" + file.split('.')[0].replace('_', ' ') + ".webp")
plot_a.write_image("images/" + file.split('.')[0].replace('_', ' ') + ".png")
gen = sorted(net.generators.bus.to_list())
load = sorted(net.loads.bus.to_list())
passive = [i for i in sorted(net.buses.index.to_list()) if i not in gen + load]
print('gen', gen)
print('load', load)
print('passive', passive)

# Once complete you can use more appropriate methods of storing
# Find at https://pypsa.readthedocs.io/en/latest/import_export.html

# net.export_to_hdf5(json_Path+file.split('.')[0]+'.hdf5')
# Then use this to import network
# net = pypsa.Network()
# net.import_from_hdf5().
