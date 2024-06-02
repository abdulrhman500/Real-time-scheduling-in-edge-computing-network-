from simulator.networkGenerator import NetworkGenerator
from simulator.Simulation import Simulation
from simulator.TaskGenerator import TaskGenerator
from simulator.algorithms.serverSelection import ServerSelection
network  = NetworkGenerator(num_clusters=10,num_devices=15)
Task_Generator = TaskGenerator(network.devices,network.gateways)
Server_Selection = ServerSelection(network.servers)


simulation_instance = Simulation(100,network,Task_Generator,Server_Selection)
simulation_instance.init_network()
simulation_instance.run()