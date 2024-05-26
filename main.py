from simulator.networkGenerator import NetworkGenerator
from simulator.Simulation import Simulation
network  = NetworkGenerator()

# network.generate_pcp_network()
# network.visualize_network()

simulation_instance = Simulation(100,network.gateways,network.devices,network)

# simulation_instance.run()