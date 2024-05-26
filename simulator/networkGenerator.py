import numpy as np
import matplotlib.pyplot as plt
import random
from components.EndDevice import EndDevice
from components.Gateway import Gateway
from simulator.constants import device_prototypes, standard_gateway

class NetworkGenerator:
    def __init__(self, lambda_p=0.1, lambda_c=100, sigma=0.1, area_size=15):
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.sigma = sigma
        self.area_size = area_size
        self.devices = []
        self.gateways = []
        self.curr_device_id = 0
        self.curr_gateway_id = 0

    # def generate_tasks(self, task_generation_prob=0.3, task_size_min=1, task_size_max=1000):
    #     count_generated_task = 0
    #     for device in self.devices:
    #         task = device.generate_task(task_generation_prob, task_size_min, task_size_max)
    #         if task:
    #             device.task_queue.append(task)
    #             count_generated_task+=1
    #     return count_generated_task

    def generate_pcp_network(self):
        """Generates a network using a Poisson Cluster Process (PCP)."""
        num_clusters = np.random.poisson(self.lambda_p * self.area_size**2)
        cluster_centers = np.random.uniform(0, self.area_size, (num_clusters, 2))

        for center in cluster_centers:
            gateway = self.create_gateway(center.tolist(), standard_gateway)
            self.gateways.append(gateway)
            self.generate_devices_for_gateway(gateway, center.tolist())

    def generate_devices_for_gateway(self, gateway, center):
        """Generates devices around a gateway."""
        num_devices = np.random.poisson(self.lambda_c)
        device_locations = center + self.sigma * np.random.randn(num_devices, 2)

        for location in device_locations:
            device = self.create_random_device(location.tolist(), gateway)
            self.devices.append(device)

    def create_random_device(self, location, gateway):
        """Creates a new EndDevice object with random properties based on prototypes."""
        random_index = random.randrange(len(device_prototypes))
        selected_prototype = device_prototypes[random_index]
        new_device = EndDevice(self.curr_device_id, gateway, location, selected_prototype["comp_capacity"],selected_prototype["energy_per_cycle"],selected_prototype["cycle_per_bit"],selected_prototype["max_waiting_tasks"]) 
        self.curr_device_id+=1
        return new_device

    def create_gateway(self, location, prototype):
        """Creates a new Gateway object based on the prototype."""
        new_gateway = Gateway(self.curr_gateway_id, prototype["num_channels"], prototype["channel_bandwidths"], location)  # Unique gateway ID
        self.curr_gateway_id +=1
        return new_gateway

    def visualize_network(self):
        """Plots the generated network."""
        print("vv")
        print(self.devices)
        print(self.gateways)
        for device in self.devices:
            plt.scatter(device.location[0], device.location[1], alpha=0.6, c="blue", label="IoT Devices" if device == self.devices[0] else "")  
        for gateway in self.gateways:
            plt.scatter(gateway.location[0], gateway.location[1], c='red', marker='x', label="Gateways" if gateway == self.gateways[0] else "")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Network Topology')
        plt.legend()
        plt.show()
