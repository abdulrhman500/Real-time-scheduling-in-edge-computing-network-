import numpy as np
import matplotlib.pyplot as plt
import random

from constants import device_prototypes,standard_gateway
from components.EndDevice import EndDevice 
from components.Gateway import Gateway

class Setup:
    def __init__(self, lambda_p=0.1, lambda_c=100, sigma=0.1, area_size=15, d0=1.0, path_loss_exponent=2.7, mu=0, sigma_fading=1.0):
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.sigma = sigma
        self.area_size = area_size
        self.d0 = d0
        self.path_loss_exponent = path_loss_exponent
        self.mu = mu
        self.sigma_fading = sigma_fading
        self.curr_device_id = 0
        self.curr_gateway_id = 0

        # Data structures to store generated points and device objects
        self.parent_points = None
        self.child_points = None
        self.network_topology = [{"gateway":Gateway(),"dndDevices":1}]

    def set_num_parents_PCP(self):
        return np.random.poisson(self.lambda_p * self.area_size * self.area_size)
    
    def set_num_parents_standard(self):
        #TODO
        return np.random.poisson(self.lambda_p * self.area_size * self.area_size)
    
    def set_num_children_PCP(self):
        return  np.random.poisson(self.lambda_c)
    
    def set_num_children_standard(self):
        #TODO
        return  np.random.poisson(self.lambda_c)
    
    def generate_points(self):
        """Generates parent and child points using a Poisson cluster process."""
        num_parents = self.set_num_parents_PCP()
        self.parent_points = np.random.uniform(0, self.area_size, (num_parents, 2))

        
        for parent_location in self.parent_points:
            gateway = self.create_standard_gateway(parent_location)
            small_network = {"gateway":gateway}
          
            
   
    def generate_device_for_gateway(self,gateway,parent_location):
        devices = []
        child_points_list = []
        num_children = self.set_num_children_PCP()
        children_location = parent_location + self.sigma * np.random.randn(num_children, 2)

        for child in children_location:
            device = self.create_random_device(child,gateway)
            devices.append(device)

        child_points_list.append(children_location)
        self.child_points = np.vstack(child_points_list)
        return devices 
            
    def visualize_points(self):
        """Plots the generated parent and child points."""
        plt.scatter(self.child_points[:, 0], self.child_points[:, 1], alpha=0.6)
        plt.scatter(self.parent_points[:, 0], self.parent_points[:, 1], c='red')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Poisson Cluster Process')
        plt.show()

    def create_random_device(self, location, gateway):
        """Creates a new EndDevice object with random properties based on prototypes."""
        if not device_prototypes:
            raise ValueError("device_prototypes list is empty.")
        random_index = random.randrange(len(device_prototypes))
        selected_prototype = device_prototypes[random_index]
        new_device = EndDevice(device_id=self.curr_device_id, gateway=gateway, location=location, **selected_prototype)
        self.curr_device_id += 1
        self.devices.append(new_device)  # Store the device object for future use
        return new_device
    
    
    def create_standard_gateway(self,location):
        new_gateway=  Gateway(self.curr_gateway_id,standard_gateway["num_channels"],standard_gateway["channel_bandwidths"], location)
        self.curr_gateway_id+=1
        return new_gateway
  
  
    #
#    # def calculate_received_power(self):
    #     """Calculates received power at each child point using path loss and Rayleigh fading."""
    #     distances = np.sqrt(np.sum(self.child_points**2, axis=1))
    #     path_losses = 10 * self.path_loss_exponent * np.log10(distances / self.d0)
    #     small_scale_fading = self.sigma_fading * np.random.rayleigh(size=distances.shape)
    #     received_power = -path_losses + 10 * np.log10(small_scale_fading)
    #     return received_power

    # def visualize_received_power(self):
    #     """Visualizes the received power at child points."""
    #     received_power = self.calculate_received_power()
    #     plt.scatter(self.child_points[:, 0], self.child_points[:, 1], c=received_power, cmap='viridis')
    #     plt.colorbar(label='Received Power (dB)')
    #     plt.xlabel('X Coordinate')
    #     plt.ylabel('Y Coordinate')
    #     plt.title('IoT User Distribution with Received Power')
    #     plt.show()
