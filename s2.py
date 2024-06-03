import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List
from abc import ABC, abstractmethod
# Task Class
class Task:
    def __init__(self, size:int, deadline, starting_time=None, arrival_time=None, server=None, channel_gain=None, id=None, num_instruction_to_be_executed=None, default_server=None):
        self.size = size
        self.deadline = deadline
        self.starting_time = starting_time
        self.arrival_time = arrival_time
        self.default_server = default_server
        self.execution_server = server
        self.channel_gain = channel_gain
        self.id = id
        self.flow = 0
        self.num_instruction_to_be_executed = num_instruction_to_be_executed or self.compute_number_of_instruction()

    def compute_number_of_instruction(self):
        return random.randint(1000, 10000) * self.size

    def __str__(self):
        return (f"Task_ID: {self.id}\n"
                f"Size: {self.size}\n"
                f"Deadline: {self.deadline}\n"
                f"Arrival Time: {self.arrival_time}\n"
                f"Starting Time: {self.starting_time}\n"
                f"Execution Server: {self.execution_server}\n"
                f"Channel Gain: {self.channel_gain}\n"
                f"num_Instructions: {self.num_instruction_to_be_executed}\n")

# EndDevice Class
class EndDevice:
    def __init__(self, device_id, gateway, location, comp_capacity, energy_per_cycle, cycle_per_bit, max_waiting_tasks, server):
        self.id = device_id
        self.gateway = gateway
        self.server = server
        self.comp_capacity = comp_capacity
        self.energy_per_cycle = energy_per_cycle
        self.cycle_per_bit = cycle_per_bit
        self.max_waiting_tasks = max_waiting_tasks
        self.location = location
        self.task_queue = deque(maxlen=max_waiting_tasks)

    def get_current_channel_gain(self, channel_id):
        pass

    def compute_transmit_power(self, channel_gain):
        pass

    def compute_energy_per_bit(self):
        return self.cycle_per_bit * self.energy_per_cycle

    def generate_task(self, id, task_generation_prob, task_size_min, task_size_max,max_time):
        from __main__ import Simulation
        if random.random() <= task_generation_prob and self.max_waiting_tasks > len(self.task_queue):
            task_size = random.randint(task_size_min, task_size_max)
            if max_time is None :
                max_time=100
            deadline = random.randint(Simulation.global_clock + 1, Simulation.global_clock +max_time )
            return Task(task_size, Simulation.global_clock + deadline, id=id, arrival_time=Simulation.global_clock, default_server=self.server)
        else:
            return None

    def __str__(self):
        return (f"Device ID: {self.id}\n"
                f"Gateway: {self.gateway}\n"
                f"Computation Capacity: {self.comp_capacity}\n"
                f"Energy Per Cycle: {self.energy_per_cycle}\n"
                f"Cycle Per Bit: {self.cycle_per_bit}\n"
                f"Max Waiting Tasks: {self.max_waiting_tasks}\n"
                f"Location: {self.location}\n"
                f"Task Queue: {list(self.task_queue)}\n")

# Gateway Class
class Gateway:
    def __init__(self, gateway_id, num_channels, channel_bandwidths, location):
        self.id = gateway_id
        self.num_channels = num_channels
        self.available_channels = set(range(num_channels))
        self.channel_bandwidths = channel_bandwidths
        self.location = location

    def __str__(self):
        return (f"Gateway ID: {self.id}\n"
                f"Number of Channels: {self.num_channels}\n"
                f"Available Channels: {self.available_channels}\n"
                f"Channel Bandwidths: {self.channel_bandwidths}\n"
                f"Location: {self.location}\n")

# Server Class
class Server:
    def __init__(self, server_id, gateway, freq, TDP, IPC, uplink_bandwidth, downlink_bandwidth, uplink_cost, downlink_cost, instruction_size, num_cores, energy_unit_price):
        self.id = server_id
        self.gateway = gateway
        self.freq = freq
        self.TDP = TDP
        self.IPC = IPC
        self.IPS = self.IPC * self.freq
        self.EPI = self.TDP / self.IPS
        self.energy_unit_price = energy_unit_price
        self.instruction_size = instruction_size
        self.uplink_bandwidth = uplink_bandwidth
        self.downlink_bandwidth = downlink_bandwidth
        self.uplink_cost = uplink_cost
        self.downlink_cost = downlink_cost
        self.num_cores = num_cores
        self.server_tasks = []
        self.is_space_shared = True

    def processing_cost(self, task: Task):
        return self.EPI * task.num_instruction_to_be_executed

    def processing_time(self, task: Task):
        return task.num_instruction_to_be_executed / self.IPS

    def remaining_comp_capacity(self, start_time, end_time):
        if self.is_space_shared:
            used_cores = [0] * self.num_cores
            for task in self.server_tasks:
                task_end_time = task.starting_time + self.processing_time(task)
                if task.starting_time < end_time and task_end_time > start_time:
                    overlap_start = max(task.starting_time, start_time)
                    overlap_end = min(task_end_time, end_time)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        for i in range(self.num_cores):
                            if used_cores[i] == 0:
                                used_cores[i] = 1
                                break
            remaining_cores = self.num_cores - sum(used_cores)
            return remaining_cores * self.IPS / self.num_cores
        else:
            used_capacity = 0
            for task in self.server_tasks:
                task_end_time = task.starting_time + self.processing_time(task)
                if task.starting_time < end_time and task_end_time > start_time:
                    overlap_start = max(task.starting_time, start_time)
                    overlap_end = min(task_end_time, end_time)
                    overlap_duration = overlap_end - start_time
                    used_capacity += (overlap_duration / (end_time - start_time)) * self.IPS
            remaining_capacity = self.IPS - used_capacity
            return remaining_capacity

    def schedule(self, tasks):
        reverse_tasks = []
        for task in tasks:
            reverse_tasks.append(task)
        reverse_tasks.sort(key=lambda t: t.deadline)
        scheduled_tasks = []
        while reverse_tasks:
            next_task = reverse_tasks.pop(0)
            task_end_time = next_task.deadline
            task_start_time = task_end_time - self.processing_time(next_task)
            next_task.starting_time = task_start_time
            next_task.end_time = task_end_time
            scheduled_tasks.append(next_task)
            self.server_tasks.append(next_task)
            next_task.execution_server = self
        return scheduled_tasks

    def remove_task(self, task_id):
        for task in self.server_tasks:
            if task.id == task_id:
                self.server_tasks.remove(task)
                break

    def __str__(self):
        return (f"\nServer_ID: {self.id}\n"
                f"Gateway: {self.gateway}\n"
                f"Frequency (cycles/sec): {self.freq}\n"
                f"TDP: {self.TDP}\n"
                f"IPC: {self.IPC}\n"
                f"IPS: {self.IPS}\n"
                f"EPI (energy/instruction): {self.EPI}\n"
                f"Energy Unit Price: {self.energy_unit_price}\n"
                f"Instruction Size: {self.instruction_size}\n"
                f"Uplink Bandwidth: {self.uplink_bandwidth}\n"
                f"Downlink Bandwidth: {self.downlink_bandwidth}\n"
                f"Uplink Cost/bit: {self.uplink_cost}\n"
                f"Downlink Cost/bit: {self.downlink_cost}\n"
                f"Number of Cores: {self.num_cores}\n"
                f"Tasks Running: {self.server_tasks}\n"
                f"Is Space Shared: {self.is_space_shared}\n")

# NetworkGenerator Class
class NetworkGenerator:
    def __init__(self, lambda_p=0.1, lambda_c=100, sigma=0.1, area_size=15, num_clusters=None, num_devices=None):
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.sigma = sigma
        self.area_size = area_size
        self.devices = []
        self.gateways = []
        self.servers = []
        self.curr_device_id = 0
        self.curr_gateway_id = 0
        self.num_clusters = num_clusters or np.random.poisson(self.lambda_p * self.area_size ** 2)
        self.num_devices = num_devices or np.random.poisson(self.lambda_c)

    def generate_pcp_network(self):
        """Generates a network using a Poisson Cluster Process (PCP)."""
        cluster_centers = np.random.uniform(0, self.area_size, (self.num_clusters, 2))

        for center in cluster_centers:
            gateway = self.create_gateway(center.tolist(), standard_gateway)
            server = self.create_random_server(gateway)
            self.gateways.append(gateway)
            self.generate_devices_for_gateway(gateway,server, center.tolist())
            # print(f"len of num_devices {self.num_devices}")

    def generate_devices_for_gateway(self, gateway,server, center):
        """Generates devices around a gateway."""
        # num_devices = np.random.poisson(self.lambda_c)
        device_locations = center + self.sigma * np.random.randn(self.num_devices, 2)

        for location in device_locations:
            device = self.create_random_device(location.tolist(), gateway,server)
            # print(f"new device for gatewat: {gateway}\n")
            self.devices.append(device)

    def create_random_server(self,gateway):
        random_index = random.randrange(len(device_prototypes)-1)
        selected_prototype = server_prototypes[random_index]
        new_server = Server(self.curr_device_id,gateway,selected_prototype["freq"],selected_prototype["TDP"],selected_prototype["IPC"],selected_prototype["uplink_bandwidth"],selected_prototype["downlink_bandwidth"],selected_prototype["uplink_cost"],selected_prototype["downlink_cost"],selected_prototype["instruction_size"],selected_prototype["num_cores"],selected_prototype["energy_unit_price"])
        # print(f"server {new_server} \n")
        self.servers.append(new_server)
        self.curr_device_id+=1
        return new_server

    def create_random_device(self, location, gateway,server):
        """Creates a new EndDevice object with random properties based on prototypes."""
        random_index = random.randrange(len(device_prototypes)-1)
        selected_prototype = device_prototypes[random_index]
        new_device = EndDevice(self.curr_device_id, gateway, location, selected_prototype["comp_capacity"],selected_prototype["energy_per_cycle"],selected_prototype["cycle_per_bit"],selected_prototype["max_waiting_tasks"],server) 
        self.curr_device_id+=1
        return new_device

    def create_gateway(self, location, prototype):
        """Creates a new Gateway object based on the prototype."""
        new_gateway = Gateway(self.curr_gateway_id, prototype["num_channels"], prototype["channel_bandwidths"], location)  # Unique gateway ID
        self.curr_gateway_id +=1
        return new_gateway

    def visualize_network(self):
        """Plots the generated network."""
        # print("vv")
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

device_prototypes = [
    {"type": "smartphone", "version": 0, "name": "High-End Smartphone", "comp_capacity": 3.0e9, "energy_per_cycle": 1e-9, "cycle_per_bit": 10, "max_waiting_tasks": 10},
    {"type": "sensor", "version": 0, "name": "Low-Power Sensor Node", "comp_capacity": 0.5e9, "energy_per_cycle": 0.8e-9, "cycle_per_bit": 20, "max_waiting_tasks": 2},
    {"type": "sbc", "version": 0, "name": "Single-Board Computer", "comp_capacity": 1.4e9, "energy_per_cycle": 1.1e-9, "cycle_per_bit": 14, "max_waiting_tasks": 5},
    {"type": "wearable", "version": 0, "name": "Wearable Device", "comp_capacity": 1.2e9, "energy_per_cycle": 1.3e-9, "cycle_per_bit": 16, "max_waiting_tasks": 3}
]
#C = B * log₂(1 + SNR)
# C is the channel capacity (maximum data rate) in bits per second (bps).
# B is the channel bandwidth in Hertz (Hz).
# SNR = Signal-to-Noise Ratio
# The bandwidth of a channel is often directly referred to as its channel width.
standard_gateway = {
    
        "type": "mobile_base_station",
        "version": 1,
        "name": "4G Mobile Base Station",
        "num_channels": 10,
        "channel_bandwidths": [10, 10, 30, 20, 20, 20, 30, 10, 30, 30]  # MHz
}


gateway_prototypes = [
    {
        "type": "residential",
        "version": 1,
        "name": "Basic Home Gateway",
        "num_channels": 3,
        "channel_bandwidths": [5, 10, 15]  # MHz
    },
    {
        "type": "enterprise",
        "version": 2,
        "name": "High-Capacity Enterprise Gateway",
        "num_channels": 8,
        "channel_bandwidths": [20, 20, 20, 20, 30, 30, 40, 50]  # MHz
    },
    {
        "type": "industrial",
        "version": 1,
        "name": "Rugged Industrial Gateway",
        "num_channels": 5,
        "channel_bandwidths": [10, 10, 15, 20, 25]  # MHz
    },
    {
        "type": "mobile",
        "version": 1,
        "name": "Mobile Hotspot",
        "num_channels": 2,
        "channel_bandwidths": [5, 10]  # MHz
    },
    {
        "type": "mobile_base_station",
        "version": 1,
        "name": "4G Mobile Base Station",
        "num_channels": 10,  # More channels for wider coverage
        "channel_bandwidths": [10, 10, 30, 20, 20, 20, 30, 10, 30, 30]  # MHz
    }
]



# server_prototypes = [
    
#     {
#         "type": "midserver",
#         "version": 1,
#         "name": "Standard Edge Server",
#         "freq": 15e9,  # 15 GHz
#         "energy_per_cycle": 1.5e-9,
#         "uplink_bandwidth":15, #bits Per sec
#         "downlink_bandwidth":15 #bits Per sec
        
        
#     },
#     {
#         "type": "highserver",
#         "version": 1,
#         "name": "High-Performance Edge Server",
#         "freq": 30e9,  # 30 GHz
#         "energy_per_cycle": 1e-9,  # More efficient energy consumption
#         "uplink_bandwidth":15, #bits Per sec
#         "downlink_bandwidth":15 #bits Per sec
        
#     }
# ]

server_prototypes = [
    {
        "server_id": 1,
        "gateway": "Gateway High",
        "freq": 3.2e9,  # 3.2 GHz
        "TDP": 125,  # 125 Watts
        "IPC": 1.5,  # 1.5 Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 32,  # 32 cores
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
    {
        "server_id": 2,
        "gateway": "Gateway Standard",
        "freq": 2.5e9,  # 2.5 GHz
        "TDP": 85,  # 85 Watts
        "IPC": 1.2,  # 1.2 Instructions per cycle
        "uplink_bandwidth": 1e9,  # 1 Gbps
        "downlink_bandwidth": 1e9,  # 1 Gbps
        "uplink_cost": 0.01,  # $0.01 per bit
        "downlink_cost": 0.01,  # $0.01 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 16,  # 16 cores
        "energy_unit_price": 0.10  # $0.10 per unit of energy
    },
    {
        "server_id": 3,
        "gateway": "Gateway Real-Time",
        "freq": 2.2e9,  # 2.2 GHz
        "TDP": 95,  # 95 Watts
        "IPC": 1.0,  # 1.0 Instructions per cycle
        "uplink_bandwidth": 800e6,  # 800 Mbps
        "downlink_bandwidth": 800e6,  # 800 Mbps
        "uplink_cost": 0.015,  # $0.015 per bit
        "downlink_cost": 0.015,  # $0.015 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 8,  # 8 cores
        "energy_unit_price": 0.12  # $0.12 per unit of energy
    }
]



class TaskGeneratorInterface(ABC):
    @abstractmethod
    def __init__(self,devices:List[EndDevice],gateways:List[Gateway]):
       pass
        

    @abstractmethod
    def generate_tasks(self,max_time, task_generation_prob=0.3, task_size_min=1, task_size_max=1000):
       pass

class ProposedTaskGenerator(TaskGeneratorInterface):
    
    def __init__(self, devices: List[EndDevice], gateways: List[Gateway]):
        self.next_available_id = 0
        self.devices = devices
        self.gateways = gateways  

    def generate_tasks(self, max_time, task_generation_prob=1, task_size_min=1, task_size_max=1000):
        count_generated_task = 0
        generated_tasks = []

        for device in self.devices:
            task = device.generate_task(self.next_available_id, task_generation_prob, 
                                        task_size_min, task_size_max, max_time)
            
            if task:
                generated_tasks.append(task)
                self.next_available_id += 1
                device.task_queue.append(task)
                count_generated_task += 1
        
        return count_generated_task, generated_tasks




    
class ServerSelectionInterface(ABC):
    @abstractmethod
    def __init__(self, servers: List[Server]):
        pass

    @abstractmethod
    def select_servers(self, tasks: List[Task]):
        pass

    @abstractmethod
    def get_valid_servers(self, task: Task) -> List[Server]:
        pass

    @abstractmethod
    def is_server_valid(self, task: Task, server: Server) -> bool:
        pass

    @abstractmethod
    def get_min_cost_server(self, task: Task, valid_servers: List[Server]) -> Server:
        pass

    
class ProposedServerSelection(ServerSelectionInterface):
    def __init__(self, servers: List[Server]):
        """
        Initialize the ServerSelection class with a list of available servers.
        """
        self.servers = servers

    def select_servers(self, tasks: List[Task]):
        """
        Select the most appropriate server for each task from the given list of tasks.
        """
        # print(f"in seelct servwers {tasks}")
        for task in tasks:
            valid_servers = self.get_valid_servers(task)
            if valid_servers:
                selected_server = self.get_min_cost_server(task, valid_servers)
                task.execution_server = selected_server

            else:
                task.execution_server = task.default_server

    def get_valid_servers(self, task: Task) -> List[Server]:
        """
        Get a list of valid servers that can execute the given task within the required time frame.
        """
        valid_servers = []
        for server in self.servers:
            if self.is_server_valid(task, server):
                valid_servers.append(server)
        return valid_servers

    def is_server_valid(self, task: Task, server: Server) -> bool:
        """
        Check if the given server can execute the task within its deadline.
        """
        # print(f"size in si_server_valid {task}" )
        transfer_time = task.size / min(server.uplink_bandwidth, server.downlink_bandwidth)
        total_time = transfer_time + server.processing_time(task)
        return total_time <= (task.deadline - task.arrival_time)

    def get_min_cost_server(self, task: Task, valid_servers: List[Server]) -> Server:
        """
        Get the server with the minimum total cost for executing the given task.
        """
        min_cost = float('inf')
        selected_server = None
        for server in valid_servers:
            cost = self.calculate_total_cost(task, server)
            if cost < min_cost:
                min_cost = cost
                selected_server = server
        return selected_server

    def calculate_total_cost(self, task: Task, server: Server) -> float:
        """
        Calculate the total cost of executing the task on the given server, including data transfer and processing costs.
        """
        transfer_cost = 0
        if server.id != task.default_server.id:
            transfer_cost = task.size * (task.default_server.uplink_cost + server.downlink_cost)
        processing_cost = server.processing_cost(task)
        return transfer_cost + processing_cost


class Scheduler:
    
    def __init__(self,simulation_instance):
        self.gateways = simulation_instance.gateways
        self.servers = simulation_instance.severs
        self.simulation_instance= simulation_instance

    def calculate_normalized_flow(self,task):
        #assume it is always positive
        return task.size / min(task.default_server.uplink_bandwidth,task.execution_server.downlink_bandwidth)
        
    def normalize_flows(self,tasks):
        for task in tasks:
            if task.execution_server != task.default_server:
                task.flow = self.calculate_normalized_flow(task)


    def calculate_intensity(self,flows, interval):
        start_time, end_time = interval
        sum = 0
        for flow in flows:
            sum += flow.flow

        return sum/(end_time-start_time)

    def control_admission(self, flows: List[Task]):
        while True:
            most_critical_interval, most_critical_server, most_critical_server_flows, most_critical_intensity = self.find_most_critical_interval(flows)

            if most_critical_interval is None:
                break

            if most_critical_intensity <= 1:
                break

            self.remove_max_flow_from_interval(most_critical_server_flows)


             
    def remove_max_flow_from_interval(self, flows: List[Task]):
        max_flow = max(flows, key=lambda flow: flow.normalized_size/(flow.deadline-flow.arrival_time), default=None)
        if max_flow:
            flows.remove(max_flow)
            max_flow.execution_server.remove_task(max_flow.id)


    def select_server_for_tasks(self,tasks):
        ProposedServerSelection(self.servers).select_servers(tasks)
        
      

    def schedule_for_computation(self,tasks:List[Task]):
        for task in tasks:
            task.execution_server.schedule([task])


    def apply_ASCO(self,tasks):
        self.select_server_for_tasks(tasks)
        self.schedule_for_computation(tasks)
        self.normalize_flows(tasks)
        self.control_admission(tasks)



    def apply(self,tasks:List[Task],type="ASCO"):
       
        if type == "ASCO":
            self.apply_ASCO(tasks)




    def find_most_critical_interval(self, flows):
        if not flows:
            return None, None, None, None

        # Calculate the max time counter based on the latest deadline
      
        most_critical_intensity = 0
        most_critical_interval = None
        most_critical_server = None

        # Map each server to its respective flows
        server_flow = {server.id: [] for server in self.servers}
        for flow in flows:
            if flow.default_server.id != flow.execution_server:
                server_flow[flow.default_server.id].append(flow)

        # Iterate through each server
        for server in self.simulation_instance.severs:
            curr_flows = server_flow.get(server.id)
            if not curr_flows:
                continue

            # Calculate the max time counter based on the flows assigned to the current server
           
            start_time = self.simulation_instance.global_clock
            end_time =  max(flow.deadline for flow in curr_flows)  

            # Iterate through possible intervals
            for i in range(end_time - start_time + 1):
                for j in range(end_time - start_time + 1):
                    if start_time + i >= end_time - j:
                        continue
                    
                    curr_s = start_time + i
                    curr_e = end_time - j
                    curr_intensity = self.calculate_intensity(curr_flows, (curr_s, curr_e))

                    if curr_intensity > most_critical_intensity:
                        most_critical_intensity = curr_intensity
                        most_critical_interval = (curr_s, curr_e)
                        most_critical_server = server

        if most_critical_interval is None:
            return None, None, None, None

        return most_critical_interval, most_critical_server, server_flow.get(most_critical_server.id, []), most_critical_intensity




class Simulation:
    global_clock = 0  

    def __init__(self,  total_tasks: int,network_generator:NetworkGenerator,task_generator:TaskGeneratorInterface,server_Selector:ServerSelectionInterface):

        self.network_generator = network_generator
        self.task_generator = task_generator
        self.total_tasks = total_tasks
        self.current_generated_tasks = 0
        self.current_completed_tasks = 0
        self.offloaded_tasks=[]
        self.server_Selector = server_Selector
        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.severs = self.network_generator.servers
        
   
   
    def init_network(self):
        self.network_generator.generate_pcp_network()
        # self.network_generator.visualize_network()

    
        

    def run(self,num,mode=2):

        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.severs = self.network_generator.servers

        if mode ==1:
            self.run_time(num)
        else:
            self.run_tasks(num)

        print(f"Gateway: { len(self.gateways)}\nDevices : {len(self.devices)}\n servers : {len(self.severs)}")
        
    def run_tasks(self,num_tasks):
        done_tasks =0 
        total_generated_tasks = 0 
        while(total_generated_tasks<=num_tasks):
            if(total_generated_tasks<num_tasks):
                count_generated_task, generated_tasks =  self.task_generator.generate_tasks(None,task_generation_prob=1, task_size_min=10, task_size_max=1000)
                total_generated_tasks+=len(generated_tasks)
                print(total_generated_tasks)
                Scheduler(simulation_instance=self).apply(generated_tasks,"ASCO")
                self.global_clock += 1


    def run_time(self,time):
        for i in range(time):
            count_generated_task, generated_tasks = self.task_generator.generate_tasks(time,task_generation_prob=0.1, task_size_min=10, task_size_max=1000)
            Scheduler(simulation_instance=self).apply(generated_tasks,"ASCO")
            self.global_clock += 1
            print(i)

    def offload_tasks(self,generated_tasks):
            # print(f"genratd tasks at this time  {len(generated_tasks)}")
            return generated_tasks
           




def main():
    num_tasks = 10
    time = 100

        
    network  = NetworkGenerator(num_clusters=10,num_devices=15)
    network.generate_pcp_network()
    Task_Generator = ProposedTaskGenerator(network.devices,network.gateways)
    Server_Selection = ProposedServerSelection(network.servers)


    proposed_simulation = Simulation(100,network,Task_Generator,Server_Selection)
    
    # baseline_simulation = Simulation(100,network,Task_Generator,Server_Selection)

    #outout should be Task addmison rate , avergae cost per task
    
    proposed_simulation.run(num_tasks,mode=1)
    # baseline_simulation.run(num_tasks)


main()
