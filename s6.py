import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List
from abc import ABC, abstractmethod
import math
import json
import os
import numpy as np
from collections import defaultdict
import heapq
from typing import List
import time

max_time_= None
max_duration = None
class Task:
    def __init__(self, size:int, deadline, starting_time=None, arrival_time=None, server=None, channel_gain=None, id=None, num_instruction_to_be_executed=None, default_server=None):
        self.size = size
        self.deadline = deadline
        self.starting_time = starting_time
        self.arrival_time = arrival_time
        self.default_server = default_server
        self.execution_server = server or default_server
        self.channel_gain = channel_gain
        self.id = id
        self.flow = 0
        self.num_instruction_to_be_executed = num_instruction_to_be_executed or self.compute_number_of_instruction()
        self.cost = 0
        self.end_time = deadline

    def compute_number_of_instruction(self):
        return random.randint(10, 10000) * self.size
    
    def __lt__(self, other):
        # Compare based on deadline
        return self.deadline < other.deadline

    def __eq__(self, other):
        return self.id == other.id
    def __hash__(self):
        return hash(self.id)
    
    def __str__(self):
        return (f"Task_ID: {self.id}\n"
                f"Size: {self.size}\n"
                f"Deadline: {self.deadline}\n"
                f"Arrival Time: {self.arrival_time}\n"
                f"Starting Time: {self.starting_time}\n"
                f"------------------- \n Execution Server: [{self.execution_server}]\n------------------------\n"
                f"Channel Gain: {self.channel_gain}\n"
                f"Cost: {self.cost}"
                f"num_Instructions: {self.num_instruction_to_be_executed}\n")
    def important_details(self):

        return (f"Task_ID: {self.id}\n"
                f"Size: {self.size}\n"
                f"Deadline: {self.deadline}\n"
                f"Arrival Time: {self.arrival_time}\n"
                f"Starting Time: {self.starting_time}\n"
                f"Cost: {self.cost}"
                f"num_Instructions: {self.num_instruction_to_be_executed}\n")
    def to_dict(self):
        return {
            "id": self.id,
            "task_size": self.size,
            "arrival_time": self.arrival_time,
            "starting_time":self.starting_time,
            "deadline": self.deadline,
            "cost":self.cost,
          
        }
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

    def generate_task(self, id, task_generation_prob, task_size_min, task_size_max, max_time):
        # from __main__ import Simulation
        global max_time_
        global max_duration
        task_size = random.randint(task_size_min, task_size_max)
        if max_time is None:
            max_time = max_time_ #2000
        arrival_time = int(random.randint( 0,max_time_))
        deadline = int(arrival_time+ random.randint( 0, max_duration))+1

        return Task(task_size,  deadline, id=id, arrival_time=arrival_time, default_server=self.server)
    
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
        self.preempted_tasks=[]
        self.is_space_shared = True

    def processing_cost(self, task: Task):
        return self.EPI * task.num_instruction_to_be_executed

    def processing_time(self, task: Task):
        return math.ceil(task.num_instruction_to_be_executed / self.IPS)

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
            return remaining_cores * self.IPS 
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
    def to_dict(self):
        return {
            "id": self.id,
            "IPC": self.IPC ,
            "EPI": self.EPI

        }
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
        random_index = random.randrange(len(server_prototypes)-1)
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
        # print(self.devices)
        # print(self.gateways)
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
server_prototypes = [
     {
         # https://www.tomshardware.com/news/amd-launches-threadripper-3990x-and-ryzen-4000-renoir-apus
        "server_id": 2,
        "gateway": "AMD Ryzen Threadripper 3990X (64 core) ",
        "freq": 4.3e9,  # GHz
        "TDP":  280,  #  Watts
        "IPC":8.46,  # 1.5 Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 64,  # 32 cores
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
    {
        #https://ark.intel.com/content/www/us/en/ark/products/212275/intel-core-i5-11600k-processor-12m-cache-up-to-4-90-ghz.html
        "server_id":3,
        "gateway": "Rocket Lake Intel Core i5-11600k",
        "freq": 3.90e9,  # GHz
        "TDP": 125,  # Watts
        "IPC":11.73,  #  Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 6,  
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },


    {
        #https://ark.intel.com/content/www/us/en/ark/products/212275/intel-core-i5-11600k-processor-12m-cache-up-to-4-90-ghz.html
        "server_id":4,
        "gateway": "Intel® Core™ i7-6950X",
        "freq": 3.50e9,  #  GHz
        "TDP": 140,  #  Watts
        "IPC":9.16,  #  Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 10,  
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
    
    {
        #https://ark.intel.com/content/www/us/en/ark/products/212275/intel-core-i5-11600k-processor-12m-cache-up-to-4-90-ghz.html
        "server_id":5,
        "gateway": "Intel® Core™ i7-6950X",
        "freq": 3.50e9,  # 3.9 GHz
        "TDP": 140,  # 125 Watts
        "IPC":9.16,  # 1.5 Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 10,  
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
    
    {
        #https://www.intel.com/content/www/us/en/products/sku/52214/intel-core-i72600k-processor-8m-cache-up-to-3-80-ghz/specifications.html
        "server_id":6,
        "gateway": "Intel Core i7 2600K ",
        "freq": 3.50e9,  # 3.9 GHz
        "TDP": 95,  #  Watts
        "IPC":8.61,  # 1.5 Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 4,  
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
]



class TaskGeneratorInterface(ABC):
    @abstractmethod
    def __init__(self,devices:List[EndDevice],gateways:List[Gateway]):
       pass
        

    @abstractmethod
    def generate_tasks(self,max_time,tasks_num ,task_generation_prob=0.3, task_size_min=1, task_size_max=1000):
       pass

class ProposedTaskGenerator(TaskGeneratorInterface):
    def __init__(self, devices: List[EndDevice], gateways: List[Gateway]):
        self.next_available_id = 0
        self.devices = devices
        self.gateways = gateways
        self.generated_tasks = []

    def generate_tasks(self, max_time, tasks_num, task_generation_prob=1, task_size_min=1, task_size_max=1000):
        self.generated_tasks = []
        # Only generate the additional tasks needed
        print(f"in generate tasks \n")
        additional_tasks_needed = tasks_num 
        # - len(self.generated_tasks)

        # If no additional tasks are needed, return the existing tasks
        if additional_tasks_needed <= 0:
            return len(self.generated_tasks), self.generated_tasks

        count_generated_task = 0
        while count_generated_task < additional_tasks_needed:
            for device in self.devices:
                if count_generated_task >= additional_tasks_needed:
                    break

                task = device.generate_task(
                    self.next_available_id,
                    task_generation_prob,
                    task_size_min,
                    task_size_max,
                    max_time
                )

                if task:
                    self.generated_tasks.append(task)
                    self.next_available_id += 1
                    device.task_queue.append(task)
                    count_generated_task += 1
        # for x in self.generated_tasks:            
            # print(x)
        # print(len(self.generated_tasks),"----------------*-*--*-*-*--*-*-*-*-")    
        return len(self.generated_tasks), self.generated_tasks

    
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
        import time
        scheduled_tasks = []
        # print(f"in seelct servwers {tasks}")
        for task in tasks:
            valid_servers = self.get_valid_servers(task)
            if valid_servers:
                selected_server,min_cost = self.get_min_cost_server(task, valid_servers)
                task.execution_server = selected_server
                task.cost= min_cost
                scheduled_tasks.append(task)

            # else:
            #     task.execution_server = task.default_server

           


        
        print("After server selection ")
        # for x in scheduled_tasks:
        #     print(x.execution_server.id)
        # print("****************************************************")    
        # time.sleep(60)
        return scheduled_tasks
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

        transfer_time = task.size / min(task.default_server.uplink_bandwidth, server.downlink_bandwidth)
        
        if task.default_server == server:
            transfer_time=0
        
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
        return selected_server,min_cost

    def calculate_total_cost(self, task: Task, server: Server) -> float:
        """
        Calculate the total cost of executing the task on the given server, including data transfer and processing costs.
        """
        transfer_cost = 0
        if server.id != task.default_server.id:
            transfer_cost = task.size * (task.default_server.uplink_cost + server.downlink_cost)
        processing_cost = server.processing_cost(task)
        return transfer_cost + processing_cost

class ASCO_Scheduler:
    def __init__(self, simulation_instance):
        self.gateways = simulation_instance.gateways
        self.servers = simulation_instance.servers
        self.simulation_instance = simulation_instance
        self.server_tasks = {server.id: set() for server in self.servers}
        self.server_dict = {server.id: server for server in self.servers}
        self.flows = []

    def calculate_normalized_flow(self, task):
        # Assume it is always positive
        return task.size / min(task.default_server.uplink_bandwidth, task.execution_server.downlink_bandwidth)

    def normalize_flows(self, tasks):
        self.flows = []
        for task in tasks:
            if task.execution_server != task.default_server:
                task.flow = self.calculate_normalized_flow(task)
                self.flows.append(task)

    def control_admission(self, flows):
        removed_flows_count = 0
        while True:
            most_critical_interval, most_critical_server, most_critical_server_flows, most_critical_intensity = self.find_most_critical_interval(flows)
            if most_critical_interval is None or most_critical_intensity <= 1:
                break
            if self.remove_max_flow_from_interval(most_critical_server_flows,flows):
                removed_flows_count += 1
        return removed_flows_count

    def remove_max_flow_from_interval(self, flows,total_flows):
        max_flow = max(flows, key=lambda flow: flow.flow / (flow.deadline - flow.arrival_time), default=None)
        if max_flow:
            flows.remove(max_flow)
            total_flows.remove(max_flow)
            self.server_tasks[max_flow.execution_server.id].remove(max_flow)
            server = self.server_dict[max_flow.execution_server.id]
            server.server_tasks.remove(max_flow)
            server.preempted_tasks.append(max_flow)
            return True
        
        return False

    def select_server_for_tasks(self, tasks):
        return ProposedServerSelection(self.servers).select_servers(tasks)


    def schedule_for_computation(self, tasks: List[Task]):
        for task in tasks:
            self.server_tasks[task.execution_server.id].add(task)

        # print(f"in computation calculation {tasks}" )
        scheduled_tasks = []
        for server_id, task_set in self.server_tasks.items():
            # print("enter")
            x=self.schedule(task_set, self.server_dict[server_id])
            # print(x)
            scheduled_tasks.extend(x)

        return scheduled_tasks    

    def apply_ASCO(self, tasks):
        total_tasks_num = len(tasks)
        scheduled_tasks = self.select_server_for_tasks(tasks)
        scheduled_tasks = self.schedule_for_computation(scheduled_tasks)
        self.normalize_flows(scheduled_tasks)
        removed_flows_count = self.control_admission(scheduled_tasks)

        processed_tasks_num = len(scheduled_tasks)

        removed_tasks = [] #from the server scheduling  or admission control  but not from the server selection
        for server in self.servers:
            removed_tasks.extend(server.preempted_tasks)

        return {
            "received_tasks_num": total_tasks_num,
            "removed_tasks": removed_tasks,
            "pruned_flows_num": removed_flows_count,
            "processed_tasks_num":processed_tasks_num,
            "scheduled_tasks":scheduled_tasks
        }

    def apply(self, tasks):
        return self.apply_ASCO(tasks)

    def calculate_intensity(self, flows, interval):
        start_time, end_time = interval
        total_flow = sum(flow.flow for flow in flows if start_time <= flow.arrival_time < end_time)
        return total_flow / (end_time - start_time)

    def calculate_free_time_interval(self, flows):
        start_time = float('inf')
        end_time = float('-inf')
        for flow in flows:
            if flow.arrival_time < start_time:
                start_time = flow.arrival_time
            if flow.starting_time > end_time:
                end_time = flow.starting_time
        return start_time, end_time

    def find_most_critical_interval(self, flows):
        if not flows:
            return None, None, None, None

        most_critical_intensity = 0
        most_critical_interval = None
        most_critical_server = None

        # Map each server to its respective flows
        server_flow = {server.id: [] for server in self.servers}
        flows.sort(key=lambda t: t.arrival_time)  # Sort flows by arrival time
        for flow in flows:
            if flow.default_server.id != flow.execution_server:
                server_flow[flow.default_server.id].append(flow)

        # Iterate through each server
        for server in self.simulation_instance.servers:
            curr_flows = server_flow.get(server.id)
            if not curr_flows:
                continue

            start_time, end_time = self.calculate_free_time_interval(curr_flows)

            # Prefix sum for flows within the current server
            prefix_sums = [0] * (end_time - start_time + 1)
            for flow in curr_flows:
                arrival_idx = flow.arrival_time - start_time
                if 0 <= arrival_idx < len(prefix_sums):
                    prefix_sums[arrival_idx] += flow.flow

            for i in range(1, len(prefix_sums)):
                prefix_sums[i] += prefix_sums[i - 1]

            # Iterate through possible intervals using prefix sums
            for i in range(start_time, end_time):
                for j in range(i + 1, end_time + 1):
                    interval_flow_sum = prefix_sums[j - start_time - 1] - (prefix_sums[i - start_time - 1] if i > start_time else 0)
                    interval_length = j - i
                    curr_intensity = interval_flow_sum / interval_length

                    if curr_intensity > most_critical_intensity:
                        most_critical_intensity = curr_intensity
                        most_critical_interval = (i, j)
                        most_critical_server = server

        if most_critical_interval is None:
            return None, None, None, None

        return most_critical_interval, most_critical_server, server_flow.get(most_critical_server.id, []), most_critical_intensity

    def schedule(self, tasks, server):
        reverse_tasks = []
        for task in tasks:
            reverse_tasks.append(task)
        reverse_tasks.sort(key=lambda t: t.deadline)
        scheduled_tasks = []
        # print(f"In scheduling")
        while reverse_tasks:
            next_task = reverse_tasks.pop(0)
            task_end_time = next_task.deadline
            task_start_time = task_end_time - server.processing_time(next_task)
            # print(f"current Task is {next_task}")
            if server.remaining_comp_capacity(task_start_time,task_end_time)>=server.IPS:
                # print("server can handle it")
                next_task.starting_time = task_start_time
                next_task.end_time = task_end_time
                scheduled_tasks.append(next_task)
                server.server_tasks.append(next_task)
                next_task.execution_server = server
            else:
                # print("server can not handle it and trying ...")
                preempted_task = self.SRTF(task_start_time,task_end_time,next_task,server,tasks)
                # print("preempted_task" , preempted_task)
                if preempted_task:
                    server.remove_task(preempted_task.id)
                    next_task.starting_time = task_start_time
                    next_task.end_time = task_end_time
                    scheduled_tasks.append(next_task)
                    server.server_tasks.append(next_task)
                    next_task.execution_server = server
                    server.preempted_tasks.append(preempted_task)

        return scheduled_tasks


    def SRTF(self, start_time, end_time, new_task,server,tasks): #o(tasks)
        shortest_task = None
        for task in server.server_tasks:
            task_end_time = task.starting_time + server.processing_time(task)
            if task.starting_time < end_time and task_end_time > start_time:
                overlap_start = max(task.starting_time, start_time)
                overlap_end = min(task_end_time, end_time)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    if shortest_task is None or server.processing_time(task) < server.processing_time(shortest_task):
                        shortest_task = task
        if shortest_task:
            already_consumed_time = start_time-shortest_task.starting_time
            if already_consumed_time <0:
                already_consumed_time=0
            remaining_time = server.processing_time(shortest_task) - (already_consumed_time)
            if server.processing_time(new_task) < remaining_time:
               
                return shortest_task
            
            return None
        return None

from typing import List


class Baseline_Scheduler:
    def __init__(self, simulation_instance):
        self.gateways = simulation_instance.gateways
        self.servers = simulation_instance.servers
        self.simulation_instance = simulation_instance


    def schedule_for_computation(self, tasks: List[Task]):
        # print(f" the Tasks: {tasks}")
        server_task = {server.id: [] for server in self.servers}
        scheduled_tasks = []
        for task in tasks:
            server_task[task.execution_server.id].append(task)

        for server_id, tasks in server_task.items():
            server = next(server for server in self.servers if server.id == server_id)
            scheduled_tasks.extend(self.schedule(tasks, server))
        
        
        return scheduled_tasks
    def EDF(self, new_task: Task, start: int, end: int, server: Server):
        selected_tasks = [new_task]
        overlap_duration = 0

        for task in server.server_tasks:
            if task.end_time > start and task.starting_time < end:
                overlap_start = max(start, task.starting_time)
                overlap_end = min(end, task.end_time)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    selected_tasks.append(task)

        selected_tasks.sort(key=lambda t: t.deadline)

        if selected_tasks[-1] == new_task:
            return None
        else:
            return selected_tasks[-1]

    def schedule(self, tasks: List[Task], server: Server):
        tasks.sort(key=lambda t: t.arrival_time)
        scheduled_tasks=[]
        for task in tasks:
            start_time = task.arrival_time
            end_time = start_time + server.processing_time(task)

            if server.remaining_comp_capacity(start_time, end_time) >= server.IPS:
                task.starting_time = start_time
                task.end_time = end_time
                server.server_tasks.append(task)
                task.cost= server.processing_cost(task)
                scheduled_tasks.append(task)
            else:
                preempted_task = self.EDF(task, start_time, end_time, server)
                if preempted_task:
                    task.starting_time = start_time
                    task.end_time = end_time
                    task.cost= server.processing_cost(task)
                    scheduled_tasks.append(task)
                    server.server_tasks.append(task)
                    server.preempted_tasks.append(preempted_task)
                    server.server_tasks.remove(preempted_task)
        return scheduled_tasks
    def apply(self, tasks: List[Task]):
        scheduled_tasks= self.schedule_for_computation(tasks)
        
        total_tasks_num = len(tasks)
        # print(f"{self.servers}")
        removed_tasks = []
        for server in self.servers:
            removed_tasks.extend(server.preempted_tasks)
        return {
            "received_tasks_num": total_tasks_num,
            "removed_tasks": removed_tasks,
            "scheduled_tasks":scheduled_tasks
        }
        


class Simulation:

    def __init__(self, network_generator: NetworkGenerator, task_generator: TaskGeneratorInterface, server_selector: ServerSelectionInterface):
        self.network_generator = network_generator
        self.task_generator = task_generator
        self.current_generated_tasks = 0
        self.current_completed_tasks = 0
        self.offloaded_tasks = []
        self.server_selector = server_selector
        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.servers = self.network_generator.servers
   
    def init_network(self):
        self.network_generator.generate_pcp_network()

    def paper_algo(self, generated_tasks):
        output_data = ASCO_Scheduler(simulation_instance=self).apply(generated_tasks)
        # for task in generated_tasks:
        #     print(task) 
        #     print("--"*10)
        # print(f"the o   {generated_tasks}")

        total_cost = sum(task.cost for task in output_data["scheduled_tasks"])
        # removed_cost = sum(task.cost for task in output_data["removed_tasks"])
        avg_cost_per_task = 0
        if total_cost != 0:
            avg_cost_per_task = (total_cost) / (len( output_data["scheduled_tasks"])) 
        # avg_cost_per_task = (total_cost - removed_cost)
        
        output_data["avg_cost_per_task"] = avg_cost_per_task
        # print(f"the output of the paper algo {output_data}")
        return output_data
    
    def baseline_algo(self, generated_tasks):
        
        output_data = Baseline_Scheduler(simulation_instance=self).apply(generated_tasks)
        total_cost = sum(task.cost for task in output_data["scheduled_tasks"])
        # removed_cost = sum(task.cost for task in output_data["removed_tasks"])
        avg_cost_per_task = (total_cost) /(len(output_data["scheduled_tasks"])) if generated_tasks and output_data["scheduled_tasks"] else 0
        # avg_cost_per_task = (total_cost - removed_cost)
        
        output_data["avg_cost_per_task"] = avg_cost_per_task
        # print(f"the output of the base algo {output_data}")
        return output_data

    def run(self, mode,tasks=None):
        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.servers = self.network_generator.servers

        if mode == 1:
            return self.paper_algo(tasks)
        else:
            return self.baseline_algo(tasks)

def run_simulation(tasks_num,network):
    
    task_generator = ProposedTaskGenerator(network.devices, network.gateways)
    server_selector = ProposedServerSelection(network.servers)
    simulation = Simulation(network, task_generator, server_selector)
    tasks_results = {}
    for i in tasks_num: # o (n) 
        # print(f"{i} lg\n")
        _, generated_tasks = task_generator.generate_tasks(None, int(i * 1000), task_generation_prob=1, task_size_min=10, task_size_max=1000)
        # print(len(generated_tasks)," TTTTTTTT\n")
        paper_gen = generated_tasks
        paper_output = simulation.run(1,generated_tasks)
        paper_admission = (len(paper_output['scheduled_tasks'])) / paper_output['received_tasks_num']
        paper_cost = paper_output['avg_cost_per_task']
        paper_results= (paper_admission, paper_cost)
        
        _, generated_tasks = task_generator.generate_tasks(None, int(i * 1000), task_generation_prob=1, task_size_min=10, task_size_max=1000)
        base_output = simulation.run(2,generated_tasks)
        base_admission = (len(paper_output['scheduled_tasks'])) / base_output['received_tasks_num']
        base_cost = base_output['avg_cost_per_task']
        base_results= (base_admission, base_cost)
        # print(f"{base_output} ddddd.  ")

        tasks_results[i] = {
        "paper": paper_results, 
        "baseline": base_results,
        "paper_tasks": {"generated":paper_gen,"scheduled_tasks":paper_output["scheduled_tasks"]},
        "baseline_tasks":{"generated":generated_tasks,"scheduled_tasks":base_output["scheduled_tasks"]}
        }
    return tasks_results    

    # visualize_results(paper_output, base_output)
def visualize_results(tasks_results_map, title, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Task Admission Rate
    ax1.set_title('Task Admission Rate')
    ax1.set_xlabel('Number of tasks (K)')
    ax1.set_ylabel('Ratio to baseline')

    # Average Per-Task Cost
    ax2.set_title('Average Per-Task Cost')
    ax2.set_xlabel('Number of tasks (K)')
    ax2.set_ylabel('Ratio to baseline')

    tasks_num = sorted(tasks_results_map.keys())

    admission_ratios = []
    cost_ratios = []

    for tasks_num_value in tasks_num:
        paper_results = tasks_results_map[tasks_num_value]["paper"]
        base_results = tasks_results_map[tasks_num_value]["baseline"]

        paper_admissions, paper_avg_cost = paper_results
        base_admissions, base_avg_cost = base_results

        admission_ratios.append(paper_admissions / base_admissions)
        cost_ratios.append(paper_avg_cost / base_avg_cost)

    ax1.plot(tasks_num, admission_ratios, marker='o')
    ax2.plot(tasks_num, cost_ratios, marker='o')

    ax1.set_xticks(tasks_num)
    ax1.set_xticklabels(tasks_num)
    ax2.set_xticks(tasks_num)
    ax2.set_xticklabels(tasks_num)

    plt.tight_layout()
    plt.suptitle(title)

    graph_filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(graph_filename)
    plt.close()  # Close the plot to free up memory

    graph_filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(graph_filename)
    plt.close()  # Close the plot to free up memory


# def save_data_to_file(data, filename):
#     """Save data to a JSON file."""
#     with open(filename, 'w') as f:
#         # json_data = {k: {
#         #     "paper": [(p[0], p[1]) for p in v["paper"]],
#         #     "baseline": [(b[0], b[1]) for b in v["baseline"]]
#         # } for k, v in data.items()}
#         json.dump(data, f, indent=4)

import json

import json
from typing import Any

def save_data_to_file(data: dict[str, Any], filename: str):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json_data = {
            k: {
                "paper": v["paper"],
                "baseline": v["baseline"],
                "paper_tasks": {
                    "generated": [task.to_dict() for task in v["paper_tasks"]["generated"]],
                    "scheduled_tasks": [task.to_dict() for task in v["paper_tasks"]["scheduled_tasks"]]
                },
                "baseline_tasks": {
                    "generated": [task.to_dict() for task in v["baseline_tasks"]["generated"]],
                    "scheduled_tasks": [task.to_dict() for task in v["baseline_tasks"]["scheduled_tasks"]]
                }
            } for k, v in data.items()
        }
        json.dump(json_data, f, indent=4)

def load_data_from_file(filename, convert_to_tuples=False):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        json_data = json.load(f)
    return json_data
        

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""

    if not os.path.exists(directory):
        os.makedirs(directory)

import os

import matplotlib.pyplot as plt
import os

def generate_individual_graphs(tasks_results_map, output_dir, scale):
    # print(f"******************************************************* {tasks_results_map}")
    
    x_axis = []
    paper_admission_rates = []
    paper_avg_costs = []
    baseline_admission_rates = []
    baseline_avg_costs = []

    for number_tasks, data in tasks_results_map.items():
        # print(f"******************************************************* {number_tasks} {data}")
        x_axis.append(number_tasks)
        
        paper_admission_rate, paper_avg_cost = data["paper"]
        baseline_admission_rate, baseline_avg_cost = data["baseline"]
        
        paper_admission_rates.append(paper_admission_rate)
        paper_avg_costs.append(paper_avg_cost)
        baseline_admission_rates.append(baseline_admission_rate)
        baseline_avg_costs.append(baseline_avg_cost)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Task Admission Rate
    ax1.set_title('Task Admission Rate')
    ax1.set_xlabel('Number of Tasks')
    ax1.set_ylabel('Admission Rate')
    ax1.plot(x_axis, paper_admission_rates, marker='o', label='Paper Admission Rate')
    ax1.plot(x_axis, baseline_admission_rates, marker='x', label='Baseline Admission Rate')
    ax1.legend()

    # Average Per-Task Cost
    ax2.set_title('Average Per-Task Cost')
    ax2.set_xlabel('Number of Tasks')
    ax2.set_ylabel('Average Cost')
    ax2.plot(x_axis, paper_avg_costs, marker='o', label='Paper Avg Cost')
    ax2.plot(x_axis, baseline_avg_costs, marker='x', label='Baseline Avg Cost')
    ax2.legend()

    plt.tight_layout()

    graph_filename = os.path.join(output_dir, f"{scale}_detailed_graphs.png")
    plt.savefig(graph_filename)
    plt.close()  # Close the plot to free up memory


def main():
    start_execution_time = time.time()
    
    print(f"start_execution_time:  {time.ctime(start_execution_time)}")
    tasks_small_num = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0,3.5,4.0]
    tasks_big_num = [2, 4, 6, 8, 12,14,16]
    
    max_time =[60*60] #60*1 = 1 min , 60 is sec number  
    duration_max=[60*3]
    

   
    

    list_ =[ 
      
         [0.5, 1.0, 1.5, 2.0, 2.5, 3.0,3.5,4.0],
         [2, 4, 6, 8, 12,14,16]
        
        ]

    for j in range(len(max_time)):
        for k in range(len(duration_max)):
            global max_time_
            global max_duration

            max_time_= max_time[j]
            max_duration = duration_max[k]

            for i in range(0,len(list_)): 
                output_dir = "simulation_results/"+str(max_time_)+"_"+str(max_duration)
                ensure_directory_exists(output_dir)
                data_dir = os.path.join(output_dir, "data")
                ensure_directory_exists(data_dir)
                graphs_dir = os.path.join(output_dir, "graphs")
                ensure_directory_exists(graphs_dir)

                network_small = NetworkGenerator(num_clusters=20, num_devices=5)
                network_small.generate_pcp_network()
                

                network_large = NetworkGenerator(num_clusters=100, num_devices=5)
                network_large.generate_pcp_network()

                small_scale_results = run_simulation(list_[i], network_small)
                generate_individual_graphs(small_scale_results, graphs_dir, generate_file_name(i,"Small_Scale",""))
                save_data_to_file(small_scale_results, os.path.join(data_dir, generate_file_name(i,"Small_Scale","json")))

                
                print_percentages(small_scale_results,"Small_Scale", os.path.join(data_dir, generate_file_name(i,"Small_Scale","txt")))
                
                
                large_scale_results = run_simulation(list_[i], network_large)
                generate_individual_graphs(large_scale_results, graphs_dir,generate_file_name(i,"large_Scale",""))
                save_data_to_file(large_scale_results, os.path.join(data_dir, generate_file_name(i,"large_Scale","json")))

                # generate_file_name(i,"Small_Scale","txt")

             
                print_percentages(large_scale_results,"large_Scale",os.path.join(data_dir, generate_file_name(i,"large_Scale","txt")))
                # print_percentages(mid_scale_results,"mid_Scale",os.path.join(data_dir, generate_file_name(i,"mid_Scale","txt")))

    
    end_execution_time = time.time()
    total_time = end_execution_time- start_execution_time
    print(f"start_execution_time:  {time.ctime(end_execution_time)}")
    print("total execution time", total_time)



   
def generate_file_name(iteration_num, title, extension) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Format time to be filename-friendly
    return f"{iteration_num}_{timestamp}_{title}.{extension}"


def print_percentages(results, title, file):
    with open(file, "w") as f:
        f.write(f"**************** {title} *****************\n")
        print(f"**************** {title} *****************\n")
        
        for num, result in results.items():
            paper_admission, paper_cost = result["paper"]
            baseline_admission, baseline_cost = result["baseline"]
            admission_rate_improvement = 0
            if baseline_admission != 0:
                admission_rate_improvement = ((paper_admission - baseline_admission) / baseline_admission) * 100
            cost_per_task_reduction = ((baseline_cost - paper_cost) / baseline_cost) * 100
            
            f.write(f"K = {num}\nadmission_rate_improvement: {admission_rate_improvement:.5f} %\ncost_per_task_reduction: {cost_per_task_reduction:.5f} %\n\n")
            print(f"K = {num}\nadmission_rate_improvement: {admission_rate_improvement:.5f} %\ncost_per_task_reduction: {cost_per_task_reduction:.5f} %\n")
main()