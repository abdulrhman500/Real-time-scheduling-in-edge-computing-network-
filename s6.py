import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List
from abc import ABC, abstractmethod

import json
import os
# Task Class
# import matplotlib.pyplot as plt
# import numpy as np
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

    def generate_task(self, id, task_generation_prob, task_size_min, task_size_max, max_time):
        # from __main__ import Simulation
        if random.random() <= task_generation_prob and self.max_waiting_tasks > len(self.task_queue):
            task_size = random.randint(task_size_min, task_size_max)
            if max_time is None:
                max_time = 100
            arrival_time = random.randint( 1,  max_time)
            deadline = random.randint( arrival_time+1, + arrival_time+max_time)

            return Task(task_size,  deadline, id=id, arrival_time=arrival_time, default_server=self.server)
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
        self.preempted_tasks=[]
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
    def generate_tasks(self,max_time,tasks_num ,task_generation_prob=0.3, task_size_min=1, task_size_max=1000):
       pass

class ProposedTaskGenerator(TaskGeneratorInterface):
    
    def __init__(self, devices: List[EndDevice], gateways: List[Gateway]):
        self.next_available_id = 0
        self.devices = devices
        self.gateways = gateways  

    def generate_tasks(self, max_time, tasks_num ,task_generation_prob=1, task_size_min=1, task_size_max=1000):
        count_generated_task = 0
        generated_tasks = []
        while len(generated_tasks)<tasks_num:
            for device in self.devices:
                
                
                task = device.generate_task(self.next_available_id, task_generation_prob, 
                                            task_size_min, task_size_max, max_time)
                
                if task:
                    generated_tasks.append(task)
                    self.next_available_id += 1
                    device.task_queue.append(task)
                    count_generated_task += 1
                    if len(generated_tasks)>=tasks_num : break
                    
        
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
                selected_server,min_cost = self.get_min_cost_server(task, valid_servers)
                task.execution_server = selected_server
                task.cost= min_cost

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
    
    def __init__(self,simulation_instance):
        self.gateways = simulation_instance.gateways
        self.servers = simulation_instance.servers
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
        removed_flows_count = 0
        while True:
            most_critical_interval, most_critical_server, most_critical_server_flows, most_critical_intensity = self.find_most_critical_interval(flows)

            if most_critical_interval is None:
                break

            if most_critical_intensity <= 1:
                break

            if self.remove_max_flow_from_interval(most_critical_server_flows):
                removed_flows_count+=1

        return removed_flows_count

             
    def remove_max_flow_from_interval(self, flows: List[Task]):
        max_flow = max(flows, key=lambda flow: flow.normalized_size/(flow.deadline-flow.arrival_time), default=None)
        if max_flow:
            flows.remove(max_flow)
            max_flow.execution_server.remove_task(max_flow.id)
            return True
        return False


    def select_server_for_tasks(self,tasks):
        ProposedServerSelection(self.servers).select_servers(tasks)
        
      

    def schedule_for_computation(self, tasks: List[Task]):
        server_task = {server.id: [] for server in self.servers}

        for task in tasks:
            server_task[task.execution_server.id].append(task)

        for server_id, tasks in server_task.items():
            server = next(server for server in self.servers if server.id == server_id)
            self.schedule(tasks,server)

    def apply_ASCO(self,tasks):
        total_tasks_num = len(tasks)

        self.select_server_for_tasks(tasks)
        self.schedule_for_computation(tasks)
        self.normalize_flows(tasks)
        removed_flows_count = self.control_admission(tasks)

        # number of scheduled tasks 
        removed_tasks = []
        for server in self.servers:
            removed_tasks.extend(server.preempted_tasks)
        return {
            "received_tasks":total_tasks_num,
            "pruned_flows":removed_flows_count,
            "preempted_tasks":sum(len(server.preempted_tasks) for server in self.servers),
            "removed_tasks":removed_tasks
        }



    def apply(self,tasks:List[Task]):
        return self.apply_ASCO(tasks)

    def schedule(self, tasks, server):
        reverse_tasks = []
        for task in tasks:
            reverse_tasks.append(task)
        reverse_tasks.sort(key=lambda t: t.deadline)
        scheduled_tasks = []
        while reverse_tasks:
            next_task = reverse_tasks.pop(0)
            task_end_time = next_task.deadline
            task_start_time = task_end_time - server.processing_time(next_task)
            if server.remaining_comp_capacity(task_start_time,task_end_time)>=server.IPS:
                next_task.starting_time = task_start_time
                next_task.end_time = task_end_time
                scheduled_tasks.append(next_task)
                server.server_tasks.append(next_task)
                next_task.execution_server = server
            else:
                preempted_task = self.SRTF(task_start_time,task_end_time,tasks,server)
                if preempted_task:
                    scheduled_tasks.append(next_task)
                    server.preempted_tasks.append(preempted_task)

            return scheduled_tasks


    def SRTF(self, start_time, end_time, new_task,server): #o(tasks)
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
            remaining_time = server.processing_time(shortest_task)
            if server.processing_time(new_task) < remaining_time:
                server.remove_task(shortest_task.id)
                server.server_tasks.append(new_task)
                new_task.starting_time = start_time
                return new_task
        return None



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
        for server in self.simulation_instance.servers:
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




from typing import List

class Baseline_Scheduler:
    def __init__(self, simulation_instance):
        self.gateways = simulation_instance.gateways
        self.servers = simulation_instance.servers
        self.simulation_instance = simulation_instance


    def schedule_for_computation(self, tasks: List[Task]):
        # print(f" the Tasks: {tasks}")
        server_task = {server.id: [] for server in self.servers}

        for task in tasks:
            server_task[task.execution_server.id].append(task)

        for server_id, tasks in server_task.items():
            server = next(server for server in self.servers if server.id == server_id)
            self.schedule(tasks, server)

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

        for task in tasks:
            start_time = task.arrival_time
            end_time = start_time + server.processing_cost(task)

            if server.remaining_comp_capacity(start_time, end_time) >= server.IPS:
                task.starting_time = start_time
                task.end_time = end_time
                server.server_tasks.append(task)
            else:
                preempted_task = self.EDF(task, start_time, end_time, server)
                if preempted_task:
                    task.starting_time = start_time
                    task.end_time = end_time
                    server.server_tasks.append(task)
                    server.preempted_tasks.append(preempted_task)
                    server.server_tasks.remove(preempted_task)

    def apply(self, tasks: List[Task]):
        self.schedule_for_computation(tasks)
        
        total_tasks_num = len(tasks)
        # print(f"{self.servers}")
        removed_tasks = []
        for server in self.servers:
            removed_tasks.extend(server.preempted_tasks)

        return {
            "received_tasks": total_tasks_num,
            "preempted_tasks": sum(len(server.preempted_tasks) for server in self.servers),
            "removed_tasks": removed_tasks
        }
        


class Simulation:
    global_clock = 0  

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
        total_cost = sum(task.size for task in generated_tasks)
        removed_cost = sum(task.size for server_tasks in output_data["removed_tasks"] for task in server_tasks)
        avg_cost_per_task = (total_cost - removed_cost) / len(generated_tasks) if generated_tasks else 0
        output_data["avg_cost_per_task"] = avg_cost_per_task
        print(f"the output of the paper algo {output_data}")
        return output_data

    def baseline_algo(self, generated_tasks):
        
        output_data = Baseline_Scheduler(simulation_instance=self).apply(generated_tasks)
        total_cost = sum(task.size for task in generated_tasks)
        removed_cost = sum(task.size for server_tasks in output_data["removed_tasks"] for task in server_tasks)
        avg_cost_per_task = (total_cost - removed_cost) / len(generated_tasks) if generated_tasks else 0
        output_data["avg_cost_per_task"] = avg_cost_per_task
        print(f"the output of the base algo {output_data}")
        return output_data

    def run(self, mode,tasks=None):
        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.servers = self.network_generator.servers

        if mode == 1:
            return self.paper_algo(tasks)
        else:
            return self.baseline_algo(tasks)

        # print(f"Gateway: {len(self.gateways)}\nDevices: {len(self.devices)}\nServers: {len(self.servers)}")
        
    # def run_tasks(self, num_tasks):
    #     total_generated_tasks = 0
    #     while total_generated_tasks <= num_tasks:
    #         count_generated_task, generated_tasks = self.task_generator.generate_tasks(None, task_generation_prob=1, task_size_min=10, task_size_max=1000)
    #         total_generated_tasks += len(generated_tasks)
    #         ASCO_Scheduler(simulation_instance=self).apply(generated_tasks)
    #         self.global_clock += 1

    # def run_time(self, time):
    #     for i in range(time):
    #         count_generated_task, generated_tasks = self.task_generator.generate_tasks(time, task_generation_prob=0.1, task_size_min=10, task_size_max=1000)
    #         ASCO_Scheduler(simulation_instance=self).apply(generated_tasks)
    #         self.global_clock += 1
    #         print(i)

    # def offload_tasks(self, generated_tasks):
    #     return generated_tasks


import matplotlib.pyplot as plt
import numpy as np



# def main():
        
#     tasks_small_num = [0.5,1.0,1.5,2.0,2.5,3.0]
#     tasks_big_num = [2,4,6,8,12]
   
#     run_simulation(tasks_small_num,20)
#     run_simulation(tasks_big_num,20)
    
#     run_simulation(tasks_small_num,100)
#     run_simulation(tasks_big_num,100)
    
    



def run_simulation(tasks_num,network):
    
    task_generator = ProposedTaskGenerator(network.devices, network.gateways)
    server_selector = ProposedServerSelection(network.servers)
    simulation = Simulation(network, task_generator, server_selector)
    tasks_results = {}
    for i in tasks_num: # o (n) 
        # print(f"{i} lg\n")
        _, generated_tasks = task_generator.generate_tasks(None, int(i * 1000), task_generation_prob=1, task_size_min=10, task_size_max=1000)
        # print(len(generated_tasks)," TTTTTTTT\n")
        paper_output = simulation.run(1,generated_tasks)
        paper_admission = (paper_output['received_tasks'] - paper_output['preempted_tasks']) / paper_output['received_tasks']
        paper_cost = paper_output['avg_cost_per_task']
        paper_results= (paper_admission, paper_cost)
        
        
        base_output = simulation.run(2,generated_tasks)
        base_admission = (base_output['received_tasks'] - base_output['preempted_tasks']) / base_output['received_tasks']
        base_cost = base_output['avg_cost_per_task']
        base_results= (base_admission, base_cost)
        # print(f"{base_output} ddddd.  ")

        tasks_results[i] = {"paper": paper_results, "baseline": base_results}
    
    return tasks_results    

    # visualize_results(paper_output, base_output)
    
def visualize_results(tasks_results_map, title,output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Task Admission Rate
    ax1.set_title('Task Admission Rate')
    ax1.set_xlabel('Number of tasks (K)')
    ax1.set_ylabel('Ratio to baseline')
    
    # Average Per-Task Cost
    ax2.set_title('Average Per-Task Cost')
    ax2.set_xlabel('Number of tasks (K)')
    ax2.set_ylabel('Ratio to baseline')
    
    positions = np.arange(len(tasks_results_map))
    width = 0.4
    
    admission_ratios = []
    cost_ratios = []
    print("888 \n")
    print(f"{tasks_results_map}")
    for i, tasks_num in enumerate(tasks_results_map):
        print(f"{i} {tasks_num}")
        paper_results = tasks_results_map[tasks_num]["paper"]
        base_results = tasks_results_map[tasks_num]["baseline"]
        
        paper_admissions, paper_avg_cost = paper_results
        base_admissions, base_avg_cost =base_results
        
        admission_ratios.append((paper_admissions/ base_admissions))
        cost_ratios.append(paper_avg_cost/ base_avg_cost)
        
        ax1.boxplot(admission_ratios, positions=[i], widths=width)
        ax2.boxplot(cost_ratios, positions=[i], widths=width)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tasks_results_map.keys())
    ax2.set_xticks(positions)
    ax2.set_xticklabels(tasks_results_map.keys())
    
    plt.tight_layout()
    plt.suptitle(title)
    # plt.show()

    graph_filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(graph_filename)
    plt.close()  # Close the plot to free up memory


def save_data_to_file(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        # json_data = {k: {
        #     "paper": [(p[0], p[1]) for p in v["paper"]],
        #     "baseline": [(b[0], b[1]) for b in v["baseline"]]
        # } for k, v in data.items()}
        json.dump(data, f, indent=4)

def load_data_from_file(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        json_data = json.load(f)
        return {int(k): {
            "paper": [tuple(p) for p in v["paper"]],
            "baseline": [tuple(b) for b in v["baseline"]]
        } for k, v in json_data.items()}

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    tasks_small_num = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    tasks_big_num = [2, 4, 6, 8, 12]

    output_dir = "simulation_results"
    ensure_directory_exists(output_dir)
    data_dir = os.path.join(output_dir, "data")
    ensure_directory_exists(data_dir)
    graphs_dir = os.path.join(output_dir, "graphs")
    ensure_directory_exists(graphs_dir)


    network_small = NetworkGenerator(num_clusters=20, num_devices=15)
    network_small.generate_pcp_network()

    # Small scale with 20 cloudlets
    small_scale_results = run_simulation(tasks_small_num,network_small)
    visualize_results(small_scale_results, "Small scale with 20 cloudlets",graphs_dir)
    save_data_to_file(small_scale_results, os.path.join(data_dir, "small_scale_results.json"))


    network_large = NetworkGenerator(num_clusters=100, num_devices=15)
    network_large.generate_pcp_network()

    # Large scale with 100 cloudlets
    large_scale_results = run_simulation(tasks_big_num,network_large)
    print(f"ffv {large_scale_results}")
    visualize_results(large_scale_results, "Large scale with 100 cloudlets",graphs_dir)
    save_data_to_file(large_scale_results, os.path.join(data_dir, "large_scale_results.json"))

main()
