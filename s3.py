import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from typing import List

# Task Class
class Task:
    def __init__(self, size: int, deadline, starting_time=None, arrival_time=None, server=None, channel_gain=None, id=None, num_instruction_to_be_executed=None, default_server=None):
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

    def generate_task(self, id, task_generation_prob, task_size_min, task_size_max):
        if random.random() <= task_generation_prob and len(self.task_queue) < self.max_waiting_tasks:
            task_size = random.randint(task_size_min, task_size_max)
            deadline = random.randint(1, 100)
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
        self.tasks_running = []
        self.is_space_shared = True

    def processing_cost(self, task: Task):
        return self.EPI * task.num_instruction_to_be_executed

    def processing_time(self, task: Task):
        return task.num_instruction_to_be_executed / self.IPS

    def remaining_comp_capacity(self, start_time, end_time):
        if self.is_space_shared:
            used_cores = [0] * self.num_cores
            for task in self.tasks_running:
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
            for task in self.tasks_running:
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
            self.tasks_running.append(next_task)
            next_task.execution_server = self
        return scheduled_tasks

    def remove_task(self, task_id):
        for task in self.tasks_running:
            if task.id == task_id:
                self.tasks_running.remove(task)
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
                f"Tasks Running: {self.tasks_running}\n"
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
            server = self.create_random_server(gateway.id, center.tolist())
            self.servers.append(server)

            num_devices = np.random.poisson(self.lambda_c)
            devices = []
            for _ in range(num_devices):
                location = self.generate_device_location(center)
                device = self.create_device(location, standard_end_device, gateway.id, server.id)
                devices.append(device)
            self.devices.extend(devices)
            self.gateways.append(gateway)
        self.connect_devices_to_nearest_gateways()

    def generate_hpcp_network(self):
        """Generates a network using a Hierarchical Poisson Cluster Process (HPCP)."""
        macro_cell_centers = np.random.uniform(0, self.area_size, (self.num_clusters, 2))
        for macro_center in macro_cell_centers:
            num_micro_cells = np.random.poisson(self.lambda_p)
            micro_cell_centers = self.generate_micro_cell_centers(macro_center, num_micro_cells)
            for micro_center in micro_cell_centers:
                gateway = self.create_gateway(micro_center.tolist(), standard_gateway)
                server = self.create_random_server(gateway.id, micro_center.tolist())
                self.servers.append(server)

                num_devices = np.random.poisson(self.lambda_c)
                devices = []
                for _ in range(num_devices):
                    location = self.generate_device_location(micro_center)
                    device = self.create_device(location, standard_end_device, gateway.id, server.id)
                    devices.append(device)
                self.devices.extend(devices)
                self.gateways.append(gateway)
        self.connect_devices_to_nearest_gateways()

    def generate_device_location(self, center):
        return np.random.normal(loc=center, scale=self.sigma).tolist()

    def create_device(self, location, device_template, gateway_id, server_id):
        device_id = self.curr_device_id
        self.curr_device_id += 1
        return EndDevice(device_id=device_id, gateway=gateway_id, location=location, comp_capacity=device_template['comp_capacity'],
                         energy_per_cycle=device_template['energy_per_cycle'], cycle_per_bit=device_template['cycle_per_bit'],
                         max_waiting_tasks=device_template['max_waiting_tasks'], server=server_id)

    def create_gateway(self, location, gateway_template):
        gateway_id = self.curr_gateway_id
        self.curr_gateway_id += 1
        return Gateway(gateway_id=gateway_id, num_channels=gateway_template['num_channels'],
                       channel_bandwidths=gateway_template['channel_bandwidths'], location=location)

    def create_random_server(self, gateway_id, location):
        server_id = len(self.servers)
        return Server(server_id=server_id, gateway=gateway_id, freq=random.randint(1, 5), TDP=random.randint(20, 100),
                      IPC=random.uniform(0.5, 2), uplink_bandwidth=random.uniform(1, 10),
                      downlink_bandwidth=random.uniform(1, 10), uplink_cost=random.uniform(0.01, 0.1),
                      downlink_cost=random.uniform(0.01, 0.1), instruction_size=random.randint(100, 1000),
                      num_cores=random.randint(1, 8), energy_unit_price=random.uniform(0.05, 0.2))

    def connect_devices_to_nearest_gateways(self):
        for device in self.devices:
            distances = [np.linalg.norm(np.array(device.location) - np.array(gateway.location)) for gateway in self.gateways]
            nearest_gateway_index = np.argmin(distances)
            device.gateway = self.gateways[nearest_gateway_index].id

    def generate_micro_cell_centers(self, macro_center, num_micro_cells):
        return np.random.normal(loc=macro_center, scale=self.sigma, size=(num_micro_cells, 2))

    def display_network(self):
        fig, ax = plt.subplots()
        for device in self.devices:
            ax.plot(device.location[0], device.location[1], 'bo')
        for gateway in self.gateways:
            ax.plot(gateway.location[0], gateway.location[1], 'rs')
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title('Network Topology')
        plt.show()

# Simulation Class
class Simulation:
    global_clock = 0
    max_simulation_time = 100
    task_arrival_prob = 0.1

    def __init__(self):
        self.end_devices: List[EndDevice] = []
        self.gateways: List[Gateway] = []
        self.servers: List[Server] = []
        self.tasks = []
        self.task_id_counter = 0

    def run(self):
        while Simulation.global_clock < Simulation.max_simulation_time:
            self.generate_tasks()
            self.schedule_tasks()
            Simulation.global_clock += 1

    def generate_tasks(self):
        for device in self.end_devices:
            task = device.generate_task(self.task_id_counter, Simulation.task_arrival_prob, 1, 10)
            if task:
                self.tasks.append(task)
                self.task_id_counter += 1

    def schedule_tasks(self):
        for server in self.servers:
            tasks_for_server = [task for task in self.tasks if task.default_server == server.id and task.execution_server is None]
            if tasks_for_server:
                scheduled_tasks = server.schedule(tasks_for_server)
                for task in scheduled_tasks:
                    task.execution_server = server.id

    def add_device(self, device):
        self.end_devices.append(device)

    def add_gateway(self, gateway):
        self.gateways.append(gateway)

    def add_server(self, server):
        self.servers.append(server)


Simulation().run() 
