import random
from collections import deque
from components.Task import Task


class EndDevice:

    def __init__(self, device_id, gateway, location,comp_capacity, energy_per_cycle, cycle_per_bit, max_waiting_tasks,server):
        self.id = device_id
        self.gateway = gateway
        self.server = server
        self.comp_capacity = comp_capacity
        self.energy_per_cycle = energy_per_cycle
        self.cycle_per_bit = cycle_per_bit
        self.max_waiting_tasks = max_waiting_tasks
        self.location = location
        self.task_queue = deque(maxlen=max_waiting_tasks)  # Use deque for efficient queue

    def get_current_channel_gain(self, channel_id):
        # Implement channel gain calculation based on location, channel, and environment factors
        pass

    def compute_transmit_power(self, channel_gain):
        # Implement transmit power calculation based on channel gain and other factors
        pass

    def compute_energy_per_bit(self):
        return self.cycle_per_bit * self.energy_per_cycle  # Energy per bit computation

    def generate_task(self,id, task_generation_prob, task_size_min, task_size_max):
        from  simulator.Simulation import Simulation
        if random.random() <= task_generation_prob and self.max_waiting_tasks>len(self.task_queue):  # Bernoulli trial
            task_size = random.randint(task_size_min, task_size_max)  # Uniform distribution
            deadline = random.randint(1, 100)
            return Task(task_size ,Simulation.global_clock+deadline,id=id,arrival_time =  Simulation.global_clock,default_server=self.server) 
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
                f"Task Queue: {list(self.task_queue)}\n"  # Convert deque to list for printing
        )
        

