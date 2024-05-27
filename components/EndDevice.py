import random
from collections import deque
from components import Task
from  simulator.Simulation import Simulation

class EndDevice:

    def __init__(self, device_id, gateway, location,comp_capacity, energy_per_cycle, cycle_per_bit, max_waiting_tasks):
        self.id = device_id
        self.gateway = gateway
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
        
        if random.random() <= task_generation_prob and self.max_waiting_tasks>len(self.task_queue):  # Bernoulli trial
            task_size = random.randint(task_size_min, task_size_max)  # Uniform distribution
            deadline = random.randint(1, 100)
            return Task(task_size ,Simulation.global_clock+deadline, Simulation.global_clock,id) 
        else:
            return None
        

