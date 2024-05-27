from constants import device_prototypes, standard_gateway
from components.EndDevice import EndDevice
from components.Gateway import Gateway
from components.Server import Server
from components.Task import Task
import networkGenerator
from typing import List

class TaskGenerator:
    next_available_id = 0
    def __init__(self,devices:List[EndDevice],gateways:List[Gateway]):
        self.devices = devices
        self.gateways = gateways
        


    def generate_tasks(self, task_generation_prob=0.3, task_size_min=1, task_size_max=1000):
        count_generated_task = 0
        for device in self.devices:
            #TODO channel_gain ???? Task is not completed structure 
            task = device.generate_task(self.next_available_id,task_generation_prob, task_size_min, task_size_max)
            if task:
                self.next_available_id+=1
                device.task_queue.append(task)
                count_generated_task+=1
        return count_generated_task