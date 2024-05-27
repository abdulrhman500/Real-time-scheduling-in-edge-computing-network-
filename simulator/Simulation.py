from constants import device_prototypes, standard_gateway
from components.EndDevice import EndDevice
from components.Gateway import Gateway
from components.Server import Server
from components.Task import Task
import networkGenerator
from typing import List
import TaskGenerator
from  algorithms.scheduling.scheduler import Scheduler
class Simulation:
    global_ck = 0  

    def __init__(self,  total_tasks: int,network_generator:networkGenerator,task_generator:TaskGenerator):

        self.network_generator = network_generator
        self.task_generator = task_generator
        self.total_tasks = total_tasks
        self.current_generated_tasks = 0
        self.current_completed_tasks = 0
        self.offloaded_tasks=[]
        # self.gateways = gateways
        # self.devices = devices
   
   
    def init_network(self):
        self.network_generator.generate_pcp_network()


    def run(self,mode="task",number=100):
        if mode == 'task':
            self.run_till_task(number)
        else:
            self.run_till_time(number)


    def run_till_task(self,num_tasks):
        done_tasks =0 
        generated_tasks = 0 
        while(done_tasks<=num_tasks):
            if(generated_tasks<num_tasks):
                current_generated_task =  self.task_generator.generate_tasks(task_generation_prob=0.1, task_size_min=10, task_size_max=1000)
                generated_tasks+=current_generated_task
            current_done_tasks = self.offload_tasks()
            done_tasks+=current_done_tasks
            self.global_clock += 1
        

    def run_till_time(self,num_iterations):
        for _ in range(num_iterations):
            self.task_generator.generate_tasks(task_generation_prob=0.1, task_size_min=10, task_size_max=1000)
            offloaded = self.offload_tasks()
            self.offloaded_tasks.append(offloaded)
            #apply the MCF- algorithm
            Scheduler.apply(self.offloaded_tasks)
            

            
            self.global_clock += 1
        # self.global_clock += 1


    def offload_tasks(self):
            return 1
            # for device in self.network_generator.devices:
            #     if device.task_queue:
            #         task = device.task_queue.popleft()
            #         # Decision making for offloading or processing locally
            #           if offload => send to the other paper algorithms
            #         # This is where you integrate your DRL algorithm
            pass