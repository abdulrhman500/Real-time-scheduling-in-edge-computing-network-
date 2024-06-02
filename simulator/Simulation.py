from simulator.constants import device_prototypes, standard_gateway
from components.EndDevice import EndDevice
from components.Gateway import Gateway
from components.Server import Server
from components.Task import Task
from simulator.networkGenerator import NetworkGenerator 
from typing import List
from simulator.TaskGenerator import TaskGenerator
from simulator.algorithms.scheduling.scheduler import Scheduler
from simulator.algorithms.serverSelection import ServerSelection
# global_clock=0
class Simulation:
    global_clock = 0  

    def __init__(self,  total_tasks: int,network_generator:NetworkGenerator,task_generator:TaskGenerator,server_Selector:ServerSelection):

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
        self.gateways = self.network_generator.gateways
        self.devices = self.network_generator.devices
        self.severs = self.network_generator.servers
        print(f"Gateway: { len(self.gateways)}\nDevices : {len(self.devices)}\n servers : {len(self.severs)}")

    def run(self,mode="time",number=100):
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
            count_generated_task, generated_tasks = self.task_generator.generate_tasks(task_generation_prob=0.1, task_size_min=10, task_size_max=1000)
            # print(generated_tasks,"fvf")
            offloaded = self.offload_tasks(generated_tasks)
            # self.offloaded_tasks.append(offloaded)
            #apply the MCF- algorithm
            #TODO check ths self.offloaded_tasks.append(offloaded)
            Scheduler(simulation_instance=self).apply(offloaded)
            

            
            self.global_clock += 1
        # self.global_clock += 1


    def offload_tasks(self,generated_tasks):
            # print(f"genratd tasks at this time  {len(generated_tasks)}")
            return generated_tasks
            # for device in self.network_generator.devices:
            #     if device.task_queue:
            #         task = device.task_queue.popleft()
            #         # Decision making for offloading or processing locally
            #           if offload => send to the other paper algorithms
            #         # This is where you integrate your DRL algorithm
            