from typing import List,Tuple
from components.Task import Task
from components.Gateway import Gateway
from components.Server import Server
from simulator.algorithms.serverSelection import ServerSelection

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


    # def find_most_critical_interval(self,flows):
    #     max_time_counter = max(flow.deadline for flow in flows) - self.simulation_instance.global_clock
    #     interval_length = 2

    #     most_critical_intensity = 0
    #     most_critical_interval = None
    #     most_critical_server = None
    #     # server_flow ={}  # server will transmit that flow

    #     server_flow = {server.id: [] for server in self.servers}
    #     for flow in flows:
    #         if flow.default_server.id != flow.execution_server:
    #             server_flow[flow.default_server.id].append(flow)


    #     for server in self.simulation_instance.severs:
    #         for start_time_counter in range(0,max_time_counter,interval_length):
    #             start_time = start_time_counter + self.simulation_instance.global_clock
    #             end_time = start_time + interval_length

    #             curr_intensity= self.calculate_intensity(server_flow.get(server.id,[]),(start_time,end_time))
    #             if curr_intensity>most_critical_intensity:
    #                 most_critical_intensity = curr_intensity
    #                 most_critical_interval=(start_time,end_time)
    #                 most_critical_server = server

    #     if most_critical_interval is None:
    #         return None,None,None,None

    #     return most_critical_interval, most_critical_server,server_flow.get(most_critical_server.id,[]),most_critical_intensity
        
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
        server_Selector  = ServerSelection(self.servers)
        server_Selector.select_servers(tasks)
      

    def schedule_for_computation(self,tasks:List[Task]):
        for task in tasks:
            task.execution_server.schedule([task])


    def apply_ASCO(self,tasks):
        self.select_server_for_tasks(tasks)
        self.schedule_for_computation(tasks)
        self.normalize_flows(tasks)
        self.control_admission(tasks)
        
        #TODO next phase
        
        pass


    def apply(self,tasks:List[Task],type="ASCO"):
        # print(f"tasks {tasks} f")
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
