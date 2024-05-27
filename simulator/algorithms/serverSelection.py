from typing import List, Dict
from components.Task import Task
from components.Server import Server

class ServerSelection:
    def __init__(self, tasks: List[Task], Servers: List[Server]):
        self.tasks = tasks
        self.Servers = Servers

    def select_Servers(self):
        for task in self.tasks:
            valid_Servers = self.get_valid_Servers(task)
            if valid_Servers:
                selected_Server = self.get_min_cost_Server(task, valid_Servers)
                task.server = selected_Server


    def get_valid_Servers(self, task: Task) -> List[Server]:
        valid_Servers = []
        for Server in self.Servers:
            if self.is_Server_valid(task, Server):
                valid_Servers.append(Server)
        return valid_Servers

    def is_Server_valid(self, task: Task, Server: Server) -> bool:
        time_transfer = task.data_size / min(Server.out_bandwidth, Server.in_bandwidth)
        time_total = time_transfer + Server.processing_time(task)
        return time_total <= (task.deadline - task.arrival_time)

    def get_min_cost_Server(self, task: Task, valid_Servers: List[Server]) -> Server:
        min_cost = float('inf')
        selected_Server = None
        for Server in valid_Servers:
            cost = self.calculate_total_cost(task, Server)
            if cost < min_cost:
                min_cost = cost
                selected_Server = Server
        return selected_Server
    #TODO
    def calculate_total_cost(self, task: Task, Server: Server) -> float:
        # Calculate the total cost including transfer and processing
        transfer_cost=0
        if Server.id != task.server_id:
            transfer_cost = task.size * (task.server.uplink_cost+ Server.downlink_cost)
        processing_cost = Server.processing_cost(task)
        return transfer_cost + processing_cost
