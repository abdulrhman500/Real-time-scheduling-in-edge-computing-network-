from typing import List
from components.Task import Task
from components.Server import Server

class ServerSelection:
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
                selected_server = self.get_min_cost_server(task, valid_servers)
                task.execution_server = selected_server

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
        return selected_server

    def calculate_total_cost(self, task: Task, server: Server) -> float:
        """
        Calculate the total cost of executing the task on the given server, including data transfer and processing costs.
        """
        transfer_cost = 0
        if server.id != task.default_server.id:
            transfer_cost = task.size * (task.default_server.uplink_cost + server.downlink_cost)
        processing_cost = server.processing_cost(task)
        return transfer_cost + processing_cost
