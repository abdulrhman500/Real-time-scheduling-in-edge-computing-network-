from typing import List
from components.Task import Task
from components.Gateway import Gateway
from components.Server import Server

class AdmissionControl:
    def __init__(self, tasks: List[Task], gateways: List[Gateway], servers: List[Server]):
        self.tasks = tasks
        self.gateways = gateways
        self.servers = servers

    def perform_admission_control(self):
        feasible_tasks = []
        for task in self.tasks:
            if self.is_task_feasible(task):
                feasible_tasks.append(task)
        return feasible_tasks

    def is_task_feasible(self, task: Task) -> bool:
        # Check if task can be scheduled within its deadline and resources are available
        for server in self.servers:
            if server.comp_capacity >= task.size and server.energy_per_cycle * task.size <= server.energy_per_cycle:
                return True
        return False
