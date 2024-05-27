from typing import List
from components.Task import Task
from components.Gateway import Gateway
from components.Server import Server

class Scheduler:
    def __init__(self, tasks: List[Task], gateways: List[Gateway], servers: List[Server]):
        self.tasks = tasks
        self.gateways = gateways
        self.servers = servers

    def schedule_tasks(self):
        # Sort tasks by deadline
        self.tasks.sort(key=lambda task: task.deadline)
        for task in self.tasks:
            self.schedule_task(task)

    def schedule_task(self, task: Task):
        # Find the most suitable server based on MCF-EDF
        best_server = None
        for server in self.servers:
            if server.comp_capacity >= task.size:
                best_server = server
                break
        if best_server:
            self.assign_task_to_server(task, best_server)

    def assign_task_to_server(self, task: Task, server: Server):
        # Assign the task to the server and update server's capacity
        task.server_id = server.id
        server.comp_capacity -= task.size
    def select_server_for_tasks(self,tasks):
        
        pass
    def apply_ASCO(self,tasks):
        
        pass


    def apply(self,tasks:List[Task],type="ASCO"):
        if type == "ASCO":
            self.apply_ASCO(tasks)
        
