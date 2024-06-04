import random
from collections import deque

# Task Class
class Task:
    #TODO the server assignment
    #TODO review parameter
    def __init__(self, size:int, deadline, starting_time=None,arrival_time=None, server=None, channel_gain=None,id=None,num_instruction_to_be_executed=None,default_server=None):
        self.size = size               
        self.deadline = deadline       
        self.starting_time = starting_time   #start_execution
        self.arrival_time = arrival_time     # Initialized when offloaded 
        self.default_server = default_server
        self.execution_server = server       
        self.channel_gain = channel_gain  
        self.cost = 0
        self.id = id
        self.flow = 0
        self.end_time = deadline
        #TODO this should be estimated using a function that takes task size and give estimated number num_instruction_to_be_executed
        self.num_instruction_to_be_executed = num_instruction_to_be_executed or self.compute_number_of_instruction()


    def compute_number_of_instruction(self):
            return random.randint(1000,10000)*self.size
        

    def __str__(self):
         return (f"Task_ID: {self.id}\n"
                f"Size: {self.size}\n"
                f"Deadline: {self.deadline}\n"
                f"Arrival Time: {self.arrival_time}\n"
                f"Starting Time: {self.starting_time}\n"
                f"Execution Server: {self.execution_server}\n"
                f"Channel Gain: {self.channel_gain}\n"
                f"num_Instructions: {self.num_instruction_to_be_executed}\n"
        )