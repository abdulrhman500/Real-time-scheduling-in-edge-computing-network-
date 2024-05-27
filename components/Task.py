import random
from collections import deque

# Task Class
class Task:
    #TODO the server assignment
    #TODO review parameter
    def __init__(self, size, deadline, starting_time, server=None, channel_gain=None,id=None,num_instruction_to_be_executed=None):
        self.size = size               
        self.deadline = deadline       
        self.starting_time = starting_time   
        self.arrival_time = None     # Initialized when offloaded 
        self.server = server        
        self.channel_gain = channel_gain  
        self.id = id
        #TODO this should be estimated using a function that takes task size and give estimated number num_instruction_to_be_executed
        self.num_instruction_to_be_executed = num_instruction_to_be_executed 