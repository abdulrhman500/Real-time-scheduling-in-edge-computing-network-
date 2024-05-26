import random
from collections import deque

# Task Class
class Task:
    def __init__(self, size, deadline, starting_time, server_id=None, channel_gain=None):
        self.size = size               
        self.deadline = deadline       
        self.starting_time = starting_time   
        self.arriving_time = None     # Initialized when offloaded 
        self.server_id = server_id        
        self.channel_gain = channel_gain  