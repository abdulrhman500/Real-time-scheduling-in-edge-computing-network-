import random
from collections import deque

class Gateway:
    def __init__(self, gateway_id, num_channels,channel_bandwidths,location):
        self.id = gateway_id
        self.num_channels = num_channels
        self.available_channels = set(range(num_channels))
        self.channel_bandwidths = channel_bandwidths
        self.location=location
