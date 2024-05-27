import Task
class Server:
    # Thermal Design Power (TDP)
    def __init__(self, server_id, gateway, freq, TDP ,IPC ,uplink_bandwidth, downlink_bandwidth,uplink_cost,downlink_cost):
        self.id = server_id
        self.IPC = IPC  #instruction per cycle
        self.IPS = self.IPC * freq # instruction per sec
        self.EPI = TDP*1/self.IPS  # energy per instruction  
        # self.remaining_IPS = self.IPS
        self.gateway = gateway
        self.freq = freq  # Number of CPU cycles per second (frequency)
        self.instruction_size = 2
        self.uplink_bandwidth = uplink_bandwidth
        self.downlink_bandwidth = downlink_bandwidth
        self.uplink_cost = uplink_cost      # per bit
        self.downlink_cost = downlink_cost  # per bit
        # self.tasks_running = []
       
         
        
    def processing_cost(self,task:Task):
        cost = self.EPI * task.num_instruction_to_be_executed
        return cost
    
    def processing_time(self, task:Task):
        return task.num_instruction_to_be_executed / self.IPS