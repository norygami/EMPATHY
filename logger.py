from datetime import datetime
import csv
import os


class LogData:
    def __init__(self, channel_type, user_channel_id, response_time, prompt_length=None, response_length=None, history_length=None):
        self.timestamp = datetime.now()
        self.channel_type = channel_type
        self.user_channel_id = user_channel_id
        self.response_time = response_time
        self.prompt_length = prompt_length
        self.response_length = response_length
        self.history_length = history_length
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'channel_type': self.channel_type, 
            'user_channel_id': self.user_channel_id,
            'response_time': self.response_time,
            'prompt_length': self.prompt_length,
            'response_length': self.response_length,
            'history_length': self.history_length
        }

def log_performance(log_data: LogData):
    filename = 'logs/bot_performance.csv'
  
    # Get the dictionary representation of log_data
    log_data_dict = log_data.to_dict()

    # Dynamically generate fieldnames from the keys of log_data_dict
    fieldnames = log_data_dict.keys()
    
    # Open the file in append mode
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If the file is new (or empty), write the headers
        if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
            writer.writeheader()

        # Write the performance data
        writer.writerow(log_data_dict)