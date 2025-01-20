
import random
import pandas as pd
import numpy as np

from activity import Activity

class BankActivities:
    def __init__(self, max_size):
        self.dtypes = Activity.activity_DTYPE
        self.activity_log = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)
        self.buffer = []
        self.max_size = max_size

    def add_activity(self, activity):
        if len(self.activity_log) == self.max_size:
            return
        self.buffer.append(vars(activity))
        
    def flush_activities(self):
        if self.buffer:
            try:
                new_data = pd.DataFrame(self.buffer, columns=self.dtypes.keys()).astype(self.dtypes)
                if len(self.activity_log) + len(new_data) > self.max_size:
                    remaining_space = self.max_size - len(self.activity_log)
                    new_data = new_data.iloc[:remaining_space]
                    print("Truncated 'new_data' to fit within max_size:", new_data)
                self.activity_log = pd.concat([self.activity_log, new_data], ignore_index=True)
                self.buffer = []
            except pd.errors.OutOfBoundsDatetime as e:
                print("Error:", e)
