
import random
import pandas as pd
import numpy as np
from datetime import datetime
from activity import Activity

class BankActivities:
    def __init__(self, max_size):
        self.dtypes = Activity.ACTIVITY_DTYPES
        self.activity_log = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)
        self.buffer = []
        self.max_size = max_size
        #self.last_activity_time = datetime.now()

    def add_activity(self, activity):
        # Add activity to buffer, flush if buffer exceeds flush_interval
        if len(self.buffer) < self.max_size:
            self.buffer.append(vars(activity))

    def flush_activities(self):
        if self.buffer:
            try:
                # Create DataFrame from the buffer
                new_data = pd.DataFrame(self.buffer, columns=self.dtypes.keys()).astype(self.dtypes)
                if len(self.activity_log) + len(new_data) > self.max_size:
                    # Trim if the new data exceeds max_size
                    remaining_space = self.max_size - len(self.activity_log)
                    new_data = new_data.iloc[:remaining_space]
                    print("Truncated activity log to fit within max_size")

                # Append to the activity log
                self.activity_log = pd.concat([self.activity_log, new_data], ignore_index=True)
                self.buffer = []  # Clear the buffer after flush
            except pd.errors.OutOfBoundsDatetime as e:
                print("Error:", e)

