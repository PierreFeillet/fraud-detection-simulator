
import random
import pandas as pd
import numpy as np

from operation import Operation

class BankActivities:
    def __init__(self, max_size):
        self.dtypes = Operation.OPERATION_DTYPE
        self.operation_log = pd.DataFrame(columns=self.dtypes.keys()).astype(self.dtypes)
        self.transactions_buffer = []
        self.max_size = max_size

    def add_operations(self, agent_operations):
        for operation in agent_operations:
            self.process_operation(operation)

    def process_operation(self, operation):
        if len(self.operation_log) == self.max_size:
            return
        self.transactions_buffer.append(operation)

    

    def flush_transactions(self):
        if self.transactions_buffer:
            try:
                new_data = pd.DataFrame(self.transactions_buffer, columns=self.dtypes.keys()).astype(self.dtypes)
                if len(self.operation_log) + len(new_data) > self.max_size:
                    remaining_space = self.max_size - len(self.operation_log)
                    new_data = new_data.iloc[:remaining_space]
                    print("Truncated 'new_data' to fit within max_size:", new_data)
                self.operation_log = pd.concat([self.operation_log, new_data], ignore_index=True)
                self.transactions_buffer = []
            except pd.errors.OutOfBoundsDatetime as e:
                print("Error:", e)
