
import random
import pandas as pd
import numpy as np

from operation import Operation
from agent import Agent  # Make sure the Agent class is accessible


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
                clean_buffer = []
                for txn in self.transactions_buffer:
                    # Check if txn is an instance of Agent
                    if isinstance(txn, Agent):
                        txn_dict = {
                            "agent_id": txn.agent_id,
                            "timestamp": txn.timestamp,
                            "activity": txn.activity,
                            "granted": txn.granted,
                            "initial_balance": txn.initial_balance,
                            "amount": txn.amount,
                            "balance": txn.balance,
                            "location": txn.location,
                            "device": txn.device,
                            "network": txn.network,
                            "compromised_device": txn.compromised_device,
                            "compromised_network": txn.compromised_network
                        }
                        clean_buffer.append(txn_dict)
                    else:
                        clean_buffer.append(txn)

                new_data = pd.DataFrame(clean_buffer, columns=self.dtypes.keys()).astype(self.dtypes)

                # Ensure 'timestamp' is in the correct format
                new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], errors='coerce')

                # Drop rows with invalid timestamps (if any)
                new_data = new_data.dropna(subset=['timestamp'])

                # Handle max size constraints
                if len(self.operation_log) + len(new_data) > self.max_size:
                    remaining_space = self.max_size - len(self.operation_log)
                    new_data = new_data.iloc[:remaining_space]
                    print("Truncated 'new_data' to fit within max_size:", new_data)

                # Update operation_log and clear the buffer
                self.operation_log = pd.concat([self.operation_log, new_data], ignore_index=True)
                self.transactions_buffer = []

            except Exception as e:
                print("Unexpected error:", e)

