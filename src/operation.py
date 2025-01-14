import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


from catalog import generate_transaction_amount, get_transaction_type, get_activity_type, extract_merchant, is_fraud
from catalog import locations
from catalog import location_weights
from catalog import devices
from catalog import device_weights
from catalog import networks
from catalog import network_weights
from catalog import possible_transactions

from agent import Agent  # Import the Agent class

class Operation(Agent):
    
    # Global dictionary for shared keys
    OPERATION_DTYPE = {
        "agent_id": "int16",
        "timestamp": "datetime64[ns]",
        "action": "category",
        "granted": "bool",
        "initial_balance": "float32",
        "amount": "float32",
        "balance": "float32",
        "merchant": "category",
        "location": "category",
        "device": "category",
        "network": "category",
        "compromised_device": "int8",
        "compromised_network": "int8",
        "fraud": "int8"
    }

    def __init__(self, agent_id, initial_balance, behavior_type):
        """
        Initialize an Operation instance.

        Args:
            agent_id (int): Unique identifier for the agent.
            balance (float): Initial balance of the agent.
            behavior_type (str): Behavior type ("normal", "fraud", etc.).
        """
        # Initialize the parent class (Agent)
        super().__init__(agent_id, initial_balance)

        # Operation-specific attributes
        self.timestamp = datetime.now()
        #self.balance = self.initial_balance
        self.fraud = is_fraud(behavior_type)
        self.action = self.get_operation_type(behavior_type)
        self.granted = True
        self.amount = (
            generate_transaction_amount(self.action, behavior_type)
            if self.action in possible_transactions
            else 0
        )
        self.merchant = (
            extract_merchant(self.action) if self.action in possible_transactions else None
        )
        self.location = np.random.choice(locations, p=location_weights)
        self.device = np.random.choice(devices, p=device_weights)
        self.network = np.random.choice(networks, p=network_weights)
        self.compromised_device = self.is_compromised()
        self.compromised_network = self.is_compromised()

    def moveTime(self):
        self.timestamp += timedelta(seconds=random.randint(1, 30))

    def is_compromised(self, probability=0.05):
        """
        Returns True if a device or network is compromised, based on the probability.
        Default probability of compromise is 5%.
        """
        return random.random() < probability

    def get_operation_type(
        self,
        behavior_type="",
        legitimate_activity_probability=0.2,
        fraud_activity_probability=0.5,
    ):
        """
        Determine the transaction type based on behavior type.

        Args:
            behavior_type (str): Type of behavior (defined in the catalog.py).
            legitimate_activity_probability (float): Probability for legitimate activity.
            fraud_activity_probability (float): Probability for fraudulent activity.

        Returns:
            str: Transaction type.
        """
        if self.fraud==0:
            if random.random() < legitimate_activity_probability:
                operation = get_activity_type(behavior_type)
            else:
                operation = get_transaction_type(behavior_type)
        elif self.fraud==1:
            if random.random() < fraud_activity_probability:
                operation = get_activity_type(behavior_type)
            else:
                operation = get_transaction_type(behavior_type)
        if operation == 'failed_login':
            self.granted = False
        return operation


    def update_balance(self):
        """
        Update the agent's balance based on the transaction type and amount.
        """
        if self.action in possible_transactions:
            self.initial_balance = self.balance

            if self.action in ["withdrawal", "purchase"]:
                if self.balance >= self.amount:
                    self.balance -= self.amount
                else:
                    self.granted = False # Insufficient funds
            elif self.action == "deposit":
                self.balance += self.amount

    def __repr__(self):
        return (
            f"Operation(agent_id={self.agent_id}, type={self.action}, amount={self.amount}, "
            f"balance={self.balance}, fraud={self.fraud}, timestamp={self.timestamp})"
        )


