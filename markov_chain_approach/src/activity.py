import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from catalog import devices
from catalog import device_weights
from catalog import networks
from catalog import network_weights

from agent import Agent  # Import the Agent class

class Activity(Agent):
    
    # Global dictionary for shared keys
    ACTIVITY_DTYPES = {
        "real_id": "int16",
        "virtual_id": "int16",
        "timestamp": "datetime64[ns]",
        "delta_time": "timedelta64[ns]",
        "behavior": "category",
        "initial_balance": "float32",
        "activity_type": "category",
        "granted": "bool",
        "amount": "float32",
        "balance": "float32",
        #"merchant": "category",
        "initial_country": "category",
        "location": "category",
        "device": "category",
        "network": "category",
        #"compromised_device": "int8",
        #"compromised_network": "int8",
        "agent_type": "category",
        "is_fraud": "int8"
    }

    def __init__(self, agent,):
        # Initialize the parent class (Agent)
        super().__init__(agent.real_id, agent.virtual_id, agent.is_fraud, agent.behavior, agent.initial_time)
        # activity-specific attributes
        self.timestamp = agent.initial_time
        self.delta_time = 0
        self.activity_type = ''
        #self.balance = self.initial_balance
        self.granted = True 
        self.amount = 0
       #self.merchant =  extract_merchant(self.activity_type) if self.amount !=0 else ''
        self.device = np.random.choice(devices, p=device_weights)
        self.network = np.random.choice(networks, p=network_weights)
       #self.compromised_device = self.is_compromised()
       # self.compromised_network = self.is_compromised()
        self.initial_country = agent.initial_country
        self.agent_type= agent.agent_type
        self.visited_countries = agent.visited_countries
        self.update_location_probabilities()
        # Assign the transaction location for this activity
        self.location = self.assign_transaction_location()
    
    
    def update_location_probabilities(self):
        """Adjusts probabilities based on agent type.
        Over time, a traveler's probability distribution shifts, reducing the likelihood of transactions in their 
        initial country while increasing the probability of transacting abroad.
        This ensures that frequent travelers are more likely to have transactions spread across multiple countries."""
        if self.agent_type == "traveler":
            # Increase probability of foreign transactions
            for country in self.visited_countries:
                self.visited_countries[country] *= 0.9  # Decay home probability
            foreign_countries = ["France", "UK", "USA", "Germany", "Japan"]
            for country in foreign_countries:
                self.visited_countries[country] = self.visited_countries.get(country, 0) + 0.1

        elif self.agent_type == "static":
            # Mostly transacts in the home country, rarely elsewhere
            self.visited_countries[self.initial_country] = 0.95
            self.visited_countries[random.choice(["France", "UK", "USA"]) if random.random() < 0.05 else self.initial_country] = 0.05

    def assign_transaction_location(self):
        """Assigns a location based on the agent type and behavior."""
        if self.is_fraud:
            if random.random() < 0.7:
                return random.choice(["Switzerland", "Cayman Islands", "Hong Kong", "Singapore"])
            else:
                return random.choice(["USA", "UK", "France", "Germany", "Canada"])
        else:
            # Assign based on probability distribution
            return random.choices(list(self.visited_countries.keys()), weights=self.visited_countries.values())[0]

    #def update_location_probabilities(self):
    #    """Updates the location probabilities for the activity based on the agent's type."""
    #    # You can call the Agent's method to update the probabilities
    #    super().update_location_probabilities()
#
    #def assign_transaction_location(self):
    #    """Assigns a location based on the agent's type and behavior."""
    #    # Call the Agent's method to assign the location
    #    return super().assign_transaction_location()

    def is_compromised(self, probability=0.05):
        """
        Returns True if a device or network is compromised, based on the probability.
        Default probability of compromise is 5%.
        """
        return random.random() < probability

    def update_balance(self):
        """
        Update the agent's balance based on the transaction activity_type and amount.
        """
        if self.amount>=0 or (self.amount<0 and self.balance>=abs(self.amount)):
            self.balance = self.balance + self.amount
        else:
            self.granted = False # Insufficient funds

    '''
    def extract_merchant(transaction):
    """
    Extract a random merchant based on the transaction activity_type and probabilities.
    
    Args:
        transaction (str): The transaction activity_type (e.g., 'purchase', 'deposit', 'withdrawal').
        
    Returns:
        str: The selected merchant.
    """
    if transaction not in TRANSACTION_activity_type_MERCHANTS:
        raise ValueError(f"Transaction activity_type '{transaction}' is not recognized.")
    merchants, probabilities = zip(*TRANSACTION_activity_type_MERCHANTS[transaction]) # Unzip the dictionary for the given transaction to get an array of merchants and an array of relative probabilities
    merchant = np.random.choice(merchants, p=probabilities)
    return merchant
    '''