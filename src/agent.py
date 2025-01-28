import random
import pandas as pd
import numpy as np
import catalog


# Include the legitimate customer, fraudster, and bank logic with merchants, locations, devices, etc.

class Agent:
    def __init__(self, real_id, virtual_id, is_fraudster, behavior,):
        self.real_id = real_id # Agent Real Identity 
        self.virtual_id = virtual_id # How the Agent look like to the system 
        self.is_fraudster = is_fraudster
        self.behavior = behavior
        self.initial_balance = round(random.uniform(1000, 100000), 2)
        self.balance = self.initial_balance
        self.initial_country = np.random.choice(catalog.locations, p=catalog.location_weights)
        #self.agent_type = agent_type

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

    def assign_transaction_location(self, behavior_type):
        """Assigns a location based on the agent type and behavior."""
        if behavior_type == "fraudulent":
            if random.random() < 0.7:
                return random.choice(["Switzerland", "Cayman Islands", "Hong Kong", "Singapore"])
            else:
                return random.choice(["USA", "UK", "France", "Germany", "Canada"])
        else:
            # Assign based on probability distribution
            return random.choices(list(self.visited_countries.keys()), weights=self.visited_countries.values())[0]