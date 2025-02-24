
import unittest
import pandas as pd
from simulator import LegitimateCustomer, Fraudster, BankWithClientActivities, run_simulation_with_activities
class TestFraudDetectionSimulator(unittest.TestCase):
    
    def test_legitimate_customer_transaction(self):
        agent = LegitimateCustomer(real_id=1, balance=1000)
        transaction = agent.generate_transaction()
        self.assertIn(transaction['activity_type'], ['deposit', 'withdrawal'])
        self.assertGreaterEqual(agent.balance, 0)

    def test_fraudster_commit_fraud(self):
        agent = Fraudster(real_id=3, balance=5000)
        fraud = agent.commit_fraud()
        self.assertEqual(fraud['activity_type'], 'fraud')
        self.assertLess(agent.balance, 5000)  # Balance should decrease due to fraud
    
    def test_simulation_runs(self):
        legitimate_agents = [LegitimateCustomer(real_id=i, balance=1000) for i in range(3)]
        fraudster_agents = [Fraudster(real_id=i+3, balance=2000) for i in range(1)]
        bank = BankWithClientActivities()
        run_simulation_with_activities(legitimate_agents, fraudster_agents, bank, steps=10)
        self.assertGreater(len(bank.transaction_log), 0)  # Ensure transactions were logged

if __name__ == "__main__":
    unittest.main()
