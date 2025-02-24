import numpy as np
import matplotlib.pyplot as plt

class TransactionDistributions:
    def __init__(self):
        self.legit_amount_distribution = None
        self.fraud_amount_distribution = None

    def generate(self):
        """
        Generate the log-normal distributions for transactions.
        This function defines two log-norm distributions (parametrized by sigma, mu) that will be used to extract the transaction amount.
        μ (mean of log values) – Controls the central tendency.
        σ (standard deviation of log values) – Controls how spread-out the values are.
        For example:
        Small everyday transactions: LogNorm(μ=5, σ=1) → Peaks around $150
        Fraudulent high-value transactions: LogNorm(μ=8, σ=1.2) → Peaks around $3,000
        """
        self.legit_amount_distribution = np.random.lognormal(mean=5, sigma=1, size=10000)
        self.fraud_amount_distribution = np.random.lognormal(mean=8, sigma=1.2, size=10000)

    def plot_distributions(self, output_folder='.'):
        # Create subplots to display the distributions
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.legit_amount_distribution, bins=100, alpha=0.7, label='Legitimate Small Transactions', color='blue', density=True, range=(0,2000))
        plt.xlabel("Transaction Amount ($)")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Legitimate Small Transactions")

        plt.subplot(1, 2, 2)
        plt.hist(self.fraud_amount_distribution, bins=100, alpha=0.7, label='Fraudulent Large Transactions', color='red', density=True, range=(0,40000))
        plt.xlabel("Transaction Amount ($)")
        plt.ylabel("Density")
        plt.legend()
        plt.title("Fraudulent Large Transactions")

        plt.tight_layout()
        plt.savefig(f'{output_folder}/transaction_distributions.png', dpi=300)
        #plt.show()
