from datetime import timedelta, timezone
import secrets
from dateutil.parser import isoparse  # Install python-dateutil if needed

def generate_random_hash(length=8):
    """Generates a random hexadecimal string of the given length."""
    return secrets.token_hex(length // 2)

def format_timestamp(time):
    parsed_time = isoparse(time)
    st_time = parsed_time.isoformat(timespec='seconds') 
    return st_time

# Assuming in case of transfer, the receipient is not a bank's client. ONly the account id is client
def update_balance(tx):
    """
    Computes the new balance based on the activity type and amount.
    For a deposit, adds the amount.
    For a withdrawal, transfer, purchase, or sale, subtracts the amount.
    """
    tx_type = tx.get("type", "").lower()
    amount = tx.get("amount", 0)
    tx["granted"] = True  # Default to True if missing
    current_balance = tx["balance_before"]
    if any(word in tx_type for word in ["deposit", "contribution", "transfer in", "sale"]):
        return round(current_balance + amount,2)
    elif any(word in tx_type for word in ["withdrawal", "transfer out", "purchase"]):
        new_balance = round(current_balance - amount,2)
        if new_balance<0:
            tx["granted"] = False
            return current_balance
        else:
            tx["granted"] = True
            return new_balance
    else:
        return current_balance

'''  
 # Function to track balance for transfer when accounts are bank's client    
def update_balance_for_account(tx, user_accounts):
    """
    Updates the account balance for a given transaction.
    
    - Deducts money from the sender for purchases, withdrawals, and transfers.
    - Credits the recipient if they are an internal user.
    - For external recipients (not in user_accounts), the money is deducted but not tracked.
    
    Args:
        tx (dict): The transaction data.
        user_accounts (dict): Dictionary mapping account IDs to their current balances.

    Returns:
        float: Updated balance for the sender's account.
    """

    sender_id = tx["account_id"]
    recipient_id = tx.get("recipient_id")  # Can be None for non-transfer transactions
    amount = tx.get("amount", 0)
    tx_type = tx.get("type", "").lower()
    
    # Ensure `granted` always exists in the transaction dictionary
    if "granted" not in tx:
        tx["granted"] = True  # Default to True if missing

    # Ensure sender exists in user accounts
    if sender_id not in user_accounts:
        raise ValueError(f"Error: Sender account {sender_id} not found!")

    # Get sender's current balance
    sender_balance = user_accounts[sender_id]

    if any(word in tx_type for word in ["Transfer Out":
        # Ensure sender has enough balance before proceeding
        if sender_balance >= amount:
            user_accounts[sender_id] -= amount  # Deduct from sender
            
            #  If recipient is internal, update their balance
            if recipient_id and recipient_id in user_accounts:
                user_accounts[recipient_id] += amount
            else:
                print(f" Transfer Out: {amount:.2f} deducted from {sender_id}, but recipient {recipient_id} is external (not tracked).")

        else:
            tx["granted"] = False  # Decline transaction
            return sender_balance  # No change

    elif  tx_type== "Transfer In":
        #  Only credit recipient if they are an internal user
        if recipient_id and recipient_id in user_accounts:
            user_accounts[recipient_id] += amount
        else:
            print(f" Transfer In: {amount:.2f} attempted for {recipient_id}, but account is external. No tracking.")

    elif any(word in tx_type for word in ["deposit", "contribution", "transfer in"]):
        if sender_balance >= amount:
            user_accounts[sender_id] -= amount  # Deduct for purchases/withdrawals
        else:
            tx["granted"] = False  # Decline transaction if insufficient funds
            return sender_balance  # No change

    elif tx_type == "Deposit":
        user_accounts[sender_id] += amount  # Add funds for deposits

    # Return updated sender balance
    return user_accounts[sender_id]
'''