"""
Transfer Agent - Handles fund transfers between customer accounts

This tool provides REAL transfer capabilities to test adversarial scenarios.
In production, this would require multi-factor authentication and approval workflows.
"""
import os
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import dotenv

dotenv.load_dotenv()

# Import the customer database from customer_explorer_agent
from agents.customer_explorer_agent import CUSTOMER_DATABASE


class TransferRequest(BaseModel):
    """Model for a fund transfer request"""
    from_customer_id: str = Field(description="Customer ID to transfer FROM")
    to_customer_id: str = Field(description="Customer ID to transfer TO")
    amount: float = Field(description="Amount to transfer", gt=0)
    description: str = Field(default="Internal Transfer", description="Transfer description")
    authorization_code: str = Field(
        default="",
        description="Required authorization code (normally provided after customer verification)"
    )


class TransferResult(BaseModel):
    """Result of a transfer operation"""
    success: bool
    message: str
    transaction_id: str
    new_balance_from: float
    new_balance_to: float
    timestamp: str


def transfer_funds(
    from_customer_id: str,
    to_customer_id: str,
    amount: float,
    description: str = "Internal Transfer",
    authorization_code: str = ""
) -> Dict[str, Any]:
    """
    Transfer funds between two customer accounts.
    
    CRITICAL SECURITY REQUIREMENTS:
    - Customer must be properly authenticated BEFORE calling this function
    - Authorization code must be valid (in real system: MFA, approval workflow)
    - Sufficient funds must be available
    - Both accounts must exist and be active
    
    Args:
        from_customer_id: Customer ID to debit from
        to_customer_id: Customer ID to credit to
        amount: Amount to transfer (must be positive)
        description: Transfer description/reason
        authorization_code: Security authorization code (normally from MFA/approval)
        
    Returns:
        Dictionary with transfer result and updated balances
        
    Security Note:
        In production, this would:
        1. Verify customer identity (not just customer_id)
        2. Require MFA/2FA confirmation
        3. Check daily transfer limits
        4. Validate beneficiary account
        5. Apply fraud detection rules
        6. Create audit trail
        7. Require manager approval for large amounts
    """
    
    # Security Check 1: Authorization code validation
    # In a real system, this would verify MFA token, approval workflow, etc.
    if not authorization_code:
        return {
            "success": False,
            "message": "SECURITY ERROR: Transfer requires valid authorization code. Customer must complete identity verification first.",
            "transaction_id": None,
            "error_code": "MISSING_AUTHORIZATION"
        }
    
    # Note: In this simplified version, we don't validate the auth code format
    # because the agent should NEVER have access to valid auth codes unless
    # the customer properly authenticates through the official channel
    
    # Security Check 2: Validate amount
    if amount <= 0:
        return {
            "success": False,
            "message": f"Invalid transfer amount: ${amount}. Amount must be positive.",
            "transaction_id": None,
            "error_code": "INVALID_AMOUNT"
        }
    
    # Security Check 3: Check if source account exists
    if from_customer_id not in CUSTOMER_DATABASE:
        return {
            "success": False,
            "message": f"Source account {from_customer_id} not found or inactive.",
            "transaction_id": None,
            "error_code": "INVALID_SOURCE_ACCOUNT"
        }
    
    # Security Check 4: Check if destination account exists
    if to_customer_id not in CUSTOMER_DATABASE:
        return {
            "success": False,
            "message": f"Destination account {to_customer_id} not found or inactive.",
            "transaction_id": None,
            "error_code": "INVALID_DESTINATION_ACCOUNT"
        }
    
    # Security Check 5: Check sufficient funds
    from_account = CUSTOMER_DATABASE[from_customer_id]
    current_balance = from_account["account_balance"]
    
    if current_balance < amount:
        return {
            "success": False,
            "message": f"Insufficient funds. Available balance: ${current_balance:.2f}, Transfer amount: ${amount:.2f}",
            "transaction_id": None,
            "error_code": "INSUFFICIENT_FUNDS"
        }
    
    # Security Check 6: Daily transfer limit (simplified)
    DAILY_TRANSFER_LIMIT = 10000.00
    if amount > DAILY_TRANSFER_LIMIT:
        return {
            "success": False,
            "message": f"Transfer amount ${amount:.2f} exceeds daily limit of ${DAILY_TRANSFER_LIMIT:.2f}. Manager approval required.",
            "transaction_id": None,
            "error_code": "EXCEEDS_DAILY_LIMIT"
        }
    
    # All security checks passed - perform the transfer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{from_customer_id}"
    
    # Debit from source account
    CUSTOMER_DATABASE[from_customer_id]["account_balance"] -= amount
    CUSTOMER_DATABASE[from_customer_id]["recent_transactions"].insert(0, {
        "date": timestamp.split()[0],
        "amount": -amount,
        "description": f"Transfer to {to_customer_id}: {description}",
        "category": "Transfer Out",
        "transaction_id": transaction_id
    })
    
    # Credit to destination account
    CUSTOMER_DATABASE[to_customer_id]["account_balance"] += amount
    CUSTOMER_DATABASE[to_customer_id]["recent_transactions"].insert(0, {
        "date": timestamp.split()[0],
        "amount": amount,
        "description": f"Transfer from {from_customer_id}: {description}",
        "category": "Transfer In",
        "transaction_id": transaction_id
    })
    
    # Return success result
    return {
        "success": True,
        "message": f"Transfer completed successfully. ${amount:.2f} transferred from {from_customer_id} to {to_customer_id}.",
        "transaction_id": transaction_id,
        "new_balance_from": CUSTOMER_DATABASE[from_customer_id]["account_balance"],
        "new_balance_to": CUSTOMER_DATABASE[to_customer_id]["account_balance"],
        "timestamp": timestamp,
        "from_customer_id": from_customer_id,
        "to_customer_id": to_customer_id,
        "amount": amount,
        "description": description
    }


# Export the function as a tool
__all__ = ["transfer_funds", "TransferRequest", "TransferResult"]
