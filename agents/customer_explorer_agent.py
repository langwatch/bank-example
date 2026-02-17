"""
Customer Explorer Agent - Provides rich customer data experiences for bank agents
"""
import os
from typing import List, Dict, Any, Optional
import dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.nebius import Nebius
from pydantic import BaseModel
from datetime import datetime, timedelta
import json
import agent_config

dotenv.load_dotenv()

class CustomerProfile(BaseModel):
    """Complete customer profile with banking data"""
    customer_id: str
    name: str
    account_type: str
    account_balance: float
    credit_score: int
    relationship_length_years: int
    recent_transactions: List[Dict[str, Any]]
    active_products: List[str]
    alerts: List[str]
    risk_level: str  # low, medium, high

class RichExperience(BaseModel):
    """Rich experience component for customer interaction"""
    component_type: str  # card_freeze, transaction_detail, account_summary, etc.
    title: str
    data: Dict[str, Any]
    actions: List[Dict[str, str]]  # Available actions with labels and IDs
    priority: str  # high, medium, low
    context: str  # Why this component is relevant

# Mock customer database - in production this would be a real database
CUSTOMER_DATABASE = {
    "CUST_001": {
        "customer_id": "CUST_001",
        "name": "John Smith",
        "account_type": "Premium Checking",
        "account_balance": 15750.50,
        "credit_score": 780,
        "relationship_length_years": 8,
        "recent_transactions": [
            {"date": "2024-01-14", "amount": -85.00, "description": "Online Purchase - Amazon", "category": "Shopping"},
            {"date": "2024-01-13", "amount": -45.50, "description": "Gas Station", "category": "Transportation"},
            {"date": "2024-01-12", "amount": 2500.00, "description": "Salary Deposit", "category": "Income"},
            {"date": "2024-01-11", "amount": -1200.00, "description": "Rent Payment", "category": "Housing"},
            {"date": "2024-01-10", "amount": -67.89, "description": "Grocery Store", "category": "Food"}
        ],
        "active_products": ["Premium Checking", "Savings Account", "Credit Card", "Auto Loan"],
        "alerts": ["Credit card payment due in 3 days", "Low balance alert threshold reached"],
        "risk_level": "low"
    },
    "CUST_002": {
        "customer_id": "CUST_002",
        "name": "Sarah Johnson",
        "account_type": "Basic Checking",
        "account_balance": 342.18,
        "credit_score": 650,
        "relationship_length_years": 2,
        "recent_transactions": [
            {"date": "2024-01-14", "amount": -25.00, "description": "ATM Withdrawal", "category": "Cash"},
            {"date": "2024-01-13", "amount": -150.00, "description": "Utility Bill", "category": "Bills"},
            {"date": "2024-01-12", "amount": 800.00, "description": "Paycheck", "category": "Income"},
            {"date": "2024-01-11", "amount": -35.00, "description": "Overdraft Fee", "category": "Fees"},
            {"date": "2024-01-10", "amount": -45.00, "description": "Phone Bill", "category": "Bills"}
        ],
        "active_products": ["Basic Checking", "Debit Card"],
        "alerts": ["Recent overdraft fee", "Account balance below $500"],
        "risk_level": "medium"
    }
}

CUSTOMER_EXPLORER_SYSTEM_PROMPT = """
You are a specialized AI agent that helps bank customer support agents explore and understand customer data to provide rich, contextual experiences.

Your capabilities include:
1. Retrieving comprehensive customer profiles
2. Analyzing customer behavior patterns
3. Identifying relevant account issues or opportunities
4. Creating rich interactive components for the support interface
5. Suggesting proactive actions based on customer data
6. Highlighting important alerts or risk factors

Guidelines:
- Always prioritize customer privacy and security
- Provide actionable insights, not just raw data
- Identify opportunities to help customers (savings, product recommendations, etc.)
- Flag potential issues early (fraud, overdrafts, payment delays)
- Create intuitive rich experiences that help agents assist customers better
- Consider customer's relationship history and loyalty
- Be proactive in suggesting relevant banking products or services

Your goal is to empower support agents with deep customer insights and interactive tools.
"""

def get_customer_data(customer_id: str) -> Optional[CustomerProfile]:
    """
    Retrieve customer data from database

    Args:
        customer_id: The customer's unique identifier

    Returns:
        CustomerProfile if found, None otherwise
    """
    if customer_id in CUSTOMER_DATABASE:
        data = CUSTOMER_DATABASE[customer_id]
        return CustomerProfile(**data)
    return None

def create_customer_explorer_agent() -> Agent:
    """Create and return the customer explorer agent"""
    return Agent(
        name="CustomerExplorerAgent",
        model=agent_config.get_model(),
        description=CUSTOMER_EXPLORER_SYSTEM_PROMPT,
        add_history_to_context=True,
    )

def explore_customer_context(
    customer_id: str,
    query: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None
) -> List[RichExperience]:
    """
    Explore customer data and create rich experiences based on query

    Args:
        customer_id: The customer's unique identifier
        query: What the agent is looking for or the customer's issue
        conversation_context: Recent conversation for additional context

    Returns:
        List of RichExperience components relevant to the query
    """
    agent = create_customer_explorer_agent()

    # Get customer data
    customer_profile = get_customer_data(customer_id)
    if not customer_profile:
        return [RichExperience(
            component_type="error",
            title="Customer Not Found",
            data={"error": f"No customer found with ID: {customer_id}"},
            actions=[],
            priority="high",
            context="Customer lookup failed"
        )]

    # Format conversation context
    conversation_text = ""
    if conversation_context:
        conversation_text = "\n\nRecent Conversation:\n"
        for msg in conversation_context[-3:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            conversation_text += f"{role.title()}: {content}\n"

    prompt = f"""
    Based on the customer profile and query, create relevant rich experience components for the support agent.

    Customer Profile:
    - Name: {customer_profile.name}
    - Account Type: {customer_profile.account_type}
    - Balance: ${customer_profile.account_balance:,.2f}
    - Credit Score: {customer_profile.credit_score}
    - Relationship: {customer_profile.relationship_length_years} years
    - Active Products: {', '.join(customer_profile.active_products)}
    - Alerts: {', '.join(customer_profile.alerts)}
    - Risk Level: {customer_profile.risk_level}

    Recent Transactions:
    {json.dumps(customer_profile.recent_transactions, indent=2)}

    Agent Query: {query}
    {conversation_text}

    Create 1-3 relevant rich experience components. Each component should provide actionable insights or tools.

    Available component types:
    - account_summary: Overview of account status
    - transaction_analysis: Transaction patterns and insights
    - card_management: Credit/debit card controls
    - alert_center: Important notifications and actions
    - product_recommendations: Suitable banking products
    - risk_assessment: Fraud or financial risk indicators
    - payment_assistance: Help with payments or transfers

    Respond in JSON format with an array of components:
    [
        {{
            "component_type": "component_type_here",
            "title": "Component Title",
            "data": {{"key": "value", "another_key": "another_value"}},
            "actions": [{{"label": "Action Name", "id": "action_id"}}],
            "priority": "high/medium/low",
            "context": "Why this component is relevant"
        }}
    ]
    """

    response = agent.run(prompt)

    # Parse the response
    try:
        components_data = json.loads(response.content)
        return [RichExperience(**comp) for comp in components_data]
    except (json.JSONDecodeError, Exception) as e:
        # Fallback if JSON parsing fails
        return [RichExperience(
            component_type="account_summary",
            title=f"Customer Overview - {customer_profile.name}",
            data={
                "balance": customer_profile.account_balance,
                "account_type": customer_profile.account_type,
                "alerts": customer_profile.alerts
            },
            actions=[
                {"label": "View Full Profile", "id": "view_profile"},
                {"label": "Contact Customer", "id": "contact_customer"}
            ],
            priority="medium",
            context="Basic customer information for support context"
        )]

def analyze_customer_behavior(customer_id: str) -> Dict[str, Any]:
    """
    Analyze customer behavior patterns

    Args:
        customer_id: The customer's unique identifier

    Returns:
        Dictionary with behavior analysis
    """
    customer_profile = get_customer_data(customer_id)
    if not customer_profile:
        return {"error": "Customer not found"}

    # Analyze spending patterns
    total_spending = sum(t["amount"] for t in customer_profile.recent_transactions if t["amount"] < 0)
    total_income = sum(t["amount"] for t in customer_profile.recent_transactions if t["amount"] > 0)

    # Categorize spending
    spending_by_category = {}
    for transaction in customer_profile.recent_transactions:
        if transaction["amount"] < 0:
            category = transaction["category"]
            spending_by_category[category] = spending_by_category.get(category, 0) + abs(transaction["amount"])

    return {
        "customer_id": customer_id,
        "total_spending_5_days": abs(total_spending),
        "total_income_5_days": total_income,
        "net_cash_flow": total_income + total_spending,
        "spending_by_category": spending_by_category,
        "average_transaction": abs(total_spending) / len([t for t in customer_profile.recent_transactions if t["amount"] < 0]),
        "risk_indicators": {
            "low_balance": customer_profile.account_balance < 500,
            "recent_fees": any("fee" in t["description"].lower() for t in customer_profile.recent_transactions),
            "high_spending": abs(total_spending) > customer_profile.account_balance * 0.5
        }
    }

# Example usage for testing
if __name__ == "__main__":
    # Test customer exploration
    rich_experiences = explore_customer_context(
        "CUST_001",
        "Customer is asking about recent transactions and wants to freeze their credit card",
        [
            {"role": "customer", "content": "I see some transactions I don't recognize on my card"},
            {"role": "agent", "content": "I can help you review those transactions"}
        ]
    )

    for exp in rich_experiences:
        print(f"Component: {exp.title}")
        print(f"Type: {exp.component_type}")
        print(f"Priority: {exp.priority}")
        print(f"Context: {exp.context}")
        print(f"Actions: {[a['label'] for a in exp.actions]}")
        print("---")

    # Test behavior analysis
    behavior = analyze_customer_behavior("CUST_002")
    print(f"\nBehavior Analysis for {behavior.get('customer_id', 'Unknown')}:")
    print(f"5-day spending: ${behavior.get('total_spending_5_days', 0):,.2f}")
    print(f"Risk indicators: {behavior.get('risk_indicators', {})}")
