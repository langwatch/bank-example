# Bank Customer Support Agent System

A comprehensive AI-powered customer support system for banking services, built with **Agno** and tested with **Scenario**. This project demonstrates advanced multi-agent coordination, tool calling, and end-to-end conversation testing.

## 🎯 Project Overview

This system replicates a real-world bank customer support platform with multiple specialized AI agents working together to provide comprehensive customer service. It showcases:

- **Multi-agent architecture** with specialized agents for different tasks
- **Tool calling correctness** - ensuring the right tools are called at the right time
- **Rich customer experiences** with personalized data insights
- **Comprehensive testing** with Scenario for agent behavior validation

## 🏗️ Architecture

### Main Support Agent
The central coordinator that handles customer interactions and delegates to specialized agents when needed.

### Specialized Agents

1. **Summary Agent** 📊
   - Analyzes conversation threads
   - Provides sentiment analysis
   - Identifies key issues and urgency levels
   - Suggests actions for support teams

2. **Next Message Agent** 💬
   - Suggests appropriate responses using knowledge base
   - Provides confidence levels and reasoning
   - Offers alternative approaches
   - Determines escalation needs

3. **Customer Explorer Agent** 🔍
   - Provides rich customer data experiences
   - Analyzes spending patterns and behavior
   - Generates interactive components for support agents
   - Identifies risk factors and opportunities

## 🚀 Key Features

- **Fraud Detection & Response**: Automatically detects fraud concerns and provides security tools
- **Escalation Management**: Intelligently escalates urgent or complex issues
- **Personalized Experiences**: Uses customer data to provide tailored support
- **Multi-turn Conversations**: Maintains context across complex interactions
- **Rich Analytics**: Provides behavior analysis and risk assessment

## 📋 Requirements

- Python 3.10+
- OpenAI API key
- UV package manager

## 🛠️ Installation

1. **Clone and navigate to the project**:
   ```bash
   cd examples/bank_customer_support
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file with:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGWATCH_API_KEY=your_langwatch_api_key_here  # Optional
   ```

## 🎮 Usage

### Basic Usage

```python
from main_support_agent import start_conversation, continue_conversation

# Start a new conversation
session_id, response = start_conversation(
    "CUST_001",
    "Hi, I'm having trouble with some transactions on my account."
)
print(f"Agent: {response}")

# Continue the conversation
next_response = continue_conversation(
    session_id,
    "I see charges I don't recognize and I'm worried about fraud."
)
print(f"Agent: {next_response}")
```

### Individual Agent Usage

```python
# Summary Agent
from agents.summary_agent import summarize_conversation

messages = [
    {"role": "customer", "content": "I'm frustrated with this issue!", "timestamp": "2024-01-15 10:30:00"},
    {"role": "agent", "content": "I understand and I'm here to help.", "timestamp": "2024-01-15 10:31:00"}
]

summary = summarize_conversation(messages)
print(f"Sentiment: {summary.sentiment}")
print(f"Key Issues: {summary.key_issues}")
```

```python
# Next Message Agent
from agents.next_message_agent import suggest_next_message

suggestion = suggest_next_message(
    "My card was declined at the store",
    [{"role": "customer", "content": "I need help with my card"}]
)
print(f"Suggested: {suggestion.suggested_message}")
print(f"Confidence: {suggestion.confidence_level}")
```

```python
# Customer Explorer Agent
from agents.customer_explorer_agent import explore_customer_context

rich_experiences = explore_customer_context(
    "CUST_001",
    "fraud concern, card security"
)

for exp in rich_experiences:
    print(f"Component: {exp.title}")
    print(f"Actions: {[action['label'] for action in exp.actions]}")
```

## 🧪 Testing

This project uses **Scenario** framework for comprehensive agent testing with realistic simulations:

### Business-Focused Test Structure

1. **Main Support Agent Tests** (`tests/test_main_support_agent.py`)
   - Fraud investigation workflows
   - Complex problem resolution
   - Customer escalation scenarios
   - Account inquiries and data exploration
   - Urgent business issues
   - Card management and security

2. **Summary Agent Tests** (`tests/test_summary_agent.py`)
   - Fraud conversation analysis
   - Escalated conversation patterns
   - Complex problem resolution summaries
   - Positive customer experience analysis
   - Sentiment progression tracking

3. **Customer Explorer Tests** (`tests/test_customer_explorer_agent.py`)
   - Fraud investigation data analysis
   - Spending pattern analysis for budgeting
   - Risk assessment for account protection

4. **Next Message Agent Tests** (`tests/test_next_message_agent.py`)
   - Complex banking issue guidance
   - Escalation recommendations
   - Knowledge base utilization

### Running Tests

```bash
# Run all Scenario tests
uv run python -m pytest tests/ -v

# Run specific agent tests
uv run python -c "
import asyncio
from tests.test_main_support_agent import test_fraud_investigation_workflow
asyncio.run(test_fraud_investigation_workflow())
"

# Run main agent demo
uv run python main_support_agent.py
```

### Key Test Features Using Scenario

1. **Realistic User Simulation**
   ```python
   @pytest.mark.agent_test
   @pytest.mark.asyncio
   async def test_fraud_investigation_workflow():
       result = await scenario.run(
           name="fraud investigation and card security",
           description="Customer discovers unauthorized transactions...",
           agents=[
               BankSupportAgentAdapter(),
               scenario.UserSimulatorAgent(),
               scenario.JudgeAgent(criteria=[...])
           ],
           script=[
               scenario.user("I think my card was stolen..."),
               scenario.agent(),
               scenario.judge(),
           ],
       )
       assert result.success
   ```

2. **Automated Quality Assessment**
   - UserSimulatorAgent generates realistic customer responses
   - JudgeAgent evaluates conversations against business criteria
   - Tests validate both tool calling and conversation quality

## 📊 Example Conversations

### Fraud Investigation
```
Customer: "I see transactions I don't recognize. I'm worried about fraud."
Agent: [Calls explore_customer_data tool]
Agent: "I've analyzed your account and prepared card management options.
       You can freeze your card immediately..."
```

### Complex Issue Resolution
```
Customer: "I've been trying to resolve this for weeks and I'm frustrated!"
Agent: [Calls get_next_message_suggestion tool]
Agent: "I sincerely apologize for the ongoing difficulties.
       Let me get the best guidance to resolve this comprehensively..."
```

### Account Analysis
```
Customer: "Can you help me understand my spending patterns?"
Agent: [Calls explore_customer_data tool]
Agent: "I've analyzed your recent transactions and spending behavior.
       Here are personalized insights and recommendations..."
```

## 🎯 Demonstration Features

This project specifically demonstrates the capabilities mentioned in customer requirements:

### ✅ Tool Calling Validation
- **Fraud Detection**: Customer mentions unauthorized transactions → Agent calls `explore_customer_data`
- **Knowledge Base**: Complex issues → Agent calls `get_next_message_suggestion`
- **Conversation Analysis**: Multi-turn discussions → Agent calls `get_conversation_summary`
- **Escalation**: Urgent/angry customers → Agent calls `escalate_to_human`

### ✅ Multi-Agent Coordination
- Main agent coordinates with 3 specialized agents
- Each agent has distinct responsibilities and expertise
- Tools abstract the sub-agent complexity (as requested)

### ✅ Rich Customer Experiences
- Card management interfaces
- Transaction analysis components
- Account overview dashboards
- Risk assessment displays

### ✅ Quality Assurance
- Comprehensive test coverage
- Response quality evaluation
- Sentiment tracking
- Escalation pattern analysis

## 📁 Project Structure

```
bank_customer_support/
├── agents/
│   ├── __init__.py
│   ├── summary_agent.py           # Conversation analysis & sentiment
│   ├── next_message_agent.py      # Response suggestions & knowledge base
│   └── customer_explorer_agent.py # Customer data & rich experiences
├── tests/
│   ├── __init__.py
│   ├── test_main_support_agent.py # Scenario tests for main agent
│   ├── test_summary_agent.py      # Unit tests for summary agent
│   ├── test_next_message_agent.py # Unit tests for next message agent
│   ├── test_customer_explorer_agent.py # Unit tests for explorer agent
│   ├── test_integration.py        # Integration tests
│   └── test_evaluations.py        # Quality evaluation tests
├── main_support_agent.py          # Main coordinator agent
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Customer Data
Mock customer data is defined in `agents/customer_explorer_agent.py`. In production, this would connect to real banking systems.

### Knowledge Base
Banking knowledge base is in `agents/next_message_agent.py`. This includes common issues and solutions for:
- Login problems
- Card issues
- Account balance inquiries
- Transfer problems

### LLM Settings
All agents use GPT-4o-mini by default. Change the model in each agent's `create_*_agent()` function.

## 🚀 Next Steps

1. **Enhanced Mock Data**: Add more diverse customer scenarios
2. **Error Handling**: Implement comprehensive error recovery
3. **Performance Testing**: Add load testing for multiple concurrent sessions
4. **Advanced Evaluations**: Implement more sophisticated quality metrics
5. **Real Integration**: Connect to actual banking APIs and databases

## 🤝 Contributing

This is a demonstration project for Scenario's agent testing capabilities. The architecture and patterns shown here can be adapted for production banking systems.

## 📝 License

This project is for demonstration purposes and showcases the integration between Agno (for agent development) and Scenario (for agent testing).
