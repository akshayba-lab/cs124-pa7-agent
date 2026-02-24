"""
Extra Credit 

In this assignment, you will create your own AI agent using DSPy and any tools/APIs you choose.

Your agent should:
1. Use DSPy's framework (signatures, modules, etc.)
2. Implement some interesting functionality beyond basic chat
3. Demonstrate your implementation with a working example

You have complete freedom to:
- Choose what your agent does (data analysis, creative generation, etc.)
- Select which APIs or libraries to integrate
- Design the interface and interaction pattern

Some ideas to get you started:
- A research assistant that uses web search
- A creative writing collaborator with style adaptation
- A data analysis agent that processes files
- A multi-step reasoning agent for complex problems
- Something completely different! Be creative!

Complete the TODOs below to build your agent.
"""

import dspy
import os
from api_keys import TOGETHER_API_KEY

# Configure environment
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# TODO: Add any imports you need for your agent
# Examples:
# from mem0 import Memory
# import requests
# import pandas as pd


# TODO: Add any configuration your agent needs
# Examples:
# - API configurations
# - Model settings
# - Tool configurations


# TODO: Implement any helper classes or functions your agent needs


# TODO: Define DSPy Signature for your agent
# This specifies what your agent takes as input and produces as output
# Include instructions in the docstring about how it should behave
class YourAgentSignature(dspy.Signature):
    """
    TODO: Write instructions for what your agent should do.
    Describe its behavior, capabilities, and how it should respond.
    """
    user_input: str = dspy.InputField()  # TODO: Modify as needed
    response: str = dspy.OutputField()   # TODO: Modify as needed


# TODO: Implement your agent as a DSPy Module
class YourAgent(dspy.Module):
    """TODO: Describe what your agent does."""

    def __init__(self):
        super().__init__()
        
        # TODO: Initialize any components your agent needs
        # Examples:
        # - Tools for ReAct
        # - Memory systems
        # - API clients
        # - Other modules
        
        # TODO: Choose and initialize your DSPy module
        # Options:
        # - dspy.ReAct for tool-using agents
        # - dspy.ChainOfThought for reasoning
        # - dspy.Predict for simple prediction
        
        # Example for ReAct:
        # self.tools = [...]
        # self.agent = dspy.ReAct(
        #     signature=YourAgentSignature,
        #     tools=self.tools,
        #     max_iters=6
        # )

    def forward(self, user_input: str):
        """Process user input and generate a response."""
        # TODO: Implement your agent's logic
        # This should call your DSPy module and return results
        pass


def run_demo():
    """Demonstration of your agent."""
    
    # Configure DSPy
    lm = dspy.LM(model='together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1')
    dspy.configure(lm=lm)

    # TODO: Initialize your agent
    # agent = YourAgent()

    print("🤖 Agent Demo")
    print("=" * 50)

    # TODO: Create test cases that demonstrate your agent's capabilities
    # Show what makes your agent interesting and useful!
    
    test_inputs = [
        # TODO: Add test inputs here
    ]
    
    # TODO: Run your demo
    # for user_input in test_inputs:
    #     print(f"\n📝 User: {user_input}")
    #     response = agent(user_input=user_input)
    #     print(f"🤖 Agent: {response}")


if __name__ == "__main__":
    run_demo()