# FourRooms Simulation Project

This Python project simulates an agent navigating through a "FourRooms" grid environment. The project includes multiple scenarios, each representing different challenges and behaviors like deterministic and stochastic actions. The project is designed to help study and demonstrate reinforcement learning techniques.

## Project Structure

The project consists of several Python scripts, one for each scenario, and a shared `FourRooms.py` module that defines the environment.

- `Scenario1.py`: Handles the simplest case with a single package collection.
- `Scenario2.py`: Involves multiple package collections without order.
- `Scenario3.py`: Extends scenario 2 by requiring packages to be collected in a specific order.
- `-Stochiastic flag`: Introduces stochastic actions where the agent's intended moves might not go as planned.
- `FourRooms.py`: The environment module used by all scenarios.
- `Makefile`: Simplifies the setup and running of simulations.

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

Create and activate a virtual environment (optional but recommended):

	python -m venv venv
	source venv/bin/activate  # On Unix or MacOS
	venv\Scripts\activate  # On Windows

Install the required packages:

	pip install -r requirements.txt

Usage
You can run each scenario using the Makefile commands provided. Here are some examples:

Run Scenario 1 (Deterministic):

	make run1

Run Scenario 1 (Stochastic):
	
	make run1-stochastic

Running All Scenarios

	make all


Cleaning Up
To clean up the environment and remove all Python bytecode compiled files:

	make clean


