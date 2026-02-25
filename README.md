# CDS524 Assignment 1 - Q-Learning Snake Game
This repository contains the complete implementation of a Q-Learning based AI-controlled Snake game for CDS524 Reinforcement Learning course. The project fully meets all assignment requirements, including game design, Q-Learning algorithm implementation, Pygame-based UI, and mouse-only interaction.

## Project Overview
The game is a classic grid-based Snake game where an AI agent (controlled by Q-Learning) learns to eat food, avoid collisions (border/self-body), and maximize cumulative reward through 500 episodes of auto-training. The agent uses an epsilon-greedy strategy to balance exploration and exploitation, and the Q-Table is optimized via the Bellman Equation.

### Core Features
- **Auto Q-Learning Training**: 500 episodes of background training (no UI) with epsilon decay (1.0 → 0.05)
- **Discrete State/Action Space**: 9-dimensional state encoding + 4 discrete actions (Up/Down/Left/Right)
- **Hierarchical Reward Function**: Positive rewards for food/eating/survival, negative rewards for collision/obstacle approach
- **Pygame UI**: Real-time display of agent action, reward, score, and exploration rate (meets assignment 3.2 requirement)
- **Mouse-Only Interaction**: Intuitive Game Start/Restart buttons with hover effects (smooth user experience)

## Technical Stack
- **Programming Language**: Python 3.7+
- **Reinforcement Learning**: Table-based Q-Learning (Bellman Equation, Epsilon-Greedy)
- **Game UI/Environment**: Pygame 2.5+ (graphics, rendering, mouse interaction)
- **Numerical Calculation**: Numpy 1.24+ (Q-Table update, state encoding)

## Environment Requirements
Install the required dependencies via pip:
```bash
pip install pygame numpy
