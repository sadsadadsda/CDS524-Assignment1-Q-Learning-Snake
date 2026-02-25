import pygame
import numpy as np
import random
import sys
import math
from pygame import gfxdraw

# Initialize Pygame core modules
pygame.init()
pygame.font.init()

# -------------------------- Global Configuration Constants --------------------------
# Game grid size: 15*15 discrete grid
GRID_SIZE = 15
# Pixel size of each grid cell for UI rendering
CELL_SIZE = 40
# Game window dimensions: grid + right-side info panel
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 320
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

# Color palette (RGB format) - high contrast for clear UI display
COLOR_BG = (245, 242, 238)       # Background color
COLOR_GRID = (208, 204, 199)     # Grid line color
COLOR_SNAKE_HEAD = (237, 85, 101)# Snake head color (distinct from body)
COLOR_SNAKE_BODY = (82, 183, 136)# Snake body color (gradient effect)
COLOR_FOOD = (250, 202, 87)      # Food color (high visibility)
COLOR_TEXT = (54, 54, 54)        # Text color
COLOR_BORDER = (129, 126, 123)   # UI border color
COLOR_BUTTON = (80, 129, 202)    # Button default color
COLOR_BUTTON_HOVER = (100, 149, 222)# Button hover color (interaction feedback)

# Font configuration (Microsoft YaHei - compatible with English/Chinese display)
FONT_LARGE = pygame.font.SysFont("Microsoft YaHei", 40, bold=True)  # Title font
FONT_MEDIUM = pygame.font.SysFont("Microsoft YaHei", 24, bold=True) # Button/key info font
FONT_SMALL = pygame.font.SysFont("Microsoft YaHei", 17)              # Info panel font
FONT_TINY = pygame.font.SysFont("Microsoft YaHei", 13)               # Hint text font

# Q-Learning core hyperparameters (tuned for balance of exploration/exploitation)
LEARNING_RATE = 0.1       # α (Learning Rate): controls Q-value update step size
DISCOUNT_FACTOR = 0.9     # γ (Discount Factor): weights future reward importance
INIT_EPSILON = 1.0        # Initial exploration rate (100% random actions)
EPSILON_DECAY = 0.995     # Exponential decay factor for epsilon
MIN_EPSILON = 0.05        # Minimum exploration rate (5% to avoid local optimum)
TRAINING_EPISODES = 500   # Total training episodes for optimal policy convergence
BATTLE_FPS = 8            # Game play FPS (slow for clear observation/ demonstration)

# -------------------------- Core Game Class: Snake --------------------------
class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()  # Initialize snake to default state on creation

    def reset(self):
        """Reset snake to initial state: center grid position, 3 body segments, move right"""
        self.head_x = self.grid_size // 2
        self.head_y = self.grid_size // 2
        # Initial body structure: head + 2 rear segments (no overlap)
        self.body = [
            (self.head_x, self.head_y),
            (self.head_x - 1, self.head_y),
            (self.head_x - 2, self.head_y)
        ]
        self.direction = (1, 0)  # Initial movement direction: right (x+1, y=0)
        self.grow = False        # Grow flag: activated when food is eaten

    def move(self):
        """Update snake position by current direction, implement grow logic on food consumption"""
        # Calculate new head position based on current direction
        new_head_x = self.head_x + self.direction[0]
        new_head_y = self.head_y + self.direction[1]
        self.head_x, self.head_y = new_head_x, new_head_y
        # Insert new head to the front of the body list
        self.body.insert(0, (self.head_x, self.head_y))
        # Remove last segment if not growing (normal movement)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False  # Reset grow flag after movement

    def change_direction(self, new_dir):
        """Change snake movement direction, forbid reverse direction (core game rule)"""
        # Prevent reverse: e.g., right (1,0) -> left (-1,0) is not allowed
        if (new_dir[0] != -self.direction[0]) or (new_dir[1] != -self.direction[1]):
            self.direction = new_dir

    def check_collision(self):
        """Check collision with grid border or self-body, return True if collision occurs"""
        # Border collision: head position exceeds grid boundaries
        if (self.head_x < 0 or self.head_x >= self.grid_size or
            self.head_y < 0 or self.head_y >= self.grid_size):
            return True
        # Self collision: head position overlaps with body segments (excluding head itself)
        if (self.head_x, self.head_y) in self.body[1:]:
            return True
        return False

    def eat_food(self, food_pos):
        """Check if snake head collides with food position, set grow flag if true"""
        if (self.head_x, self.head_y) == food_pos:
            self.grow = True
            return True
        return False

# -------------------------- Core Game Class: Food --------------------------
class Food:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.pos = (0, 0)
        self.reset()  # Randomly generate initial food position

    def reset(self):
        """Randomly generate food position within grid boundaries"""
        self.pos = (random.randint(0, self.grid_size-1),
                    random.randint(0, self.grid_size-1))

    def respawn(self, snake_body):
        """Respawn food to empty grid cell, avoid overlapping with snake body"""
        # Filter all grid cells not occupied by the snake body
        empty_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in snake_body]
        # Select random empty cell (default to (0,0) if grid is full - snake max length)
        self.pos = random.choice(empty_cells) if empty_cells else (0, 0)

# -------------------------- Q-Learning Agent Class (Core Algorithm) --------------------------
class QLearningAgent:
    def __init__(self, grid_size, lr, df, init_eps, eps_decay, min_eps):
        self.grid_size = grid_size
        self.lr = lr          # Learning Rate (α)
        self.gamma = df       # Discount Factor (γ)
        self.epsilon = init_eps    # Current exploration rate
        self.epsilon_decay = eps_decay  # Epsilon decay coefficient
        self.min_epsilon = min_eps # Minimum exploration rate (global optimum search)
        # Discrete action space: 4 possible actions (up, down, left, right) -> coordinate offset
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.n_actions = len(self.actions)  # Number of possible actions: 4
        # Q-Table: dictionary with discrete state tuple as key, Q-value list as value
        # Structure: {state: [q_up, q_down, q_left, q_right]}
        self.q_table = {}

    def get_state(self, snake, food):
        """
        Encode game state to 9-dimensional discrete tuple (hashable for Q-Table key)
        State features (9D): 4x obstacle detection + 4x food relative position + 1x movement direction
        All features are binary (0/1) for efficient Q-Table storage and update
        """
        # Extract core game state information
        hx, hy = snake.head_x, snake.head_y
        fx, fy = food.pos
        dx, dy = snake.direction

        # Feature 1: Obstacle detection (4D) - 1=obstacle (border/body), 0=no obstacle
        obs_up = 1 if (hy - 1 < 0 or (hx, hy - 1) in snake.body[1:]) else 0
        obs_down = 1 if (hy + 1 >= self.grid_size or (hx, hy + 1) in snake.body[1:]) else 0
        obs_left = 1 if (hx - 1 < 0 or (hx - 1, hy) in snake.body[1:]) else 0
        obs_right = 1 if (hx + 1 >= self.grid_size or (hx + 1, hy) in snake.body[1:]) else 0

        # Feature 2: Food relative position (4D) - 1=food in direction, 0=no food
        food_up = 1 if fy < hy else 0
        food_down = 1 if fy > hy else 0
        food_left = 1 if fx < hx else 0
        food_right = 1 if fx > hx else 0

        # Feature 3: Movement direction (1D) - encode to 0-3 (up/down/left/right)
        dir_code = 0 if (dx, dy) == (0, -1) else \
                   1 if (dx, dy) == (0, 1) else \
                   2 if (dx, dy) == (-1, 0) else 3

        # Compose final 9-dimensional discrete state tuple
        state = (
            obs_up, obs_down, obs_left, obs_right,
            food_up, food_down, food_left, food_right,
            dir_code
        )
        # Initialize Q-values to [0.0, 0.0, 0.0, 0.0] for new/unseen states
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.n_actions)]
        return state

    def choose_action(self, state):
        """
        Epsilon-greedy action selection strategy: balance exploration and exploitation
        - Exploration: random action selection (probability = epsilon) -> discover new policies
        - Exploitation: select action with maximum Q-value (probability = 1-epsilon) -> use optimal policy
        Returns: action index (0=up, 1=down, 2=left, 3=right)
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: random action (0-3)
            return random.choice(range(self.n_actions))
        else:
            # Exploitation: optimal action (max Q-value)
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-Table using Bellman Equation (core of Q-Learning algorithm)
        Formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        :param state: Current discrete game state (9D tuple)
        :param action: Current action index (0-3)
        :param reward: Immediate reward from current state-action pair
        :param next_state: Next discrete game state after action execution
        :param done: Boolean (True = terminal state/collision, False = non-terminal state)
        """
        current_q = self.q_table[state][action]
        # Terminal state (collision): no future reward, target Q = immediate reward
        if done:
            target_q = reward
        # Non-terminal state: target Q = immediate reward + discounted future max Q-value
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        # Update current Q-value for state-action pair
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Exponentially decay exploration rate, stop when reaching minimum epsilon"""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# -------------------------- Game Manager Class (UI + Logic Integration) --------------------------
class SnakeGame:
    def __init__(self):
        # Initialize game window and set title (CDS524 Assignment 1 identification)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CDS524 Assignment 1 - Q-Learning Snake Game")
        self.clock = pygame.time.Clock()  # FPS controller for smooth UI rendering

        # Initialize core game objects
        self.snake = Snake(GRID_SIZE)
        self.food = Food(GRID_SIZE)
        self.agent = QLearningAgent(
            GRID_SIZE, LEARNING_RATE, DISCOUNT_FACTOR,
            INIT_EPSILON, EPSILON_DECAY, MIN_EPSILON
        )

        # Game state variables for UI display (meet assignment 3.2 requirement)
        self.score = 0               # Number of food eaten (core game metric)
        self.cumulative_reward = 0   # Total cumulative reward (RL metric)
        self.instant_reward = 0      # Immediate reward of current step
        self.current_action = "None" # Current agent action (for UI display)
        self.game_over = False       # Game over flag (collision detected)
        self.start_screen = False    # Start screen flag (activated after auto-training)

        # Auto start Q-Learning training (no manual operation required)
        self.auto_train()

    def auto_train(self):
        """
        Auto train Q-Learning agent for TRAINING_EPISODES (500) in background (no UI rendering)
        Fast training without graphics: focus on Q-Table optimization
        Prints training progress to console for demonstration/ verification
        """
        print("=== CDS524 Assignment 1 - Q-Learning Auto Training Start ===")
        print(f"Training Configuration | Episodes: {TRAINING_EPISODES} | Init Epsilon: {INIT_EPSILON} | Min Epsilon: {MIN_EPSILON}")
        print(f"Hyperparameters | Learning Rate: {LEARNING_RATE} | Discount Factor: {DISCOUNT_FACTOR} | Epsilon Decay: {EPSILON_DECAY}")
        episode = 0
        while episode < TRAINING_EPISODES:
            # Reset game state for each training episode
            self.snake.reset()
            self.food.reset()
            game_over = False

            # Single episode training loop: state -> action -> reward -> Q-Table update
            while not game_over:
                state = self.agent.get_state(self.snake, self.food)
                action_idx = self.agent.choose_action(state)
                self.snake.change_direction(self.agent.actions[action_idx])
                self.snake.move()
                reward = self._calc_train_reward()
                game_over = self.snake.check_collision()
                next_state = self.agent.get_state(self.snake, self.food)
                self.agent.update_q_table(state, action_idx, reward, next_state, game_over)
                self.agent.decay_epsilon()

            episode += 1
            # Print training progress every 50 episodes (easy to track convergence)
            if episode % 50 == 0:
                print(f"Training Progress: {episode}/{TRAINING_EPISODES} Episodes | Current Exploration Rate: {round(self.agent.epsilon, 3)}")

        # Post-training configuration: fix epsilon to minimum, activate start screen
        self.agent.epsilon = MIN_EPSILON
        self.start_screen = True
        print("=== Training Completed! Optimal Q-Table Generated ===")
        print("=== Click 'Game Start' to launch the game ===")

    def _calc_train_reward(self):
        """Reward function for training (no UI update, faster computation)"""
        reward = 0
        # Severe negative reward (-20) for collision (terminal state - game over)
        if self.snake.check_collision():
            reward -= 20
        # Major positive reward (+10) for eating food (core game objective)
        elif self.snake.eat_food(self.food.pos):
            reward += 10
            self.food.respawn(self.snake.body)
        # Minor positive reward (+2) for moving toward food (guidance for exploration)
        elif self._is_toward_food():
            reward += 2
        # Minor negative reward (-3) for moving toward obstacle (danger avoidance)
        elif self._is_toward_obstacle():
            reward -= 3
        # Survival positive reward (+1) for valid movement (encourage exploration)
        else:
            reward += 1
        return reward

    def calculate_reward(self):
        """Reward function for game play (with UI update for real-time display)"""
        reward = 0
        if self.snake.check_collision():
            reward -= 20
        elif self.snake.eat_food(self.food.pos):
            reward += 10
            self.score += 1
            self.food.respawn(self.snake.body)
        elif self._is_toward_food():
            reward += 2
        elif self._is_toward_obstacle():
            reward -= 3
        else:
            reward += 1
        # Update reward variables for UI display (Assignment 3.2 requirement)
        self.instant_reward = reward
        self.cumulative_reward += reward
        return reward

    def _is_toward_food(self):
        """Check if snake is moving directly toward food (horizontal/vertical), return True if yes"""
        hx, hy = self.snake.head_x, self.snake.head_y
        fx, fy = self.food.pos
        dx, dy = self.snake.direction
        # Check if movement direction aligns with food position (x/y axis)
        return (fx > hx and dx == 1) or (fx < hx and dx == -1) or (fy > hy and dy == 1) or (fy < hy and dy == -1)

    def _is_toward_obstacle(self):
        """Check if snake's next step is an obstacle (border/self-body), return True if yes"""
        hx, hy = self.snake.head_x, self.snake.head_y
        dx, dy = self.snake.direction
        # Calculate next step position based on current direction
        nx, ny = hx + dx, hy + dy
        # Check if next step is out of grid or overlaps with body
        return nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or (nx, ny) in self.snake.body[1:]

    def draw_gradient_bg(self):
        """Draw gradient background for enhanced visual experience (no functional impact)"""
        for y in range(WINDOW_HEIGHT):
            # Simple linear gradient calculation
            color = (
                min(COLOR_BG[0] + y * 0.05, 255),
                min(COLOR_BG[1] + y * 0.05, 255),
                min(COLOR_BG[2] + y * 0.05, 255)
            )
            pygame.draw.line(self.screen, color, (0, y), (WINDOW_WIDTH, y))

    def draw_grid(self):
        """Draw 15x15 game grid with outer border and internal grid lines (core game environment)"""
        # Draw grid outer border (3px width for visibility)
        pygame.draw.rect(
            self.screen, COLOR_BORDER,
            (0, 0, self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE),
            3
        )
        # Draw vertical grid lines (1px width)
        for x in range(1, self.grid_size):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, self.grid_size * CELL_SIZE),
                1
            )
        # Draw horizontal grid lines (1px width)
        for y in range(1, self.grid_size):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (0, y * CELL_SIZE),
                (self.grid_size * CELL_SIZE, y * CELL_SIZE),
                1
            )

    def draw_snake(self):
        """Draw snake with distinct head/body design, gradient body color and direction-aligned eyes"""
        # Draw snake head (rounded rectangle, distinct color - high visibility)
        head_rect = pygame.Rect(
            self.snake.head_x * CELL_SIZE + 2,
            self.snake.head_y * CELL_SIZE + 2,
            CELL_SIZE - 4,
            CELL_SIZE - 4
        )
        pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, head_rect, border_radius=8)

        # Draw snake eyes (white small circles, aligned with movement direction)
        eye_radius = 3
        head_center = head_rect.center
        if self.snake.direction == (1, 0):  # Move Right
            eyes = [(head_center[0] + 10, head_center[1] - 5), (head_center[0] + 10, head_center[1] + 5)]
        elif self.snake.direction == (-1, 0):  # Move Left
            eyes = [(head_center[0] - 10, head_center[1] - 5), (head_center[0] - 10, head_center[1] + 5)]
        elif self.snake.direction == (0, 1):  # Move Down
            eyes = [(head_center[0] - 5, head_center[1] + 10), (head_center[0] + 5, head_center[1] + 10)]
        else:  # Move Up
            eyes = [(head_center[0] - 5, head_center[1] - 10), (head_center[0] + 5, head_center[1] - 10)]
        # Render eyes
        for (x, y) in eyes:
            gfxdraw.filled_circle(self.screen, x, y, eye_radius, (255, 255, 255))

        # Draw snake body (gradient color, rounded rectangle - visual depth)
        for i, (x, y) in enumerate(self.snake.body[1:]):
            # Gradient alpha: body segments far from head are lighter (0.5 min opacity)
            alpha = max(0.5, 1 - i * 0.02)
            body_color = (
                int(COLOR_SNAKE_BODY[0] * alpha),
                int(COLOR_SNAKE_BODY[1] * alpha),
                int(COLOR_SNAKE_BODY[2] * alpha)
            )
            # Body segment rectangle (smaller than cell - no overlap)
            body_rect = pygame.Rect(
                x * CELL_SIZE + 4,
                y * CELL_SIZE + 4,
                CELL_SIZE - 8,
                CELL_SIZE - 8
            )
            pygame.draw.rect(self.screen, body_color, body_rect, border_radius=6)

    def draw_food(self):
        """Draw food as a five-pointed star (distinct from snake - high visibility)"""
        fx, fy = self.food.pos
        # Calculate food cell center position
        center_x = fx * CELL_SIZE + CELL_SIZE // 2
        center_y = fy * CELL_SIZE + CELL_SIZE // 2
        # Star size parameters (scaled to cell size)
        outer_radius = CELL_SIZE // 2 - 6
        inner_radius = CELL_SIZE // 4 - 3
        star_points = []
        # Calculate five-pointed star vertex coordinates (trigonometric calculation)
        for i in range(5):
            # Outer star vertex (72° interval)
            angle_outer = math.radians(90 + i * 72)
            star_points.append((int(center_x + outer_radius * math.cos(angle_outer)),
                                int(center_y - outer_radius * math.sin(angle_outer))))
            # Inner star vertex (72° interval)
            angle_inner = math.radians(126 + i * 72)
            star_points.append((int(center_x + inner_radius * math.cos(angle_inner)),
                                int(center_y - inner_radius * math.sin(angle_inner))))
        # Draw filled star and border (for visibility)
        gfxdraw.filled_polygon(self.screen, star_points, COLOR_FOOD)
        gfxdraw.aapolygon(self.screen, star_points, COLOR_BORDER)

    def draw_info_panel(self):
        """
        Draw right-side info panel (meets CDS524 Assignment 3.2 requirement)
        Displays real-time game/RL metrics: action, reward, score, exploration rate
        """
        panel_x = self.grid_size * CELL_SIZE + 15  # Panel left edge position
        # Panel title and horizontal divider
        self.screen.blit(FONT_MEDIUM.render("Game & RL Info", True, COLOR_TEXT), (panel_x, 20))
        pygame.draw.line(self.screen, COLOR_GRID, (panel_x, 60), (WINDOW_WIDTH - 15, 60), 2)

        # Info panel content (key metrics for assignment demonstration)
        info_items = [
            (f"Current Agent Action: {self.current_action}", 80),
            (f"Immediate Step Reward: {self.instant_reward}", 115),
            (f"Total Cumulative Reward: {self.cumulative_reward}", 150),
            (f"Food Eaten (Score): {self.score}", 185),
            ("", 220),  # Empty line for spacing
            (f"Exploration Rate (Epsilon): {round(self.agent.epsilon, 3)}", 255),
            ("✅ Trained Q-Table Loaded", 290)  # RL status indicator
        ]
        # Render all info items
        for text, y_pos in info_items:
            text_surface = FONT_SMALL.render(text, True, COLOR_TEXT)
            self.screen.blit(text_surface, (panel_x, y_pos))

        # Panel bottom divider and hint text
        pygame.draw.line(self.screen, COLOR_GRID, (panel_x, 380), (WINDOW_WIDTH - 15, 380), 2)
        self.screen.blit(FONT_TINY.render("Hint: Click 'Restart Game' to play again", True, COLOR_TEXT), (panel_x, 400))

    def draw_start_screen(self):
        """Draw post-training start screen with single 'Game Start' button (mouse interaction)"""
        self.draw_gradient_bg()
        # Game title (centralized)
        game_title = FONT_LARGE.render("Q-Learning Snake Game", True, COLOR_TEXT)
        self.screen.blit(game_title, game_title.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60)))
        # Training completion status (centralized)
        train_status = FONT_SMALL.render("500 Episodes Trained | Optimal Q-Table Ready", True, COLOR_TEXT)
        self.screen.blit(train_status, train_status.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 10)))

        # Game Start button (centralized, hover effect for interaction)
        self.start_btn = pygame.Rect(WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT // 2 + 30, 240, 50)
        btn_color = COLOR_BUTTON_HOVER if self.start_btn.collidepoint(pygame.mouse.get_pos()) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.start_btn, border_radius=10)
        # Button text
        start_text = FONT_MEDIUM.render("Game Start", True, (255, 255, 255))
        self.screen.blit(start_text, start_text.get_rect(center=self.start_btn.center))

        # Assignment identification (bottom center)
        assigment_text = FONT_SMALL.render("CDS524 Assignment 1 - Reinforcement Learning", True, COLOR_TEXT)
        self.screen.blit(assigment_text, assigment_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 50)))

    def draw_game_over(self):
        """Draw game over screen with restart button and final game/RL metrics (mouse interaction)"""
        # Transparent black mask (highlight game over content, no full screen blackout)
        mask_surface = pygame.Surface((self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE), pygame.SRCALPHA)
        mask_surface.fill((0, 0, 0, 128))  # 128 = 50% opacity
        self.screen.blit(mask_surface, (0, 0))

        # Game over text (red, high visibility)
        game_over_text = FONT_LARGE.render("Game Over", True, (237, 85, 101))
        self.screen.blit(game_over_text, (self.grid_size * CELL_SIZE // 2 - 120, self.grid_size * CELL_SIZE // 2 - 70))
        # Final game metrics (white, on mask for contrast)
        final_score = FONT_MEDIUM.render(f"Final Food Score: {self.score}", True, (255, 255, 255))
        final_reward = FONT_MEDIUM.render(f"Total Cumulative Reward: {self.cumulative_reward}", True, (255, 255, 255))
        self.screen.blit(final_score, (self.grid_size * CELL_SIZE // 2 - 80, self.grid_size * CELL_SIZE // 2 - 10))
        self.screen.blit(final_reward, (self.grid_size * CELL_SIZE // 2 - 100, self.grid_size * CELL_SIZE // 2 + 30))

        # Restart Game button (centralized, hover effect)
        self.restart_btn = pygame.Rect(self.grid_size * CELL_SIZE // 2 - 80, self.grid_size * CELL_SIZE // 2 + 70, 160, 40)
        btn_color = COLOR_BUTTON_HOVER if self.restart_btn.collidepoint(pygame.mouse.get_pos()) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.restart_btn, border_radius=8)
        # Button text
        restart_text = FONT_SMALL.render("Restart Game", True, (255, 255, 255))
        self.screen.blit(restart_text, restart_text.get_rect(center=self.restart_btn.center))

    def reset_game(self):
        """
        Reset game state for restart (meets assignment interaction requirement)
        Reuses trained Q-Table (no re-training) - core RL application logic
        Resets only game play metrics, not agent/ Q-Table
        """
        self.snake.reset()
        self.food.reset()
        self.score = 0
        self.cumulative_reward = 0
        self.instant_reward = 0
        self.current_action = "None"
        self.game_over = False

    def battle_step(self):
        """Game play step: agent uses trained Q-Table to select optimal actions (no training)"""
        # Get current discrete game state
        current_state = self.agent.get_state(self.snake, self.food)
        # Select action (epsilon fixed to MIN_EPSILON: 95% exploit, 5% explore)
        action_index = self.agent.choose_action(current_state)
        # Execute action (change direction + move)
        self.snake.change_direction(self.agent.actions[action_index])
        self.snake.move()
        # Calculate reward and update UI metrics
        self.calculate_reward()
        # Check for game over (collision)
        self.game_over = self.snake.check_collision()
        # Update current action text for UI display (map index to action name)
        self.current_action = ["Up", "Down", "Left", "Right"][action_index]

    def run(self):
        """
        Main game loop (core UI/ interaction logic)
        Mouse-only interaction (no keyboard) - simple and intuitive for demonstration
        Meets CDS524 Assignment 3.1/3.4 requirements
        """
        while True:
            # Get mouse state (position + left click press)
            mouse_position = pygame.mouse.get_pos()
            mouse_click = pygame.mouse.get_pressed()

            # Event handling: only window close event (simple interaction)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Start screen logic: click Game Start to enter game play
            if self.start_screen:
                self.draw_start_screen()
                if mouse_click[0] and self.start_btn.collidepoint(mouse_position):
                    self.start_screen = False
                    self.reset_game()
            else:
                # Game play logic: render environment + execute agent step
                self.draw_gradient_bg()
                self.draw_grid()
                if not self.game_over:
                    self.battle_step()
                # Draw all game/UI elements
                self.draw_snake()
                self.draw_food()
                self.draw_info_panel()

                # Game over logic: draw game over screen + restart on click
                if self.game_over:
                    self.draw_game_over()
                    if mouse_click[0] and self.restart_btn.collidepoint(mouse_position):
                        self.reset_game()

            # Update game window and control FPS (smooth rendering)
            pygame.display.flip()
            self.clock.tick(BATTLE_FPS)

# -------------------------- Main Program Entry Point --------------------------
if __name__ == "__main__":
    # Initialize game object and start main loop
    snake_game = SnakeGame()
    snake_game.run()
