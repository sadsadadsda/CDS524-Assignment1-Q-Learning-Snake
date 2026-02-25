import pygame
import numpy as np
import random
import sys
import math
from pygame import gfxdraw

# Initialize Pygame library
pygame.init()
# Initialize Pygame font module
pygame.font.init()

# -------------------------- Global Configuration Constants --------------------------
# Game grid size: 15*15 grid
GRID_SIZE = 15
# Pixel size of each grid cell
CELL_SIZE = 40
# Game window width: grid width + right info panel width
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 320
# Game window height: same as grid height
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE

# Color configuration (RGB format)
COLOR_BG = (245, 242, 238)       # Background color
COLOR_GRID = (208, 204, 199)     # Grid line color
COLOR_SNAKE_HEAD = (237, 85, 101)# Snake head color
COLOR_SNAKE_BODY = (82, 183, 136)# Snake body color
COLOR_FOOD = (250, 202, 87)      # Food color
COLOR_TEXT = (54, 54, 54)        # Text color
COLOR_BORDER = (129, 126, 123)   # Border color
COLOR_BUTTON = (80, 129, 202)    # Button default color
COLOR_BUTTON_HOVER = (100, 149, 222)# Button hover color

# Font configuration (Microsoft YaHei, compatible with English display)
FONT_LARGE = pygame.font.SysFont("Microsoft YaHei", 40, bold=True)  # Large font (title)
FONT_MEDIUM = pygame.font.SysFont("Microsoft YaHei", 24, bold=True) # Medium font (button/key info)
FONT_SMALL = pygame.font.SysFont("Microsoft YaHei", 17)              # Small font (info panel)
FONT_TINY = pygame.font.SysFont("Microsoft YaHei", 13)               # Tiny font (tips)

# Q-Learning core hyperparameters
LEARNING_RATE = 0.1       # Learning rate (α): controls the step of Q-value update
DISCOUNT_FACTOR = 0.9     # Discount factor (γ): weights future rewards
INIT_EPSILON = 1.0        # Initial epsilon: 100% random exploration at the start
EPSILON_DECAY = 0.995     # Epsilon decay factor: gradually reduce exploration
MIN_EPSILON = 0.05        # Minimum epsilon: reserve 5% exploration to avoid local optimum
TRAINING_EPISODES = 500   # Total training episodes: 500 rounds for optimal policy
BATTLE_FPS = 8            # Game play FPS: slow down to 8 for clear observation

# -------------------------- Core Game Class: Snake --------------------------
class Snake:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()  # Initialize snake state by reset function

    def reset(self):
        """Reset snake to initial state: center position, 3 body segments, move right"""
        self.head_x = self.grid_size // 2
        self.head_y = self.grid_size // 2
        # Initial body: head + 2 rear segments
        self.body = [
            (self.head_x, self.head_y),
            (self.head_x - 1, self.head_y),
            (self.head_x - 2, self.head_y)
        ]
        self.direction = (1, 0)  # Initial move direction: right (x+1, y=0)
        self.grow = False        # Grow flag: True when snake eats food

    def move(self):
        """Update snake position by current direction, implement grow logic"""
        # Calculate new head position
        new_head_x = self.head_x + self.direction[0]
        new_head_y = self.head_y + self.direction[1]
        self.head_x, self.head_y = new_head_x, new_head_y
        # Insert new head to the front of body
        self.body.insert(0, (self.head_x, self.head_y))
        # If not grow, remove the last body segment (normal move)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False  # Reset grow flag after moving

    def change_direction(self, new_dir):
        """Change move direction, forbid reverse direction (game rule)"""
        # Avoid reverse: e.g., right (1,0) -> left (-1,0) is not allowed
        if (new_dir[0] != -self.direction[0]) or (new_dir[1] != -self.direction[1]):
            self.direction = new_dir

    def check_collision(self):
        """Check collision with grid border or self body, return True if collision"""
        # Border collision: head out of grid range
        if (self.head_x < 0 or self.head_x >= self.grid_size or
            self.head_y < 0 or self.head_y >= self.grid_size):
            return True
        # Self collision: head position in body segments (exclude head itself)
        if (self.head_x, self.head_y) in self.body[1:]:
            return True
        return False

    def eat_food(self, food_pos):
        """Check if snake head collides with food, return True if eat food"""
        if (self.head_x, self.head_y) == food_pos:
            self.grow = True  # Set grow flag to True
            return True
        return False

# -------------------------- Core Game Class: Food --------------------------
class Food:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.pos = (0, 0)
        self.reset()  # Initialize food position

    def reset(self):
        """Random generate food position in grid"""
        self.pos = (random.randint(0, self.grid_size-1),
                    random.randint(0, self.grid_size-1))

    def respawn(self, snake_body):
        """Respawn food, avoid overlapping with snake body"""
        # Filter all empty cells not occupied by snake body
        empty_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in snake_body]
        if empty_cells:
            self.pos = random.choice(empty_cells)  # Random select empty cell
        else:
            self.pos = (0, 0)  # Default position if grid is full (snake max length)

# -------------------------- Q-Learning Agent Class --------------------------
class QLearningAgent:
    def __init__(self, grid_size, learning_rate, discount_factor, init_epsilon, epsilon_decay, min_epsilon):
        self.grid_size = grid_size
        self.lr = learning_rate        # Learning rate (α)
        self.gamma = discount_factor   # Discount factor (γ)
        self.epsilon = init_epsilon    # Current exploration rate
        self.epsilon_decay = epsilon_decay  # Epsilon decay coefficient
        self.min_epsilon = min_epsilon # Minimum exploration rate
        # Action space: 4 discrete actions (up, down, left, right) -> coordinate offset
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.n_actions = len(self.actions)  # Number of actions: 4
        self.q_table = {}  # Q-table: dictionary {state: [q_up, q_down, q_left, q_right]}

    def get_state(self, snake, food):
        """
        Encode game state to discrete 9-dimensional tuple (hashable for Q-table key)
        State features: 4 obstacle + 4 food position + 1 move direction
        """
        # Extract basic game information
        head_x, head_y = snake.head_x, snake.head_y
        food_x, food_y = food.pos
        dir_x, dir_y = snake.direction

        # Feature 1: Obstacle detection (4D) - 1=obstacle (border/body), 0=no obstacle
        obstacle_up = 1 if (head_y - 1 < 0 or (head_x, head_y - 1) in snake.body[1:]) else 0
        obstacle_down = 1 if (head_y + 1 >= self.grid_size or (head_x, head_y + 1) in snake.body[1:]) else 0
        obstacle_left = 1 if (head_x - 1 < 0 or (head_x - 1, head_y) in snake.body[1:]) else 0
        obstacle_right = 1 if (head_x + 1 >= self.grid_size or (head_x + 1, head_y) in snake.body[1:]) else 0

        # Feature 2: Food relative position (4D) - 1=food in this direction, 0=no food
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0

        # Feature 3: Current move direction (1D) - encode to 0-3 (up/down/left/right)
        dir_code = 0 if (dir_x, dir_y) == (0, -1) else \
                   1 if (dir_x, dir_y) == (0, 1) else \
                   2 if (dir_x, dir_y) == (-1, 0) else 3

        # Compose 9-dimensional discrete state tuple
        state = (
            obstacle_up, obstacle_down, obstacle_left, obstacle_right,
            food_up, food_down, food_left, food_right,
            dir_code
        )
        # Initialize Q-value to [0.0, 0.0, 0.0, 0.0] if state is new in Q-table
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.n_actions)]
        return state

    def choose_action(self, state):
        """
        Epsilon-greedy action selection: balance exploration and exploitation
        - Exploration: random action (probability = epsilon)
        - Exploitation: optimal action (max Q-value) (probability = 1-epsilon)
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: random select action index (0-3)
            return random.choice(range(self.n_actions))
        else:
            # Exploitation: select action with maximum Q-value
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        """
        Update Q-table with Bellman equation (core of Q-Learning)
        Formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        :param state: current state
        :param action: current action index
        :param reward: immediate reward
        :param next_state: next state after action
        :param done: bool, True if game over (terminal state)
        """
        current_q = self.q_table[state][action]
        # Terminal state (collision): no future reward, target Q = immediate reward
        if done:
            target_q = reward
        # Non-terminal state: target Q = immediate reward + discounted future max Q-value
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        # Update current Q-value
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate, stop when reach minimum epsilon"""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

# -------------------------- Game Manager Class (Core Logic) --------------------------
class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE, cell_size=CELL_SIZE):
        # Initialize game window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Learning Snake Game | CDS524 Assignment 1")
        self.clock = pygame.time.Clock()  # FPS controller
        self.grid_size = grid_size
        self.cell_size = cell_size

        # Initialize game objects
        self.snake = Snake(grid_size)
        self.food = Food(grid_size)
        self.agent = QLearningAgent(
            grid_size, LEARNING_RATE, DISCOUNT_FACTOR,
            INIT_EPSILON, EPSILON_DECAY, MIN_EPSILON
        )

        # Game state variables (for UI display)
        self.score = 0               # Food eaten count
        self.cumulative_reward = 0   # Total cumulative reward
        self.instant_reward = 0      # Immediate reward of current step
        self.current_action = "None" # Current agent action (for display)
        self.game_over = False       # Game over flag
        self.start_screen = False    # Start screen flag: True after auto training

        # Auto start training when game initializes (no manual operation)
        self.auto_train()

    def auto_train(self):
        """
        Auto train Q-Learning agent for TRAINING_EPISODES rounds (background, no UI)
        Train fast without rendering, generate optimal Q-table for game play
        """
        print("Start auto training Q-Learning agent... (Total {} episodes)".format(TRAINING_EPISODES))
        episode = 0
        while episode < TRAINING_EPISODES:
            # Reset game state for each training episode
            self.snake.reset()
            self.food.reset()
            game_over = False

            # Single episode training loop
            while not game_over:
                # Q-Learning step: get state -> choose action -> execute action
                current_state = self.agent.get_state(self.snake, self.food)
                action_idx = self.agent.choose_action(current_state)
                action = self.agent.actions[action_idx]
                self.snake.change_direction(action)
                self.snake.move()

                # Calculate immediate reward
                reward = self._calculate_train_reward()
                # Check if game over (collision)
                game_over = self.snake.check_collision()
                # Get next state
                next_state = self.agent.get_state(self.snake, self.food)
                # Update Q-table
                self.agent.update_q_table(current_state, action_idx, reward, next_state, game_over)
                # Decay exploration rate
                self.agent.decay_epsilon()

            # Update training episode count
            episode += 1
            # Print training progress every 50 episodes
            if episode % 50 == 0:
                print("Training Progress: {}/{} episodes, Current Epsilon: {:.3f}".format(episode, TRAINING_EPISODES, self.agent.epsilon))

        # After training: fix epsilon to minimum, enable start screen
        self.agent.epsilon = MIN_EPSILON
        self.start_screen = True
        print("✅ Training Completed! Optimal Q-table generated, click Game Start to play.")

    def _calculate_train_reward(self):
        """Reward calculation for training (no UI update, faster training)"""
        reward = 0
        # Severe penalty (-20) for collision (game over)
        if self.snake.check_collision():
            reward -= 20
        # Major positive reward (+10) for eating food (core game objective)
        elif self.snake.eat_food(self.food.pos):
            reward += 10
            self.food.respawn(self.snake.body)
        # Minor positive reward (+2) for moving toward food (guide exploration)
        elif self._is_toward_food():
            reward += 2
        # Minor penalty (-3) for moving toward obstacle (avoid danger)
        elif self._is_toward_obstacle():
            reward -= 3
        # Survival reward (+1) for normal move (encourage exploration)
        else:
            reward += 1
        return reward

    def calculate_reward(self):
        """Reward calculation for game play (with UI update for display)"""
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
        # Update reward variables for UI display
        self.instant_reward = reward
        self.cumulative_reward += reward
        return reward

    def _is_toward_food(self):
        """Check if snake is moving toward food, return True if yes"""
        head_x, head_y = self.snake.head_x, self.snake.head_y
        food_x, food_y = self.food.pos
        dir_x, dir_y = self.snake.direction
        # Check horizontal/vertical direction match
        if (food_x > head_x and dir_x == 1) or (food_x < head_x and dir_x == -1):
            return True
        if (food_y > head_y and dir_y == 1) or (food_y < head_y and dir_y == -1):
            return True
        return False

    def _is_toward_obstacle(self):
        """Check if snake's next step is obstacle, return True if yes"""
        head_x, head_y = self.snake.head_x, self.snake.head_y
        dir_x, dir_y = self.snake.direction
        # Calculate next step position
        next_x = head_x + dir_x
        next_y = head_y + dir_y
        # Check if next step is border or snake body
        if (next_x < 0 or next_x >= self.grid_size or
            next_y < 0 or next_y >= self.grid_size or
            (next_x, next_y) in self.snake.body[1:]):
            return True
        return False

    def draw_gradient_bg(self):
        """Draw gradient background for better visual experience"""
        for y in range(WINDOW_HEIGHT):
            # Gradient color calculation
            color = (
                min(COLOR_BG[0] + y * 0.05, 255),
                min(COLOR_BG[1] + y * 0.05, 255),
                min(COLOR_BG[2] + y * 0.05, 255)
            )
            pygame.draw.line(self.screen, color, (0, y), (WINDOW_WIDTH, y))

    def draw_grid(self):
        """Draw 15*15 game grid with border and grid lines"""
        # Draw grid outer border
        pygame.draw.rect(
            self.screen, COLOR_BORDER,
            (0, 0, self.grid_size * self.cell_size, self.grid_size * self.cell_size),
            3
        )
        # Draw vertical grid lines
        for x in range(1, self.grid_size):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size * self.cell_size),
                1
            )
        # Draw horizontal grid lines
        for y in range(1, self.grid_size):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (0, y * self.cell_size),
                (self.grid_size * self.cell_size, y * self.cell_size),
                1
            )

    def draw_snake(self):
        """Draw snake with different styles for head and body (gradient body)"""
        # Draw snake head (rounded rectangle, different color)
        head_x, head_y = self.snake.head_x, self.snake.head_y
        head_rect = pygame.Rect(
            head_x * self.cell_size + 2,
            head_y * self.cell_size + 2,
            self.cell_size - 4,
            self.cell_size - 4
        )
        pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, head_rect, border_radius=8)

        # Draw snake eyes (white small circles)
        eye_radius = 3
        if self.snake.direction == (1, 0): # Right
            eye1 = (head_rect.right - 10, head_rect.centery - 5)
            eye2 = (head_rect.right - 10, head_rect.centery + 5)
        elif self.snake.direction == (-1, 0): # Left
            eye1 = (head_rect.left + 10, head_rect.centery - 5)
            eye2 = (head_rect.left + 10, head_rect.centery + 5)
        elif self.snake.direction == (0, 1): # Down
            eye1 = (head_rect.centerx - 5, head_rect.bottom - 10)
            eye2 = (head_rect.centerx + 5, head_rect.bottom - 10)
        else: # Up
            eye1 = (head_rect.centerx - 5, head_rect.top + 10)
            eye2 = (head_rect.centerx + 5, head_rect.top + 10)
        gfxdraw.filled_circle(self.screen, eye1[0], eye1[1], eye_radius, (255, 255, 255))
        gfxdraw.filled_circle(self.screen, eye2[0], eye2[1], eye_radius, (255, 255, 255))

        # Draw snake body (gradient color, rounded rectangle)
        for i, (x, y) in enumerate(self.snake.body[1:]):
            # Gradient alpha: body segments far from head are lighter
            alpha = max(0.5, 1 - i * 0.02)
            body_color = (
                int(COLOR_SNAKE_BODY[0] * alpha),
                int(COLOR_SNAKE_BODY[1] * alpha),
                int(COLOR_SNAKE_BODY[2] * alpha)
            )
            body_rect = pygame.Rect(
                x * self.cell_size + 4,
                y * self.cell_size + 4,
                self.cell_size - 8,
                self.cell_size - 8
            )
            pygame.draw.rect(self.screen, body_color, body_rect, border_radius=6)

    def draw_food(self):
        """Draw food as a five-pointed star (better visual than simple rectangle)"""
        food_x, food_y = self.food.pos
        # Calculate center position of food cell
        center_x = food_x * self.cell_size + self.cell_size // 2
        center_y = food_y * self.cell_size + self.cell_size // 2
        # Star size parameters
        outer_r = self.cell_size // 2 - 6
        inner_r = self.cell_size // 4 - 3
        points = []
        # Calculate five-pointed star vertex coordinates
        for i in range(5):
            # Outer vertex
            angle = math.radians(90 + i * 72)
            x = center_x + outer_r * math.cos(angle)
            y = center_y - outer_r * math.sin(angle)
            points.append((int(x), int(y)))
            # Inner vertex
            angle = math.radians(126 + i * 72)
            x = center_x + inner_r * math.cos(angle)
            y = center_y - inner_r * math.sin(angle)
            points.append((int(x), int(y)))
        # Draw filled star and border
        gfxdraw.filled_polygon(self.screen, points, COLOR_FOOD)
        gfxdraw.aapolygon(self.screen, points, COLOR_BORDER)

    def draw_info_panel(self):
        """Draw right info panel: display action, reward, score, epsilon (for CDS524 requirement)"""
        panel_x = self.grid_size * CELL_SIZE + 15  # Panel left position
        # Panel title
        title = FONT_MEDIUM.render("Game Info", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x, 20))
        pygame.draw.line(self.screen, COLOR_GRID, (panel_x, 60), (WINDOW_WIDTH - 15, 60), 2)

        # Info items for display (action, reward, score, epsilon)
        info_items = [
            (f"Current Action: {self.current_action}", 80),
            (f"Instant Reward: {self.instant_reward}", 115),
            (f"Cumulative Reward: {self.cumulative_reward}", 150),
            (f"Score (Food): {self.score}", 185),
            ("", 220),
            (f"Epsilon: {round(self.agent.epsilon, 3)}", 255),
            ("✅ Trained Q-Table Loaded", 290)
        ]
        # Render all info items
        for text, y in info_items:
            surface = FONT_SMALL.render(text, True, COLOR_TEXT)
            self.screen.blit(surface, (panel_x, y))

        # Panel bottom tips
        pygame.draw.line(self.screen, COLOR_GRID, (panel_x, 380), (WINDOW_WIDTH - 15, 380), 2)
        tips = FONT_TINY.render("Tips: Click Restart to play again", True, COLOR_TEXT)
        self.screen.blit(tips, (panel_x, 400))

    def draw_start_screen(self):
        """Draw start screen with single Game Start button (after auto training)"""
        self.draw_gradient_bg()
        # Game title
        title = FONT_LARGE.render("Q-Learning Snake Game", True, COLOR_TEXT)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 60))
        self.screen.blit(title, title_rect)

        # Training completion tip
        train_tip = FONT_SMALL.render("500 Rounds Training Completed | Q-Table Ready", True, COLOR_TEXT)
        train_tip_rect = train_tip.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 10))
        self.screen.blit(train_tip, train_tip_rect)

        # Single Game Start button (mouse click to play)
        self.game_start_btn = pygame.Rect(WINDOW_WIDTH//2 - 120, WINDOW_HEIGHT//2 + 30, 240, 50)
        # Button hover effect
        btn_color = COLOR_BUTTON_HOVER if self.game_start_btn.collidepoint(pygame.mouse.get_pos()) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.game_start_btn, border_radius=10)
        start_text = FONT_MEDIUM.render("Game Start", True, (255,255,255))
        start_text_rect = start_text.get_rect(center=self.game_start_btn.center)
        self.screen.blit(start_text, start_text_rect)

        # Assignment info (CDS524)
        ass_text = FONT_SMALL.render("CDS524 Assignment 1 - Reinforcement Learning", True, COLOR_TEXT)
        ass_text_rect = ass_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT - 50))
        self.screen.blit(ass_text, ass_text_rect)

    def draw_game_over_screen(self):
        """Draw game over screen with restart button (mouse click to restart)"""
        # Transparent black mask (highlight game over info)
        mask = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 128))
        self.screen.blit(mask, (0, 0))

        # Game over info text
        game_over_text = FONT_LARGE.render("Game Over", True, (237, 85, 101))
        score_text = FONT_MEDIUM.render(f"Final Score: {self.score}", True, (255,255,255))
        reward_text = FONT_MEDIUM.render(f"Total Reward: {self.cumulative_reward}", True, (255,255,255))
        # Position text in center of grid
        self.screen.blit(game_over_text, (self.grid_size * self.cell_size//2 - 120, self.grid_size * self.cell_size//2 - 70))
        self.screen.blit(score_text, (self.grid_size * self.cell_size//2 - 80, self.grid_size * self.cell_size//2 - 10))
        self.screen.blit(reward_text, (self.grid_size * self.cell_size//2 - 100, self.grid_size * self.cell_size//2 + 30))

        # Restart game button
        self.restart_btn = pygame.Rect(self.grid_size * self.cell_size//2 - 80, self.grid_size * self.cell_size//2 + 70, 160, 40)
        # Button hover effect
        btn_color = COLOR_BUTTON_HOVER if self.restart_btn.collidepoint(pygame.mouse.get_pos()) else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, self.restart_btn, border_radius=8)
        restart_text = FONT_SMALL.render("Restart Game", True, (255,255,255))
        restart_text_rect = restart_text.get_rect(center=self.restart_btn.center)
        self.screen.blit(restart_text, restart_text_rect)

    def reset_game(self):
        """Reset game state for restart (no Q-table change, reuse trained Q-table)"""
        self.snake.reset()
        self.food.reset()
        self.score = 0
        self.cumulative_reward = 0
        self.instant_reward = 0
        self.current_action = "None"
        self.game_over = False

    def battle_step(self):
        """Game play step: use trained Q-table to select optimal action"""
        # Get current game state
        current_state = self.agent.get_state(self.snake, self.food)
        # Select action (epsilon fixed to MIN_EPSILON: 95% exploit, 5% explore)
        action_idx = self.agent.choose_action(current_state)
        action = self.agent.actions[action_idx]
        # Execute action
        self.snake.change_direction(action)
        self.snake.move()
        # Calculate reward and check game over
        self.calculate_reward()
        if self.snake.check_collision():
            self.game_over = True
        # Update current action for UI display
        action_names = ["Up", "Down", "Left", "Right"]
        self.current_action = action_names[action_idx]

    def run(self):
        """Main game loop: mouse interaction only (no keyboard operation)"""
        while True:
            # Get mouse state (position + click)
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = pygame.mouse.get_pressed()

            # Event handling: only window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Start screen: click Game Start to enter game play
            if self.start_screen:
                self.draw_start_screen()
                if mouse_click[0]:  # Left mouse button click
                    if self.game_start_btn.collidepoint(mouse_pos):
                        self.start_screen = False
                        self.reset_game()
            else:
                # Game play screen: render game + execute game step
                self.draw_gradient_bg()
                self.draw_grid()
                if not self.game_over:
                    self.battle_step()
                # Draw all game elements
                self.draw_snake()
                self.draw_food()
                self.draw_info_panel()

                # Game over: draw game over screen + restart logic
                if self.game_over:
                    self.draw_game_over_screen()
                    if mouse_click[0]:  # Left mouse button click
                        if self.restart_btn.collidepoint(mouse_pos):
                            self.reset_game()

            # Update game window, control FPS with battle FPS
            pygame.display.flip()
            self.clock.tick(BATTLE_FPS)

# -------------------------- Main Program Entry --------------------------
if __name__ == "__main__":
    # Initialize game and start main loop
    game = SnakeGame()
    game.run()
