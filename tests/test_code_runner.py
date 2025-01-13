from fastchat.serve.sandbox.code_runner import extract_code_from_markdown, SandboxEnvironment, extract_inline_pip_install_commands

def test_vue_component_extraction():
    # Test markdown content with Vue component
    markdown_content = '''
Here's a Vue calculator component:

```typescript
<template>
  <div class="calculator">
    <input type="text" v-model="expression" @keyup.enter="calculate">
    <button @click="clear">C</button>
    <button @click="calculate">=</button>
    <div v-for="(btnRow, index) in buttons" :key="index" class="btn-row">
      <button v-for="btn in btnRow" :key="btn" @click="updateExpression(btn)">{{ btn }}</button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';

export default defineComponent({
  name: 'Calculator',
  setup() {
    const expression = ref('');

    const buttons = [
      ['7', '8', '9', '/'],
      ['4', '5', '6', '*'],
      ['1', '2', '3', '-'],
      ['.', '0', '=', '+']
    ];

    const updateExpression = (value: string) => {
      expression.value += value;
    };

    const clear = () => {
      expression.value = '';
    };

    const calculate = () => {
      try {
        expression.value = eval(expression.value);
      } catch {
        expression.value = 'Error';
      }
    };

    return {
      expression,
      buttons,
      updateExpression,
      clear,
      calculate
    };
  },
});
</script>

<style scoped>
.calculator {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 5px;
  padding: 10px;
}

.btn-row {
  display: flex;
  justify-content: space-between;
}
</style>
```
'''

    # Extract code and verify results
    result = extract_code_from_markdown(markdown_content)
    assert result is not None, "Failed to extract code from markdown"
    
    code, code_lang, dependencies, env = result
    
    # Test code extraction
    assert '<template>' in code, "Template section not found in extracted code"
    assert '<script lang="ts">' in code, "Script section not found in extracted code"
    assert '<style scoped>' in code, "Style section not found in extracted code"
    
    # Test language detection
    assert code_lang == 'typescript', "TypeScript not detected in Vue component"
    
    # Test environment detection
    assert env == SandboxEnvironment.VUE, "Vue environment not detected"
    
    # Test dependency extraction
    npm_deps = dependencies[1]  # npm dependencies are second in tuple
    assert 'vue' in npm_deps, "Vue dependency not detected"

def test_vue_component_typescript_detection():
    # Test specific TypeScript patterns in Vue component
    markdown_content = '''
```vue
<template>
  <div class="calculator">
    <h1>Simple Calculator</h1>
    <div class="display">{{ current || "0" }}</div>
    <div class="buttons">
      <button @click="clear">C</button>
      <button @click="sign">±</button>
      <button @click="percent">%</button>
      <button @click="append(' / ')">÷</button>
      <button @click="append('7')">7</button>
      <button @click="append('8')">8</button>
      <button @click="append('9')">9</button>
      <button @click="append(' * ')">×</button>
      <button @click="append('4')">4</button>
      <button @click="append('5')">5</button>
      <button @click="append('6')">6</button>
      <button @click="append(' - ')">-</button>
      <button @click="append('1')">1</button>
      <button @click="append('2')">2</button>
      <button @click="append('3')">3</button>
      <button @click="append(' + ')">+</button>
      <button @click="append('0')">0</button>
      <button @click="append('.')">.</button>
      <button @click="calculate('=')">=</button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';

export default defineComponent({
  name: 'Calculator',
  setup() {
    const current = ref('');

    const append = (value: string) => {
      current.value += value;
    };

    const clear = () => {
      current.value = '';
    };

    const sign = () => {
      try {
        current.value = String(eval(current.value) * -1);
      } catch (e) {
        current.value = "Error";
      }
    };

    const percent = () => {
      try {
        current.value = String(eval(current.value) / 100);
      } catch (e) {
        current.value = "Error";
      }
    };

    const calculate = (value: string) => {
      try {
        current.value = String(eval(current.value));
      } catch (e) {
        current.value = "Error";
      }
    };

    return { current, append, clear, sign, percent, calculate };
  },
});
</script>

<style scoped>
.calculator {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.display {
  margin-bottom: 10px;
  padding: 10px;
  background-color: #f2f2f2;
  width: 210px;
  text-align: right;
}

.buttons {
  display: grid;
  grid-template-columns: repeat(4, 50px);
  gap: 10px;
}

button {
  padding: 10px;
  border: none;
  background-color: #e4e4e4;
  cursor: pointer;
}

button:hover {
  background-color: #d4d4d4;
}
</style>
```
'''
    result = extract_code_from_markdown(markdown_content)
    assert result is not None
    code, code_lang, (python_packages, npm_packages), sandbox_env_name = result
    print("Code language:", code_lang)
    print("NPM packages:", npm_packages)
    print("Sandbox environment:", sandbox_env_name)
    assert code_lang == 'typescript', "TypeScript not detected in Vue component with explicit TS patterns" 
    assert sandbox_env_name == SandboxEnvironment.VUE, "Vue environment not detected"
    print(npm_packages)
    assert 'vue' in npm_packages, "Vue dependency not detected"

def test_pygame_code_extraction():
    # Test markdown content with Pygame code
    markdown_content = '''
Here's a Ping Pong game in Pygame:

```python
import pygame
import random

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ping Pong Game")

# Game variables
player_score = 0
opponent_score = 0
player_speed = 0
opponent_speed = 6
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_dx = 5 * random.choice((1, -1))
ball_dy = 5 * random.choice((1, -1))
player_y = SCREEN_HEIGHT // 2 - 50
opponent_y = SCREEN_HEIGHT // 2 - 50

# Draw elements on the screen
def draw_elements():
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (20, player_y, 10, 100))
    pygame.draw.rect(screen, WHITE, (770, opponent_y, 10, 100))
    pygame.draw.ellipse(screen, WHITE, (ball_x, ball_y, 15, 15))
    font = pygame.font.Font(None, 36)
    player_text = font.render("Player: " + str(player_score), True, WHITE)
    screen.blit(player_text, (50, 20))
    opponent_text = font.render("Opponent: " + str(opponent_score), True, WHITE)
    screen.blit(opponent_text, (550, 20))

# Update the game state
def update_state():
    global ball_x, ball_y, ball_dx, ball_dy, player_score, opponent_score

    ball_x += ball_dx
    ball_y += ball_dy

    # Ball collision with top and bottom walls
    if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - 15:
        ball_dy = -ball_dy

    # Ball collision with player and opponent
    if ball_x <= 30 and player_y < ball_y < player_y + 100:
        ball_dx = -ball_dx
    elif ball_x >= 755 and opponent_y < ball_y < opponent_y + 100:
        ball_dx = -ball_dx
    elif ball_x <= 0:
        opponent_score += 1
        reset_ball()
    elif ball_x >= SCREEN_WIDTH - 15:
        player_score += 1
        reset_ball()

# Reset the ball position
def reset_ball():
    global ball_x, ball_y, ball_dx, ball_dy
    ball_x = SCREEN_WIDTH // 2
    ball_y = SCREEN_HEIGHT // 2
    ball_dx = 5 * random.choice((1, -1))
    ball_dy = 5 * random.choice((1, -1))

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] and player_y > 0:
        player_y -= 5
    if keys[pygame.K_DOWN] and player_y < SCREEN_HEIGHT - 100:
        player_y += 5

    if opponent_y + 50 < ball_y:
        opponent_y += opponent_speed
    elif opponent_y + 50 > ball_y:
        opponent_y -= opponent_speed

    draw_elements()
    update_state()

    pygame.display.update()
    clock.tick(60)

pygame.quit()
```
'''

    # Extract code and verify results
    result = extract_code_from_markdown(markdown_content)
    assert result is not None, "Failed to extract code from markdown"
    
    code, code_lang, dependencies, env = result
    
    # Test code extraction
    assert 'import pygame' in code, "Pygame import not found in extracted code"
    assert 'pygame.init()' in code, "Pygame initialization not found in extracted code"
    assert 'pygame.display.set_mode' in code, "Screen setup not found in extracted code"
    
    # Test language detection
    assert code_lang == 'python', "Python not detected as language"
    
    # Test environment detection
    assert env == SandboxEnvironment.PYGAME, "Pygame environment not detected"
    
    # Test dependency extraction
    python_deps = dependencies[0]  # python dependencies are first in tuple
    assert 'pygame' in python_deps, "Pygame dependency not detected"
    assert 'random' not in python_deps, "Standard library module incorrectly included as dependency"

def test_extract_inline_pip_install_commands():
    from fastchat.serve.sandbox.code_runner import extract_inline_pip_install_commands

    # Test code with various pip install formats
    test_code = """
# Regular imports
import numpy as np
import pandas as pd

# pip install numpy pandas
x = np.array([1, 2, 3])

!pip install scikit-learn>=0.24.0
from sklearn import metrics

# pip3 install -r requirements.txt tensorflow
model = tf.keras.Sequential()

!python -m pip install --upgrade torch
import torch

# Some regular code
def my_function():
    pass
"""

    expected_packages = ['numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch']
    expected_code = """
# Regular imports
import numpy as np
import pandas as pd

x = np.array([1, 2, 3])

from sklearn import metrics

model = tf.keras.Sequential()

import torch

# Some regular code
def my_function():
    pass
"""

    packages, cleaned_code = extract_inline_pip_install_commands(test_code)
    
    # Test that all expected packages are found
    assert sorted(packages) == sorted(expected_packages), f"Expected {expected_packages}, but got {packages}"
    
    # Test that cleaned code matches expected code
    assert cleaned_code.strip() == expected_code.strip(), f"Expected:\n{expected_code}\nGot:\n{cleaned_code}"

    # Test with no pip install commands
    code_without_pip = """
import numpy as np
def test():
    return np.sum([1, 2, 3])
"""
    packages, cleaned_code = extract_inline_pip_install_commands(code_without_pip)
    assert len(packages) == 0, f"Expected no packages, but got {packages}"
    assert cleaned_code.strip() == code_without_pip.strip(), "Code without pip commands should remain unchanged"

    # Test with only pip install commands
    only_pip_commands = """
# pip install numpy
!pip install pandas
# pip3 install -r requirements.txt tensorflow
!python -m pip install torch
"""
    packages, cleaned_code = extract_inline_pip_install_commands(only_pip_commands)
    assert sorted(packages) == sorted(['numpy', 'pandas', 'tensorflow', 'torch']), f"Expected ['numpy', 'pandas', 'tensorflow', 'torch'], but got {packages}"
    assert cleaned_code.strip() == "", "Code with only pip commands should result in empty string"

# test_vue_component_typescript_detection()
# test_vue_component_extraction()
# test_pygame_code_extraction()
test_extract_inline_pip_install_commands()