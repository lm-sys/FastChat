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
    assert code_lang == 'typescript', "TypeScript not detected in Vue component with explicit TS patterns" 
    assert sandbox_env_name == SandboxEnvironment.VUE, "Vue environment not detected"
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

def test_determine_js_environment():
    from fastchat.serve.sandbox.code_runner import determine_js_environment, SandboxEnvironment

    # Test Vue SFC structure detection
    vue_sfc_code = '''
<template>
  <div>Hello Vue</div>
</template>
<script>
export default {
  name: 'HelloWorld'
}
</script>
'''
    assert determine_js_environment(vue_sfc_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue SFC structure"

    # Test Vue script setup detection
    vue_setup_code = '''
<template>
  <div>{{ message }}</div>
</template>
<script setup>
const message = 'Hello Vue'
</script>
'''
    assert determine_js_environment(vue_setup_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue script setup"

    # Test Vue directives detection
    vue_directives_code = '''
export default {
  template: `
    <div>
      <div v-if="show">Conditional</div>
      <div v-for="item in items">{{ item }}</div>
      <input v-model="text" />
      <button @click="handleClick">Click</button>
      <div :class="{ active: isActive }">Dynamic Class</div>
    </div>
  `
}
'''
    assert determine_js_environment(vue_directives_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue directives"

    # Test Vue Composition API detection
    vue_composition_code = '''
import { ref, computed, watch, onMounted } from 'vue'

export default {
  setup() {
    const count = ref(0)
    const doubled = computed(() => count.value * 2)
    watch(count, (newVal) => console.log(newVal))
    onMounted(() => console.log('mounted'))
    return { count, doubled }
  }
}
'''
    assert determine_js_environment(vue_composition_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue Composition API"

    # Test Vue Options API detection
    vue_options_code = '''
export default {
  data() {
    return { count: 0 }
  },
  computed: {
    doubled() { return this.count * 2 }
  },
  methods: {
    increment() { this.count++ }
  },
  watch: {
    count(newVal) { console.log(newVal) }
  }
}
'''
    assert determine_js_environment(vue_options_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue Options API"

    # Test React detection - should require both JSX syntax AND React dependency/imports
    react_code = '''
function App() {
  return (
    <div>
      <h1>Hello React</h1>
      <button onClick={() => alert('clicked')}>Click me</button>
    </div>
  )
}
'''
    # Without React dependency, should fallback to JavaScript
    assert determine_js_environment(react_code, []) == SandboxEnvironment.REACT, "Should detect React when JSX and react dependency are present"
    
    # With React dependency, should detect as React
    assert determine_js_environment(react_code, ['react']) == SandboxEnvironment.REACT, "Should detect React when JSX and react dependency are present"

    # With React import, should detect as React
    react_code_with_import = '''
import React from 'react';

function App() {
  return (
    <div>
      <h1>Hello React</h1>
      <button onClick={() => alert('clicked')}>Click me</button>
    </div>
  )
}
'''
    assert determine_js_environment(react_code_with_import, ['react']) == SandboxEnvironment.REACT, "Should detect React when import is present"

    # Test package-based detection
    vue_import_code = "import { createApp } from 'vue'"
    assert determine_js_environment(vue_import_code, ['vue']) == SandboxEnvironment.VUE, "Failed to detect Vue from imports"

    react_import_code = "import { useState } from 'react'"
    assert determine_js_environment(react_import_code, ['react']) == SandboxEnvironment.REACT, "Failed to detect React from imports"

    # Test fallback to JavaScript Code Interpreter
    plain_js_code = '''
function add(a, b) {
  return a + b
}
'''
    assert determine_js_environment(plain_js_code, []) == SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER, "Failed to fallback to JavaScript Code Interpreter"


    real_code = '''
<template>
  <div :class="['min-h-screen', darkMode ? 'dark bg-gray-900 text-white' : 'bg-gray-100 text-gray-900']">
    <div class="container mx-auto p-6">
      <header class="flex justify-between items-center mb-6">
        <h1 class="text-3xl font-bold">Vue Todo List</h1>
        <button
          @click="toggleDarkMode"
          class="px-4 py-2 rounded bg-blue-500 text-white hover:bg-blue-600"
        >
          Toggle Dark Mode
        </button>
      </header>

      <div class="mb-6">
        <form @submit.prevent="addTodo" class="flex gap-2">
          <input
            v-model="newTodo"
            type="text"
            placeholder="Enter a new todo"
            class="flex-grow px-4 py-2 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          <button
            type="submit"
            class="px-4 py-2 rounded bg-green-500 text-white hover:bg-green-600"
          >
            Add
          </button>
        </form>
      </div>

      <ul>
        <li
          v-for="(todo, index) in todos"
          :key="index"
          class="flex justify-between items-center p-4 mb-2 rounded bg-white dark:bg-gray-800 shadow"
        >
          <span>{{ todo }}</span>
          <button
            @click="removeTodo(index)"
            class="px-3 py-1 rounded bg-red-500 text-white hover:bg-red-600"
          >
            Remove
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: "App",
  data() {
    return {
      darkMode: false,
      newTodo: "",
      todos: [],
    };
  },
  methods: {
    toggleDarkMode() {
      this.darkMode = !this.darkMode;
    },
    addTodo() {
      if (this.newTodo.trim() !== "") {
        this.todos.push(this.newTodo.trim());
        this.newTodo = "";
      }
    },
    removeTodo(index) {
      this.todos.splice(index, 1);
    },
  },
};
</script>

<style>
/* Tailwind CSS should be included in your project for the styles to work */
@import "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css";
</style>
'''
    assert determine_js_environment(real_code, []) == SandboxEnvironment.VUE, "Failed to detect Vue environment in real code example"

def test_vue_in_html_detection():
    # Test HTML content with Vue.js integration
    html_with_vue = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vue 2048 Game</title>
    <script src="https://unpkg.com/vue@next"></script>
    <!-- Tailwind CSS CDN -->
    <link href="https://unpkg.com/tailwindcss@^2.0/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .game-container {
            max-width: 500px;
            margin: 30px auto;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        .cell {
            width: 100px;
            height: 100px;
            display: flex;
            justify-content: center;
            align-items: center;
            font-weight: bold;
            font-size: 24px;
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    <div id="app" class="text-center">
        <h1 class="text-4xl font-bold mb-4">Vue 2048 Game</h1>
        <div class="game-container">
            <div v-for="(cell, index) in cells" :key="index" class="cell" :style="{ backgroundColor: cellColor(cell) }">
                {{ cell || '' }}
            </div>
        </div>
        <button @click="startGame" class="mt-4 px-4 py-2 bg-blue-500 text-white rounded">Start Game</button>
    </div>

    <script>
        const { createApp, reactive } = Vue;

        createApp({
            setup() {
                const state = reactive({
                    cells: Array(16).fill(0),
                });

                function startGame() {
                    state.cells = Array(16).fill(0);
                    addRandomCell();
                    addRandomCell();
                }

                function addRandomCell() {
                    let emptyCells = state.cells.map((cell, index) => cell === 0 ? index : null).filter(index => index !== null);
                    if(emptyCells.length) {
                        let randomCellIndex = emptyCells[Math.floor(Math.random() * emptyCells.length)];
                        state.cells[randomCellIndex] = Math.random() > 0.1 ? 2 : 4;
                    }
                }

                function cellColor(cell) {
                    const colors = {
                        0: '#f0f0f0',
                        2: '#eee4da',
                        4: '#ede0c8',
                        // Add more colors for different numbers
                    };
                    return colors[cell] || '#f0f0f0';
                }

                return { ...state, startGame, cellColor };
            },
        }).mount('#app');
    </script>
</body>
</html>'''

    # Extract code and verify results
    result = extract_code_from_markdown(f"```html\n{html_with_vue}\n```")
    assert result is not None, "Failed to extract code from markdown"
    
    code, code_lang, dependencies, env = result
    
    # Test code extraction
    assert '<!DOCTYPE html>' in code, "HTML doctype not found in extracted code"
    assert '<script src="https://unpkg.com/vue@next"></script>' in code, "Vue.js script import not found"
    assert 'createApp(' in code, "Vue createApp not found in code"
    
    # Test language detection
    assert code_lang == 'html', "HTML not detected as language"
    
    # Test environment detection
    assert env == SandboxEnvironment.HTML, "HTML environment not detected for Vue in HTML"
    
    # Test that Vue.js is detected in dependencies
    npm_deps = dependencies[1]  # npm dependencies are second in tuple
    assert 'vue' in npm_deps, "Vue dependency not detected in HTML with Vue.js"

def test_vue_calendar_component():
    # Test markdown content with Vue calendar component
    markdown_content = '''
```html
<template>
  <div class="p-4 max-w-md mx-auto">
    <!-- 日历头部 -->
    <div class="flex justify-between items-center mb-4">
      <button
        @click="changeMonth(-1)"
        class="text-white bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded"
      >
        Prev
      </button>
      <h2 class="text-lg font-bold">
        {{ currentYear }} - {{ currentMonth + 1 }}
      </h2>
      <button
        @click="changeMonth(1)"
        class="text-white bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded"
      >
        Next
      </button>
    </div>

    <!-- 星期标题 -->
    <div class="grid grid-cols-7 text-center font-bold mb-2">
      <div v-for="day in weekDays" :key="day" class="text-gray-700">
        {{ day }}
      </div>
    </div>

    <!-- 日期 -->
    <transition name="fade" mode="out-in">
      <div
        :key="currentKey"
        class="grid grid-cols-7 gap-2 transition-transform duration-300"
      >
        <!-- 前面补空白 -->
        <div
          v-for="_, index in firstDayOfMonth"
          :key="'empty-' + index"
        ></div>

        <!-- 日期 -->
        <div
          v-for="day in daysInMonth"
          :key="day"
          class="p-2 text-center rounded hover:bg-blue-100"
        >
          {{ day }}
        </div>
      </div>
    </transition>
  </div>
</template>

<script>
export default {
  data() {
    return {
      currentDate: new Date(),
      currentKey: 0, // 用于触发 Vue 的过渡效果
      weekDays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    };
  },
  computed: {
    currentYear() {
      return this.currentDate.getFullYear();
    },
    currentMonth() {
      return this.currentDate.getMonth();
    },
    daysInMonth() {
      const year = this.currentYear;
      const month = this.currentMonth;
      const days = [];
      const lastDay = new Date(year, month + 1, 0).getDate();
      for (let day = 1; day <= lastDay; day++) {
        days.push(day);
      }
      return days;
    },
    firstDayOfMonth() {
      // 获取当前月份第一天是星期几
      const firstDay = new Date(this.currentYear, this.currentMonth, 1);
      return firstDay.getDay();
    },
  },
  methods: {
    changeMonth(offset) {
      this.currentDate.setMonth(this.currentMonth + offset);
      this.currentKey++; // 更改 key 触发 Vue 的过渡效果
    },
  },
};
</script>

<style>
/* 自定义 Vue 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: all 0.3s ease;
}
.fade-enter {
  opacity: 0;
  transform: translateX(100%);
}
.fade-leave-to {
  opacity: 0;
  transform: translateX(-100%);
}
</style>
```
'''

    # Extract code and verify results
    result = extract_code_from_markdown(markdown_content)
    assert result is not None, "Failed to extract code from markdown"
    
    code, code_lang, dependencies, env = result

    assert code_lang == 'typescript', "TypeScript not detected as language"
    
    # Test environment detection
    assert env == SandboxEnvironment.VUE, "Vue environment not detected"
    
    # Test dependency extraction
    npm_deps = dependencies[1]  # npm dependencies are second in tuple
    assert 'vue' in npm_deps, "Vue dependency not detected"

def test_mermaid_diagram_html():
    # Test markdown content with Mermaid diagram HTML
    markdown_content = '''
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Software Arena Mermaid Diagram</title>
    <!-- Import Mermaid -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({startOnLoad:true});
    </script>
</head>
<body>

<!-- Mermaid diagram definition -->
<div class="mermaid">
graph TD;
    Agile -->|Informs| DevOps;
    DevOps --> CI[Continuous Integration];
    CI --> CD[Continuous Deployment];
    CI --> AutomatedTesting[Automated Testing];
    CD --> Feedback;
    AutomatedTesting --> Feedback;
    Feedback --> Agile;
</div>

</body>
</html>
```
'''

    # Extract code and verify results
    result = extract_code_from_markdown(markdown_content)
    assert result is not None, "Failed to extract code from markdown"
    
    code, code_lang, dependencies, env = result

    # Test language detection
    assert code_lang == 'html', "HTML not detected as language"
    
    # Test environment detection
    assert env == SandboxEnvironment.HTML, "HTML environment not detected"
    
    # Test dependency extraction
    npm_deps = dependencies[1]  # npm dependencies are second in tuple
    print(npm_deps)

    assert 'mermaid' in npm_deps, "Mermaid dependency not detected"

test_vue_in_html_detection()
test_mermaid_diagram_html()