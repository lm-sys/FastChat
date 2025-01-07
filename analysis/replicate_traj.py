import json
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os

def parse_timestamp(ts_str):
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()

def replay_interactions(json_file):
    # Read interaction data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Setup webdriver
    driver = webdriver.Chrome()
    actions = ActionChains(driver)
    
    # Create screenshots directory
    screenshots_dir = "interaction_screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    try:
        # Navigate to the sandbox URL
        driver.get(data['sandboxUrl'])
        
        # Initial pause to let page load
        time.sleep(2)
        
        # Take initial screenshot
        driver.save_screenshot(os.path.join(screenshots_dir, "000_initial.png"))
        
        # Process each interaction
        prev_time = None
        for idx, interaction in enumerate(data['userInteractionRecords'], 1):
            current_time = parse_timestamp(interaction['time'])
            
            # Wait appropriate time between actions
            if prev_time is not None:
                time_diff = current_time - prev_time
                if time_diff > 0:
                    time.sleep(time_diff)
            
            # Replay the interaction
            if interaction['type'] == 'resize':
                driver.set_window_size(interaction['width'], interaction['height'])
            
            elif interaction['type'] == 'scroll':
                driver.execute_script(
                    f"window.scrollTo({interaction['scrollLeft']}, {interaction['scrollTop']});"
                )
            
            elif interaction['type'] == 'click':
                actions.move_by_offset(interaction['x'], interaction['y']).click().perform()
                actions.reset_actions()
            
            elif interaction['type'] == 'keydown':
                actions.send_keys(interaction['key']).perform()
                actions.reset_actions()
            
            # Take screenshot after action
            screenshot_name = f"{idx:03d}_{interaction['type']}.png"
            driver.save_screenshot(os.path.join(screenshots_dir, screenshot_name))
            
            prev_time = current_time
        
        # Final pause and screenshot
        time.sleep(2)
        driver.save_screenshot(os.path.join(screenshots_dir, "final.png"))
        
    finally:
        driver.quit()
        print(f"Screenshots saved in {screenshots_dir}/")

if __name__ == "__main__":
    replay_interactions('userInteractions.json')
