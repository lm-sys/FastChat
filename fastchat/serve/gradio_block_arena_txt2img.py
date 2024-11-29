import os
import gradio as gr
import requests
import time
import io
import hashlib
from PIL import Image
import datetime
import json
import random

from fastchat.constants import LOGDIR
from fastchat.utils import upload_image_file_to_gcs
from fastchat.serve.gradio_web_server import enable_btn, disable_btn

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
API_BASE = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/{model}/text_to_image"
DUMMY_MODELS = ["stable-diffusion-3p5-medium", 
                "stable-diffusion-3p5-large",
                "stable-diffusion-3p5-large-turbo",
                "flux-1-dev-fp8",
                "flux-1-schnell-fp8",
                ]
ANONY_NAMES = ["", ""]

class State:
    def __init__(self, model_name):
        self.model_name = model_name
        self.prompt = ""
        self.image_filename = ""
        self.generated_image = None

def get_conv_log_filename():
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
    return os.path.join(LOGDIR, f"txt2img-{conv_log_filename}")

def get_battle_pair(models):
    return random.sample(models, 2)

def add_text(state_left, state_right, prompt):
    if state_left is None or state_right is None:
       models = get_battle_pair(DUMMY_MODELS)
       state_left = State(models[0])
       state_right = State(models[1])

    state_left.prompt = prompt
    state_right.prompt = prompt
    return state_left, state_right

def generate_image(state):
    """Generate image from text prompt using Fireworks API"""
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json", 
        "Accept": "image/jpeg",
    }
    
    data = {
        "prompt": state.prompt,
        "aspect_ratio": "16:9",
        "guidance_scale": 4.5,
        "num_inference_steps": 3,
    }

    api_url = API_BASE.format(model=state.model_name)

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to generate image: {response.status_code} {response.text}")
        
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
        state.generated_image = image
            
    except requests.exceptions.RequestException as e:
        return f"Error generating image: {str(e)}"

    log_filename = get_conv_log_filename()
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    image_filename = f"{image_hash}.png"
    upload_image_file_to_gcs(image, image_filename)
    state.image_filename = image_filename

    with open(log_filename, "a") as f:
        data = {
            "model": state.model_name,
            "prompt": state.prompt,
            "image_filename": image_filename
        }
        f.write(json.dumps(data) + "\n")

    return image

def generate_image_multi(state_left, state_right):
    # Randomly sample two different models
    images = []
    states = [state_left, state_right]
    for i in range(2):
        images.append(generate_image(states[i]))

    return images

def flash_buttons():
    btn_updates = [
        [disable_btn] * 4,
        [enable_btn] * 4,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.3)

def reveal_models(state_left, state_right):
    return [f"Model A: {state_left.model_name}", f"Model B: {state_right.model_name}"]

# Create Gradio interface
with gr.Blocks(title="Text to Image Generator") as demo:
    gr.Markdown("# Text to Image Generator")
    gr.Markdown("Enter a text prompt to generate an image")
    
    num_sides = 2
    model_selectors = [None] * num_sides
    states = [gr.State() for _ in range(num_sides)]

    with gr.Column():
        with gr.Group():
            with gr.Row():
                output_left = gr.Image(
                    type="pil",
                    show_label=False
                )
                output_right = gr.Image(
                    type="pil",
                    show_label=False
                )

            with gr.Row():
                for i in range(num_sides):
                    model_selectors[i] = gr.Markdown(
                        value=ANONY_NAMES[i],
                        show_label=False,
                    )

        with gr.Row():
            left_btn = gr.Button(
                value="Left",
                interactive=False,
                visible=False
            )
            tie_btn = gr.Button(
                value="Tie",
                interactive=False,
                visible=False
            )
            right_btn = gr.Button(
                value="Right",
                interactive=False,
                visible=False
            )
            idk_btn = gr.Button(
                value="IDK",
                interactive=False,
                visible=False
            )
            
        with gr.Row():
            text_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                show_label=False
            )
            send_btn = gr.Button("Generate", variant="primary")

    # Handle generation
    gen_output = send_btn.click(
        fn=add_text,
        inputs=states + [text_input],
        outputs=states
    ).then(
        fn=generate_image_multi,
        inputs=states,
        outputs=[output_left, output_right]
    ).then(
        fn=flash_buttons,
        outputs=[left_btn, tie_btn, right_btn, idk_btn]
    )
    
    text_input.submit(
        fn=add_text,
        inputs=states + [text_input],
        outputs=states
    ).then(
        fn=generate_image_multi,
        inputs=states,
        outputs=[output_left, output_right]
    ).then(
        fn=flash_buttons,
        outputs=[left_btn, tie_btn, right_btn, idk_btn]
    )

    # Handle voting buttons
    for btn in [left_btn, tie_btn, right_btn, idk_btn]:
        btn.click(
            fn=reveal_models,
            inputs=states,
            outputs=model_selectors
        )

if __name__ == "__main__":
    demo.launch(share=True)