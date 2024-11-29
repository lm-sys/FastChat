import os
import gradio as gr
import requests
import base64
import io
import hashlib
from PIL import Image
import datetime
import json

from fastchat.constants import LOGDIR
from fastchat.utils import upload_image_file_to_gcs

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
API_BASE = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/{model}/text_to_image"
DUMMY_MODELS = ["stable-diffusion-3p5-medium", 
                "stable-diffusion-3p5-large",
                "stable-diffusion-3p5-large-turbo",
                "flux-1-dev-fp8",
                "flux-1-schnell-fp8",
                ]

def get_conv_log_filename():
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json"
    return os.path.join(LOGDIR, f"txt2img-{conv_log_filename}")

def generate_image(model, prompt):
    """Generate image from text prompt using Fireworks API"""
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json", 
        "Accept": "image/jpeg",
    }
    
    data = {
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "guidance_scale": 4.5,
        "num_inference_steps": 3,
    }

    api_url = API_BASE.format(model=model)

    try:
        response = requests.post(api_url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Failed to generate image: {response.status_code} {response.text}")
        
        image_bytes = response.content
        image = Image.open(io.BytesIO(image_bytes))
            
    except requests.exceptions.RequestException as e:
        return f"Error generating image: {str(e)}"

    log_filename = get_conv_log_filename()
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    image_filename = f"{image_hash}.png"
    upload_image_file_to_gcs(image, image_filename)

    with open(log_filename, "a") as f:
        data = {
            "model": model,
            "prompt": prompt,
            "image_filename": image_filename
        }
        f.write(json.dumps(data) + "\n")

    return image

def generate_image_multi(model_left, model_right, prompt):
    images = []
    for model in [model_left, model_right]:
        images.append(generate_image(model, prompt))

    return images


# Create Gradio interface
with gr.Blocks(title="Text to Image Generator") as demo:
    gr.Markdown("# Text to Image Generator")
    gr.Markdown("Enter a text prompt to generate an image")
    
    
    num_sides = 2
    model_selectors = [None] * num_sides

    with gr.Column():        
        with gr.Group():
            with gr.Row():
                for i in range(num_sides):
                    model_selectors[i] = gr.Dropdown(
                        choices=DUMMY_MODELS,
                        value=DUMMY_MODELS[i] if DUMMY_MODELS else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )


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
            text_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                show_label=False
            )
            send_btn = gr.Button("Generate", variant="primary")
    
    # Handle generation
    send_btn.click(
        fn=generate_image_multi,
        inputs=model_selectors + [text_input],
        outputs=[output_left, output_right]
    )
    
    text_input.submit(
        fn=generate_image_multi,
        inputs=model_selectors + [text_input],
        outputs=[output_left, output_right]
    )

if __name__ == "__main__":
    demo.launch(share=True)