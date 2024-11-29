import os
import gradio as gr
import requests
import base64
import io
from PIL import Image

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
API_BASE = "https://api.fireworks.ai/inference/v1/workflows/accounts/fireworks/models/{}/text_to_image"
DUMMY_MODELS = ["stable-diffusion-3p5-medium", 
                "stable-diffusion-3p5-large",
                "stable-diffusion-3p5-large-turbo",
                "flux-1-dev-fp8",
                "flux-1-schnell-fp8",
                ]

def generate_image(prompt, model):
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
    
        return image
        
    except requests.exceptions.RequestException as e:
        return f"Error generating image: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Text to Image Generator") as demo:
    gr.Markdown("# Text to Image Generator")
    gr.Markdown("Enter a text prompt to generate an image")
    
    with gr.Column():        
        with gr.Group():
            model_selector = gr.Dropdown(
                choices=DUMMY_MODELS,
                interactive=True,
                show_label=False,
                container=False,
            )
            image_output = gr.Image(
                label="Generated Image",
                type="pil"
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
        fn=generate_image,
        inputs=[text_input, model_selector],
        outputs=image_output
    )
    
    text_input.submit(
        fn=generate_image,
        inputs=[text_input, model_selector],
        outputs=image_output
    )

if __name__ == "__main__":
    demo.launch(share=True)