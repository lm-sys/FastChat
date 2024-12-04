
import gradio as gr
from gradio_sandboxcomponent import SandboxComponent

example = SandboxComponent().example_value()


with gr.Blocks() as demo:
    with gr.Tab("Sandbox Demo"):
        with gr.Row():
            gr.Markdown("## Sandbox")
        with gr.Row():
            SandboxComponent(
                label="Sandbox Example",
                value=("https://www.gradio.app/", "Hello World"),
                show_label=True)


if __name__ == "__main__":
    demo.launch()
