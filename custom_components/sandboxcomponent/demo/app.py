
import gradio as gr
from gradio_sandboxcomponent import SandboxComponent


example = SandboxComponent().example_value()


with gr.Blocks() as demo:
    with gr.Tab("My iFrame"):
        with gr.Row():
            gr.Markdown("## Baidu iFrame")
        with gr.Row():
            SandboxComponent(
                label="iFrame Example",
                value=("https://www.baidu.com/", "Hello World"),
                show_label=True)


if __name__ == "__main__":
    demo.launch()
