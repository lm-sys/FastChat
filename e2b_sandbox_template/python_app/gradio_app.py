import gradio as gr
import time

def test(x):
    time.sleep(4)
    return x

gr.Interface(test, "textbox", "textbox").queue().launch(
    root_path="/gradio",
    server_port=7860,
)