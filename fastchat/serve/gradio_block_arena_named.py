"""
Chatbot Arena (side-by-side) tab.
Users chat with two chosen models.
"""

import json
import time

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.serve.sandbox.code_runner import DEFAULT_SANDBOX_INSTRUCTIONS, SUPPORTED_SANDBOX_ENVIRONMENTS, create_chatbot_sandbox_state, on_click_run_code, update_sandbox_config
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False


def set_global_vars_named(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_named(models, url_params):
    states = [None] * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 64)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = [
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    ]

    return states + selector_updates


def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (named). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (named). ip: {get_ip(request)}")
    states = [state0, state1]
    if state0.regen_support and state1.regen_support:
        for i in range(num_sides):
            states[i].conv.update_last_message(None)
        return (
            states + [x.to_gradio_chatbot() for x in states] + [""] + [disable_btn] * 6
        )
    states[0].skip_next = True
    states[1].skip_next = True
    return states + [x.to_gradio_chatbot() for x in states] + [""] + [no_change_btn] * 6


def clear_history(sandbox_state0, sandbox_state1, request: gr.Request):
    logger.info(f"clear_history (named). ip: {get_ip(request)}")
    sandbox_states = [sandbox_state0, sandbox_state1]
    sandbox_state0["enabled_round"] = 0
    sandbox_state1["enabled_round"] = 0
    return (
        sandbox_states
        + [None] * num_sides
        + [None] * num_sides
        + [""]
        + [invisible_btn] * 4
        + [disable_btn] * 2
    )

def clear_sandbox_components(*components):
    updates = []
    for component in components:
        updates.append(gr.update(value="", visible=False))
    return updates

def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (named). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


def add_text(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (named). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]
    sandbox_states = [sandbox_state0, sandbox_state1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + sandbox_states
            + ["", None]
            + [
                no_change_btn,
            ]
            * 6
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[1].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list)
    if flagged:
        logger.info(f"violate moderation (named). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
             + sandbox_states
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
        )

    # add snadbox instructions if enabled
    if sandbox_state0['enable_sandbox'] and sandbox_state0['enabled_round'] == 0:
        text = f"> {sandbox_state0['sandbox_instruction']}\n\n" + text
        sandbox_state0['enabled_round'] += 1
        sandbox_state1['enabled_round'] += 1

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    return (
        states
        + [x.to_gradio_chatbot() for x in states]
         + sandbox_states
        + [""]
        + [
            disable_btn,
        ]
        * 6
    )


def bot_response_multi(
    state0,
    state1,
    temperature,
    top_p,
    max_new_tokens,
    sandbox_state0,
    sandbox_state1,
    request: gr.Request,
):
    logger.info(f"bot_response_multi (named). ip: {get_ip(request)}")

    if state0.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state0,
            state1,
            state0.to_gradio_chatbot(),
            state1.to_gradio_chatbot(),
        ) + (no_change_btn,) * 6
        return

    states = [state0, state1]
    gen = []
    for i in range(num_sides):
        gen.append(
            bot_response(
                states[i],
                temperature,
                top_p,
                max_new_tokens,
                sandbox_state0,
                request
            )
        )

    model_tpy = []
    for i in range(num_sides):
        token_per_yield = 1
        if states[i].model_name in [
            "gemini-pro",
            "gemma-1.1-2b-it",
            "gemma-1.1-7b-it",
            "phi-3-mini-4k-instruct",
            "phi-3-mini-128k-instruct",
            "snowflake-arctic-instruct",
        ]:
            token_per_yield = 30
        elif states[i].model_name in [
            "qwen-max-0428",
            "qwen-vl-max-0809",
            "qwen1.5-110b-chat",
        ]:
            token_per_yield = 7
        elif states[i].model_name in [
            "qwen2.5-72b-instruct",
            "qwen2-72b-instruct",
            "qwen-plus-0828",
            "qwen-max-0919",
            "llama-3.1-405b-instruct-bf16",
        ]:
            token_per_yield = 4
        model_tpy.append(token_per_yield)

    chatbots = [None] * num_sides
    iters = 0
    while True:
        stop = True
        iters += 1
        for i in range(num_sides):
            try:
                # yield fewer times if chunk size is larger
                if model_tpy[i] == 1 or (iters % model_tpy[i] == 1 or iters < 3):
                    ret = next(gen[i])
                    states[i], chatbots[i] = ret[0], ret[1]
                stop = False
            except StopIteration:
                pass
        yield states + chatbots + [disable_btn] * 6
        if stop:
            break


def flash_buttons():
    btn_updates = [
        [disable_btn] * 4 + [enable_btn] * 2,
        [enable_btn] * 6,
    ]
    for i in range(4):
        yield btn_updates[i % 2]
        time.sleep(0.3)


def build_side_by_side_ui_named(models):
    notice_markdown = f"""
# ⚔️  Chatbot Arena (formerly LMSYS): Free AI Chat to Compare & Test Best AI Chatbots
[Blog](https://blog.lmarena.ai/blog/2023/arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2403.04132) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/6GXcFg3TH8) | [Kaggle Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

{SURVEY_LINK}

## 📜 How It Works
- Ask any question to two chosen models (e.g., ChatGPT, Gemini, Claude, Llama) and vote for the better one!
- You can chat for multiple turns until you identify a winner.

## 👇 Choose two models to compare
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots: list[gr.Chatbot | None] = [None] * num_sides

    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
        with gr.Row():
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(models)} models", open=False
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            for i in range(num_sides):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    chatbots[i] = gr.Chatbot(
                        label=label,
                        elem_id=f"chatbot",
                        height=650,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                    )

    # sandbox states and components
    sandbox_states: list[gr.State | None] = [None for _ in range(num_sides)]
    sandboxes_components: list[tuple[
        gr.Markdown, # sandbox_output
        SandboxComponent,  # sandbox_ui
        gr.Code, # sandbox_code
    ] | None] = [None for _ in range(num_sides)]

    hidden_components = []

    with gr.Group(visible=False) as sandbox_group:
        hidden_components.append(sandbox_group)
        with gr.Row(visible=False) as sandbox_row:
            hidden_components.append(sandbox_row)
            for chatbotIdx in range(num_sides):
                with gr.Column(scale=1, visible=False) as column:
                    sandbox_state = gr.State(create_chatbot_sandbox_state())
                    # Add containers for the sandbox output
                    sandbox_title = gr.Markdown(value=f"### Model {chatbotIdx + 1} Sandbox", visible=False)
                    with gr.Tab(label="Output", visible=False) as sandbox_output_tab:
                        sandbox_output = gr.Markdown(value="", visible=False)
                        sandbox_ui = SandboxComponent(
                            value=("", ""),
                            show_label=True,
                            visible=False,
                        )
                    with gr.Tab(label="Code", visible=False) as sandbox_code_tab:
                        sandbox_code = gr.Code(value="", interactive=False, visible=False)
                    sandbox_states[chatbotIdx] = sandbox_state
                    sandboxes_components[chatbotIdx] = (
                        sandbox_output,
                        sandbox_ui,
                        sandbox_code,
                    )
                    hidden_components.extend([column, sandbox_title, sandbox_output_tab, sandbox_code_tab])

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )


    with gr.Group():
        with gr.Row():
            enable_sandbox_checkbox = gr.Checkbox(
                value=False,
                label="Enable Sandbox",
                info="Run generated code in a remote sandbox",
                interactive=True,
            )
            sandbox_env_choice = gr.Dropdown(choices=SUPPORTED_SANDBOX_ENVIRONMENTS, label="Sandbox Environment", interactive=True, visible=False)
        with gr.Group():
            with gr.Accordion("Sandbox Instructions", open=False, visible=False) as sandbox_instruction_accordion:
                sandbox_instruction_textarea = gr.TextArea(
                    value='',visible=False
                )
        hidden_components.extend([sandbox_env_choice, sandbox_instruction_accordion, sandbox_instruction_textarea])

        sandbox_env_choice.change(
            fn=lambda env, enable: "" if not enable else DEFAULT_SANDBOX_INSTRUCTIONS[env],
            inputs=[sandbox_env_choice, enable_sandbox_checkbox],
            outputs=[sandbox_instruction_textarea]
        ).then(
            fn=update_sandbox_config,
            inputs=[
                enable_sandbox_checkbox,
                sandbox_env_choice,
                sandbox_instruction_textarea,
                *sandbox_states
            ],
            outputs=[*sandbox_states]
        )

        sandbox_instruction_textarea.change(
            fn=update_sandbox_config,
            inputs=[
                enable_sandbox_checkbox,
                sandbox_env_choice,
                sandbox_instruction_textarea,
                *sandbox_states
            ],
            outputs=[*sandbox_states]
        )

        # update sandbox global config
        enable_sandbox_checkbox.change(
            fn=lambda enable, env: "" if not enable else DEFAULT_SANDBOX_INSTRUCTIONS.get(env, ""),
            inputs=[enable_sandbox_checkbox, sandbox_env_choice],
            outputs=[sandbox_instruction_textarea]
        ).then(            
            fn=update_sandbox_config,
            inputs=[
                enable_sandbox_checkbox,
                sandbox_env_choice,
                sandbox_instruction_textarea,
                *sandbox_states
            ],
            outputs=[*sandbox_states]
        ).then(
            fn=lambda enable: [gr.update(visible=enable) for _ in hidden_components],
            inputs=[enable_sandbox_checkbox],
            outputs=hidden_components
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
            clear_history, 
            sandbox_states, 
            sandbox_states + states + chatbots + [textbox] + btn_list 
        ).then(
            clear_sandbox_components,
            inputs=[component for components in sandboxes_components for component in components],
            outputs=[component for components in sandboxes_components for component in components]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[sandbox_env_choice]
        )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    for i in range(num_sides):
        model_selectors[i].change(
            clear_history, 
            sandbox_states, 
            sandbox_states + states + chatbots + [textbox] + btn_list 
        ).then(
            clear_sandbox_components,
            inputs=[component for components in sandboxes_components for component in components],
            outputs=[component for components in sandboxes_components for component in components]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[sandbox_env_choice]
        )

    textbox.submit(
        add_text,
        states + model_selectors + sandbox_states + [textbox],
        states + chatbots + sandbox_states + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    ).then(
        lambda sandbox_state: gr.update(interactive=sandbox_state['enabled_round'] == 0),
        inputs=[sandbox_states[0]],
        outputs=[sandbox_env_choice]
    )
    send_btn.click(
        add_text,
        states + model_selectors + sandbox_states + [textbox],
        states + chatbots + sandbox_states + [textbox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
        ).then(
        lambda sandbox_state: gr.update(interactive=sandbox_state['enabled_round'] == 0),
        inputs=[sandbox_states[0]],
        outputs=[sandbox_env_choice]
    )

    for chatbotIdx in range(num_sides):
        chatbot = chatbots[chatbotIdx]
        state = states[chatbotIdx]
        sandbox_state = sandbox_states[chatbotIdx]
        sandbox_components = sandboxes_components[chatbotIdx]

        # trigger sandbox run
        chatbot.select(
            fn=on_click_run_code,
            inputs=[state, sandbox_state, *sandbox_components],
            outputs=[*sandbox_components],
        )

    return states + model_selectors
