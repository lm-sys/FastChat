"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time
import re

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    BLIND_MODE_INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    enable_text,
    disable_text,
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
anony_names = ["", ""]
models = []


def set_global_vars_anony(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_


def load_demo_side_by_side_anony(models_, url_params):
    global models
    models = models_

    states = [None] * num_sides
    selector_updates = [
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
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

    gr.Info(
        "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."
    )
    if ":" not in model_selectors[0]:
        for i in range(5):
            names = (
                "### Model A: " + states[0].model_name,
                "### Model B: " + states[1].model_name,
            )
            # yield names + ("",) + (disable_btn,) * 4
            yield names + (disable_text,) + (disable_btn,) * 5
            time.sleep(0.1)
    else:
        names = (
            "### Model A: " + states[0].model_name,
            "### Model B: " + states[1].model_name,
        )
        # yield names + ("",) + (disable_btn,) * 4
        yield names + (disable_text,) + (disable_btn,) * 5


def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    ):
        yield x


def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    ):
        yield x


def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    ):
        yield x


def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (anony). ip: {get_ip(request)}")
    for x in vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    ):
        yield x


def regenerate(state0, state1, request: gr.Request):
    logger.info(f"regenerate (anony). ip: {get_ip(request)}")
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
    logger.info(f"clear_history (anony). ip: {get_ip(request)}")
    sandbox_states = [sandbox_state0, sandbox_state1]
    sandbox_state0["enabled_round"] = 0
    sandbox_state1["enabled_round"] = 0
    return (
        sandbox_states
        + [None] * num_sides
        + [None] * num_sides
        + anony_names
        + [enable_text]
        + [invisible_btn] * 4
        + [disable_btn] * 2
        + [""]
        + [enable_btn]
    )

def clear_sandbox_components(*components):
    updates = []
    for component in components:
        updates.append(gr.update(value="", visible=False))
    return updates

def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (anony). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )


SAMPLING_WEIGHTS = {'gpt-3.5-turbo':0.5,'gpt-4o-mini':0.5}

# target model sampling weights will be boosted.
BATTLE_TARGETS = {}

BATTLE_STRICT_TARGETS = {}

ANON_MODELS = []

SAMPLING_BOOST_MODELS = []

# outage models won't be sampled.
OUTAGE_MODELS = []


def get_sample_weight(model, outage_models, sampling_weights, sampling_boost_models=[]):
    if model in outage_models:
        return 0
    weight = sampling_weights.get(model, 0)
    if model in sampling_boost_models:
        weight *= 5
    return weight


def is_model_match_pattern(model, patterns):
    flag = False
    for pattern in patterns:
        pattern = pattern.replace("*", ".*")
        if re.match(pattern, model) is not None:
            flag = True
            break
    return flag


def get_battle_pair(
    models, battle_targets, outage_models, sampling_weights, sampling_boost_models
):
    if len(models) == 1:
        return models[0], models[0]

    model_weights = []
    for model in models:
        weight = get_sample_weight(
            model, outage_models, sampling_weights, sampling_boost_models
        )
        model_weights.append(weight)
    total_weight = np.sum(model_weights)
    
    model_weights = model_weights / total_weight
    # print(models)
    # print(model_weights)
    chosen_idx = np.random.choice(len(models), p=model_weights)
    chosen_model = models[chosen_idx]
    # for p, w in zip(models, model_weights):
    #     print(p, w)

    rival_models = []
    rival_weights = []
    for model in models:
        if model == chosen_model:
            continue
        if model in ANON_MODELS and chosen_model in ANON_MODELS:
            continue
        if chosen_model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(model, BATTLE_STRICT_TARGETS[chosen_model]):
                continue
        if model in BATTLE_STRICT_TARGETS:
            if not is_model_match_pattern(chosen_model, BATTLE_STRICT_TARGETS[model]):
                continue
        weight = get_sample_weight(model, outage_models, sampling_weights)
        if (
            weight != 0
            and chosen_model in battle_targets
            and model in battle_targets[chosen_model]
        ):
            # boost to 20% chance
            weight = 0.5 * total_weight / len(battle_targets[chosen_model])
        rival_models.append(model)
        rival_weights.append(weight)
    # for p, w in zip(rival_models, rival_weights):
    #     print(p, w)
    rival_weights = rival_weights / np.sum(rival_weights)
    rival_idx = np.random.choice(len(rival_models), p=rival_weights)
    rival_model = rival_models[rival_idx]

    swap = np.random.randint(2)
    if swap == 0:
        return chosen_model, rival_model
    else:
        return rival_model, chosen_model


def add_text(
    #state0, state1, model_selector0, model_selector1, text, request: gr.Request
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    sandbox_states = [sandbox_state0, sandbox_state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair(
            models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
        states = [
            State(model_left),
            State(model_right),
        ]

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
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    # turn on moderation in battle mode
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[0].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list, do_moderation=True)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
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
            + [""]
        )
    # add snadbox instructions if enabled
    if sandbox_state0['enable_sandbox'] and sandbox_state0['enabled_round'] == 0:
        text = f"> {sandbox_state0['sandbox_instruction']}\n\n" + text
        sandbox_state0['enabled_round'] += 1
        sandbox_state1['enabled_round'] += 1

    text = text[:BLIND_MODE_INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + sandbox_states
        + [""]
        + [
            disable_btn,
        ]
        * 6
        + [hint_msg]
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
    logger.info(f"bot_response_multi (anony). ip: {get_ip(request)}")

    if state0 is None or state0.skip_next:
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
            "llava-v1.6-34b",
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


def build_side_by_side_ui_anony(models):
    notice_markdown = f"""
# ⚔️  Chatbot Arena (formerly LMSYS): Free AI Chat to Compare & Test Best AI Chatbots
[Blog](https://blog.lmarena.ai/blog/2023/arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2403.04132) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/6GXcFg3TH8) | [Kaggle Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

{SURVEY_LINK}

## 📣 News
- Chatbot Arena now supports images in beta. Check it out [here](https://lmarena.ai/?vision).

## 📜 How It Works
- **Blind Test**: Ask any question to two anonymous AI chatbots (ChatGPT, Gemini, Claude, Llama, and more).
- **Vote for the Best**: Choose the best response. You can keep chatting until you find a winner.
- **Play Fair**: If AI identity reveals, your vote won't count.

## 🏆 Chatbot Arena LLM [Leaderboard](https://lmarena.ai/leaderboard)
- Backed by over **1,000,000+** community votes, our platform ranks the best LLM and AI chatbots. Explore the top AI models on our LLM [leaderboard](https://lmarena.ai/leaderboard)!

## 👇 Chat now!
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots: list[gr.Chatbot | None] = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anony"):
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
                        elem_id="chatbot",
                        height=650,
                        show_copy_button=True,
                        latex_delimiters=[
                            {"left": "$", "right": "$", "display": False},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": r"\(", "right": r"\)", "display": False},
                            {"left": r"\[", "right": r"\]", "display": True},
                        ],
                    )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(
                        anony_names[i], elem_id="model_selector_md"
                    )
        with gr.Row():
            slow_warning = gr.Markdown("")

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
    
        # chatbox sandbox global config
    with gr.Group():
        with gr.Row():
            enable_sandbox_checkbox = gr.Checkbox(value=False, label="Enable Sandbox", interactive=True)
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
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
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
            value=2000,
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
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors
        + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, send_btn],
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
        sandbox_states
        + states
        + chatbots
        + model_selectors
        + [textbox]
        + btn_list
        + [slow_warning]
        + [send_btn],
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
    const captureElement = document.querySelector('#share-region-anony');
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

    textbox.submit(
        add_text,
        states + model_selectors + sandbox_states + [textbox],
        states + chatbots + sandbox_states + [textbox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens] + sandbox_states,
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
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
