
# `gradio_sandboxcomponent`
<a href="https://pypi.org/project/gradio_sandboxcomponent/" target="_blank"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/gradio_sandboxcomponent"></a>  

Gradio library for easily interacting with remote sandbox.

## Installation

```bash
pip install gradio_sandboxcomponent
```

## Usage

```python

from typing import Any
import gradio as gr
from gradio_sandboxcomponent import SandboxComponent

example = SandboxComponent().example_value()


with gr.Blocks() as demo:
    with gr.Tab("Sandbox Demo"):
        with gr.Row():
            gr.Markdown("## Sandbox")
        with gr.Row():
            sandboxUrl = gr.Textbox(
                label="Sandbox URL",
                value='https://www.gradio.app/',
                placeholder="Enter sandbox URL",
                lines=1,
                show_label=True,
                elem_id=None,
                elem_classes=None,
                key=None,
            )
            sandboxInteractions = gr.Textbox(
                label="Sandbox Interactions",
                value='[]',
                placeholder="Enter sandbox interactions",
                lines=1,
                show_label=True,
                elem_id=None,
                elem_classes=None,
                key=None,
            )
        with gr.Row():
            sandbox = SandboxComponent(
                label="Sandbox Example",
                value=("https://www.gradio.app/", True, []),
                show_label=True)

        def update_outputs(sandboxData: tuple[str, list[Any]]):
            sandboxUrl, _, sandboxInteractions = sandboxData
            print(
                "UPDATING",
                sandboxData
            )
            return sandboxUrl, str(sandboxInteractions)

        sandbox.change(
            update_outputs,
            inputs=[sandbox],
            outputs=[sandboxUrl, sandboxInteractions]
        )

if __name__ == "__main__":
    demo.launch()

```

## `SandboxComponent`

### Initialization

<table>
<thead>
<tr>
<th align="left">name</th>
<th align="left" style="width: 25%;">type</th>
<th align="left">default</th>
<th align="left">description</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><code>value</code></td>
<td align="left" style="width: 25%;">

```python
tuple[str, bool, list[Any]] | Callable | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">url string and interactions.</td>
</tr>

<tr>
<td align="left"><code>label</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">the label for this component, displayed above the component if `show_label` is `True` and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component corresponds to.</td>
</tr>

<tr>
<td align="left"><code>every</code></td>
<td align="left" style="width: 25%;">

```python
Timer | float | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">Continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.</td>
</tr>

<tr>
<td align="left"><code>inputs</code></td>
<td align="left" style="width: 25%;">

```python
Component | Sequence[Component] | set[Component] | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.</td>
</tr>

<tr>
<td align="left"><code>show_label</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">if True, will display label.</td>
</tr>

<tr>
<td align="left"><code>scale</code></td>
<td align="left" style="width: 25%;">

```python
int | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.</td>
</tr>

<tr>
<td align="left"><code>min_width</code></td>
<td align="left" style="width: 25%;">

```python
int
```

</td>
<td align="left"><code>160</code></td>
<td align="left">minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.</td>
</tr>

<tr>
<td align="left"><code>interactive</code></td>
<td align="left" style="width: 25%;">

```python
bool | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">if True, will be rendered as an editable textbox; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.</td>
</tr>

<tr>
<td align="left"><code>visible</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, component will be hidden.</td>
</tr>

<tr>
<td align="left"><code>elem_id</code></td>
<td align="left" style="width: 25%;">

```python
str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">None</td>
</tr>

<tr>
<td align="left"><code>elem_classes</code></td>
<td align="left" style="width: 25%;">

```python
list[str] | str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.</td>
</tr>

<tr>
<td align="left"><code>render</code></td>
<td align="left" style="width: 25%;">

```python
bool
```

</td>
<td align="left"><code>True</code></td>
<td align="left">If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.</td>
</tr>

<tr>
<td align="left"><code>key</code></td>
<td align="left" style="width: 25%;">

```python
int | str | None
```

</td>
<td align="left"><code>None</code></td>
<td align="left">if assigned, will be used to assume identity across a re-render. Components that have the same key across a re-render will have their value preserved.</td>
</tr>
</tbody></table>


### Events

| name | description |
|:-----|:------------|
| `change` | Triggered when the value of the SandboxComponent changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input. |
| `input` | This listener is triggered when the user changes the value of the SandboxComponent. |
| `submit` | This listener is triggered when the user presses the Enter key while the SandboxComponent is focused. |



### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As output:** Is passed, the preprocessed input data sent to the user's function in the backend.
- **As input:** Should return, the output data received by the component from the user's function in the backend.

 ```python
 def predict(
     value: tuple[str, bool, list[typing.Any]] | None
 ) -> tuple[str, bool, list[typing.Any]] | dict | None:
     return value
 ```
 
