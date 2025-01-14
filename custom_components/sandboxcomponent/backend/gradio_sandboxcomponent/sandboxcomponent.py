from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from typing import TYPE_CHECKING, Any

from gradio.components.base import Component, FormComponent
from gradio.events import Events
from gradio.data_classes import GradioModel

if TYPE_CHECKING:
    from gradio.components import Timer

from pydantic import BaseModel
from typing import Union
from datetime import datetime

class LoadInteraction(BaseModel):
    type: str = "load"
    time: datetime

class KeydownInteraction(BaseModel):
    type: str = "keydown"
    time: datetime
    key: str

class ClickInteraction(BaseModel):
    type: str = "click"
    time: datetime
    x: float
    y: float

class ScrollInteraction(BaseModel):
    type: str = "scroll"
    time: datetime
    scrollTop: float
    scrollLeft: float

class ResizeInteraction(BaseModel):
    type: str = "resize"
    time: datetime
    width: float
    height: float

class CaptureErrorInteraction(BaseModel):
    type: str = "captureError"
    time: datetime
    error: str

UserInteraction = Union[
    LoadInteraction,
    KeydownInteraction,
    ClickInteraction,
    ScrollInteraction,
    ResizeInteraction,
    CaptureErrorInteraction
]

class SandboxData(GradioModel):
    sandboxUrl: str
    enableTelemetry: bool
    userInteractionRecords: list[UserInteraction]

class SandboxComponent(Component):

    data_model = SandboxData # the data model for this component

    EVENTS = [
        Events.change,
        Events.input,
        Events.submit,
    ]

    def __init__(
        self,
        value: tuple[str, bool, list[Any]] | Callable | None = None,
        *,
        label: str | None = None,
        every: Timer | float | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
        show_label: bool | None = None,
        scale: int | None = None,
        min_width: int = 160,
        interactive: bool | None = None,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: list[str] | str | None = None,
        render: bool = True,
        key: int | str | None = None,
    ):
        """
        Parameters:
            value: url string and interactions.
            placeholder: placeholder hint to provide behind textbox.
            label: the label for this component, displayed above the component if `show_label` is `True` and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component corresponds to.
            every: Continously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.
            inputs: Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.
            show_label: if True, will display label.
            scale: relative size compared to adjacent Components. For example if Components A and B are in a Row, and A has scale=2, and B has scale=1, A will be twice as wide as B. Should be an integer. scale applies in Rows, and to top-level Components in Blocks where fill_height=True.
            min_width: minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.
            interactive: if True, will be rendered as an editable textbox; if False, editing will be disabled. If not provided, this is inferred based on whether the component is used as an input or output.
            visible: If False, component will be hidden.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
            render: If False, component will not render be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.
            key: if assigned, will be used to assume identity across a re-render. Components that have the same key across a re-render will have their value preserved.
        """

        super().__init__(
            label=label,
            every=every,
            inputs=inputs,
            show_label=show_label,
            scale=scale,
            min_width=min_width,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            render=render,
            key=key,
        )

    def preprocess(self, payload: SandboxData | None) -> tuple[str, bool, list[Any]] | None:
        if payload is None:
            return None
        return (
            payload.sandboxUrl,
            payload.enableTelemetry,
            [dict(record) for record in payload.userInteractionRecords]
        )

    def postprocess(self, value: tuple[str, bool, list[Any]] | dict | None) -> SandboxData | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return SandboxData(**value)
        if isinstance(value, tuple):
            sandboxUrl, enableTelemetry, userInteractionRecords = value
            return SandboxData(sandboxUrl=sandboxUrl, enableTelemetry=enableTelemetry, userInteractionRecords=userInteractionRecords)
        return None

    def example_payload(self) -> Any:
        return SandboxData(
            sandboxUrl="https://www.google.com",
            enableTelemetry=True,
            userInteractionRecords=[]
        )

    def example_value(self) -> Any:
        return SandboxData(
            sandboxUrl="https://www.google.com",
            enableTelemetry=True,
            userInteractionRecords=[]
        )
