"""
Adopted from https://github.com/gradio-app/gradio/blob/main/gradio/components.py
Fix a markdown render problem.
"""
from __future__ import annotations

from gradio.components import *
from markdown2 import Markdown


class _Keywords(Enum):
    NO_VALUE = "NO_VALUE"  # Used as a sentinel to determine if nothing is provided as a argument for `value` in `Component.update()`
    FINISHED_ITERATING = "FINISHED_ITERATING"  # Used to skip processing of a component's value (needed for generators + state)


@document("style")
class Chatbot(Changeable, Selectable, IOComponent, JSONSerializable):
    """
    Displays a chatbot output showing both user submitted messages and responses. Supports a subset of Markdown including bold, italics, code, and images.
    Preprocessing: this component does *not* accept input.
    Postprocessing: expects function to return a {List[Tuple[str | None | Tuple, str | None | Tuple]]}, a list of tuples with user message and response messages. Messages should be strings, tuples, or Nones. If the message is a string, it can include Markdown. If it is a tuple, it should consist of (string filepath to image/video/audio, [optional string alt text]). Messages that are `None` are not displayed.

    Demos: chatbot_simple, chatbot_multimodal
    """

    def __init__(
        self,
        value: List[Tuple[str | None, str | None]] | Callable | None = None,
        color_map: Dict[str, str] | None = None,  # Parameter moved to Chatbot.style()
        *,
        label: str | None = None,
        every: float | None = None,
        show_label: bool = True,
        visible: bool = True,
        elem_id: str | None = None,
        elem_classes: List[str] | str | None = None,
        **kwargs,
    ):
        """
        Parameters:
            value: Default value to show in chatbot. If callable, the function will be called whenever the app loads to set the initial value of the component.
            label: component name in interface.
            every: If `value` is a callable, run the function 'every' number of seconds while the client connection is open. Has no effect otherwise. Queue must be enabled. The event can be accessed (e.g. to cancel it) via this component's .load_event attribute.
            show_label: if True, will display label.
            visible: If False, component will be hidden.
            elem_id: An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.
            elem_classes: An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.
        """
        if color_map is not None:
            warnings.warn(
                "The 'color_map' parameter has been deprecated.",
            )
        #self.md = utils.get_markdown_parser()
        self.md = Markdown(extras=["fenced-code-blocks", "tables", "break-on-newline"])
        self.select: EventListenerMethod
        """
        Event listener for when the user selects message from Chatbot.
        Uses event data gradio.SelectData to carry `value` referring to text of selected message, and `index` tuple to refer to [message, participant] index.
        See EventData documentation on how to use this event data.
        """

        IOComponent.__init__(
            self,
            label=label,
            every=every,
            show_label=show_label,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            value=value,
            **kwargs,
        )

    def get_config(self):
        return {
            "value": self.value,
            "selectable": self.selectable,
            **IOComponent.get_config(self),
        }

    @staticmethod
    def update(
        value: Any | Literal[_Keywords.NO_VALUE] | None = _Keywords.NO_VALUE,
        label: str | None = None,
        show_label: bool | None = None,
        visible: bool | None = None,
    ):
        updated_config = {
            "label": label,
            "show_label": show_label,
            "visible": visible,
            "value": value,
            "__type__": "update",
        }
        return updated_config

    def _process_chat_messages(
        self, chat_message: str | Tuple | List | Dict | None
    ) -> str | Dict | None:
        if chat_message is None:
            return None
        elif isinstance(chat_message, (tuple, list)):
            mime_type = processing_utils.get_mimetype(chat_message[0])
            return {
                "name": chat_message[0],
                "mime_type": mime_type,
                "alt_text": chat_message[1] if len(chat_message) > 1 else None,
                "data": None,  # These last two fields are filled in by the frontend
                "is_file": True,
            }
        elif isinstance(
            chat_message, dict
        ):  # This happens for previously processed messages
            return chat_message
        elif isinstance(chat_message, str):
            #return self.md.render(chat_message)
            return str(self.md.convert(chat_message))
        else:
            raise ValueError(f"Invalid message for Chatbot component: {chat_message}")

    def postprocess(
        self,
        y: List[
            Tuple[str | Tuple | List | Dict | None, str | Tuple | List | Dict | None]
        ],
    ) -> List[Tuple[str | Dict | None, str | Dict | None]]:
        """
        Parameters:
            y: List of tuples representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.  It can also be a tuple whose first element is a string filepath or URL to an image/video/audio, and second (optional) element is the alt text, in which case the media file is displayed. It can also be None, in which case that message is not displayed.
        Returns:
            List of tuples representing the message and response. Each message and response will be a string of HTML, or a dictionary with media information.
        """
        if y is None:
            return []
        processed_messages = []
        for message_pair in y:
            assert isinstance(
                message_pair, (tuple, list)
            ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
            assert (
                len(message_pair) == 2
            ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"
            processed_messages.append(
                (
                    #self._process_chat_messages(message_pair[0]),
                    '<pre style="font-family: var(--font)">' +
                    message_pair[0] + "</pre>",
                    self._process_chat_messages(message_pair[1]),
                )
            )
        return processed_messages

    def style(self, height: int | None = None, **kwargs):
        """
        This method can be used to change the appearance of the Chatbot component.
        """
        if height is not None:
            self._style["height"] = height
        if kwargs.get("color_map") is not None:
            warnings.warn("The 'color_map' parameter has been deprecated.")

        Component.style(
            self,
            **kwargs,
        )
        return self


