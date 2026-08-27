"""Lightweight helpers for recorder trace parsing."""

import base64
import re


def encode_image(img_path: str, return_url: bool = False) -> str:
    """Encode an image file to base64."""
    with open(img_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode()
    if return_url:
        return f"data:image/jpeg;base64,{encoded}"
    return encoded


def is_keyboard_action(action: str) -> bool:
    return "press" in action


def is_click_action(action: str) -> bool:
    return "click" in action


def is_scroll_action(action: str) -> bool:
    return "scroll" in action


def is_drag_action(action: str) -> bool:
    return "drag" in action


def get_key_input(action: str) -> str:
    """Parse the key input from the action string."""
    if "(" in action and ")" in action:
        key_input = action.split("(")[1].split(")")[0]
    else:
        key_input = action
    key_input = key_input.replace("'", "").strip()

    if key_input == "Key.space":
        return " "
    if key_input == "Key.shift":
        return ""
    if key_input == "Key.backspace":
        return key_input
    if key_input.startswith("Key."):
        return key_input + "+"
    return key_input


def compose_key_input(input_list: list[str]) -> str:
    """Compose a final text value from buffered key inputs."""
    composed_input_list = []
    for item in input_list:
        if item == "Key.backspace" and composed_input_list:
            composed_input_list[-1] = composed_input_list[-1][:-1]
        else:
            composed_input_list.append(item)
    return "".join(composed_input_list)
