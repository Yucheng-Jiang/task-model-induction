from action_grounding_service.app.schemas import ActionGroundingRequest, ActionGroundingResponse


def test_request_schema_accepts_contract():
    req = ActionGroundingRequest.model_validate(
        {
            "before_image": "data:image/jpeg;base64,AAAA",
            "after_image": None,
            "action": "click(10, 20)",
            "screen_size": {"width": 1212, "height": 758},
            "highlight_action": True,
            "cursor_location": {"x": 10, "y": 20},
        }
    )
    assert req.screen_size.width == 1212
    assert req.highlight_action is True
    assert req.cursor_location is not None
    assert req.cursor_location.x == 10


def test_response_schema_accepts_contract():
    res = ActionGroundingResponse.model_validate(
        {
            "goal": "Concise immediate user intent",
            "active_application": "Application/window/tab name",
            "visual_content": "Focused UI/content artifact",
            "ocr_results": {
                "screen_size": {"width": 1212, "height": 758},
                "md_results": "optional OCR/markdown text",
                "layout_details": [],
                "data_info": {},
            },
            "md_results": "optional OCR/markdown text",
            "cost": {
                "currency": "USD",
                "total_usd": 0.0,
                "ocr": {"total_usd": 0.0, "items": []},
                "grounding": {"total_usd": 0.0, "items": []},
                "redaction": {"total_usd": 0.0, "items": []},
            },
        }
    )
    assert res.ocr_results.md_results == "optional OCR/markdown text"
    assert res.cost.redaction.total_usd == 0.0
