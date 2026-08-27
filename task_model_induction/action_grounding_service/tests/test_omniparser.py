from action_grounding_service.app.omniparser import normalize_bbox


def test_normalize_bbox_scales_relative_coordinates():
    assert normalize_bbox([0.1, 0.2, 0.5, 0.6], 1000, 500) == [100.0, 100.0, 500.0, 300.0]


def test_normalize_bbox_rejects_empty_boxes():
    assert normalize_bbox([10, 10, 10, 20], 100, 100) is None
