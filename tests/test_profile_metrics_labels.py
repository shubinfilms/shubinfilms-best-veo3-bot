from metrics import profile_first_paint_ms, profile_open_total, profile_render_ms


def test_profile_metrics_accept_expected_labels():
    profile_render_ms.labels(force_refresh="true", source="button")
    profile_open_total.labels(force_refresh="false", source="menu", result="ok")
    profile_first_paint_ms.labels(force_refresh="false", source="unknown")
