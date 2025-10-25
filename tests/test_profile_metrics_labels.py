from metrics import profile_first_paint_ms, profile_open_total


def test_profile_metrics_accept_expected_labels():
    profile_open_total.labels(source="menu", result="ok")
    profile_first_paint_ms.labels(source="unknown")
