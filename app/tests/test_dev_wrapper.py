from pathlib import Path


def test_dev_wrapper_uses_uv_tool_run() -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "reviewbuddy"
    script_text = script_path.read_text(encoding="utf-8")

    assert 'uv tool run --from . reviewbuddy "$@"' in script_text
