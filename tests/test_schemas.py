"""Smoke test that pydantic + pytest + fixtures wire up correctly.

Once `src/ugc_pipeline/state/schemas.py` exists with a real `VideoState`,
this test will validate it against a fixture. For Phase 0 it asserts the
test harness is wired correctly with a placeholder pydantic model.
"""

from __future__ import annotations

import pydantic


def test_pydantic_harness_works() -> None:
    class _Stub(pydantic.BaseModel):
        video_id: str
        status: str

    instance = _Stub(video_id="prod_abc_v1", status="in_progress")
    assert instance.video_id == "prod_abc_v1"
    assert instance.status == "in_progress"
