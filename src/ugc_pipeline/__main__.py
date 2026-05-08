"""Enables `python -m ugc_pipeline` as an alias for `python -m ugc_pipeline.main`."""

from ugc_pipeline.main import cli_entry

cli_entry()
