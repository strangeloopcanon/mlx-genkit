# Release Notes

## Unreleased

- (no changes yet)

## v0.4.5

- Hardened MLX backend output handling (tuple/dict outputs, logits rank) and added mlx-lm signature compatibility shims.
- Fixed `merge_lora()` to actually merge adapters and restore patched forwards.
- Improved packaging (platform markers for `mlx`/`mlx-lm`, optional extras) and added GitHub Actions unit tests.
- Fixed release automation so version bumps are committed before tagging.

## v0.4.4

- Added incremental JSON streaming with live token callbacks and configurable invalid-path handling.
- Documented streaming workflow and expanded structured generation examples.
- Broadened evaluation coverage with stub suites and additional streaming-focused unit tests.
- Verified structured adherence end-to-end on Gemma 2B (MLX backend).
- Added a `make publish` convenience target to bump the patch version, push the git tag, and push to PyPI.
- Adjusted packaging metadata (license handling) to keep PyPI uploads compatible with current validators.
