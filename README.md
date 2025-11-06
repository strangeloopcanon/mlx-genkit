# mlx-genkit
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/strangeloopcanon/mlx-genkit)

Small, reusable MLX generation and training toolkit that brings HF/Torch `generate()` feature parity and clean persona steering to Apple Silicon. It reuses mlx-lm primitives (caches, projections, speculative) and fills the missing parity pieces.

Features
- HF-style `GenerationConfig` with processors/warpers: repetition penalty, no-repeat-ngrams, frequency/presence, bad-words, min_new_tokens, `typical_p`, `epsilon_cutoff`.
- Constraints: `force_words_ids` (strict start + continuation), `suppress_tokens`, `begin_suppress_tokens`, multiple `eos_token_ids`, forced BOS/EOS and per-position `forced_decoder_ids`.
- Modes: sampling (fast path via mlx-lm), beam (`num_beams`, `length_penalty`, `early_stopping`), speculative (mlx-lm), sliding KV (`max_kv_size`).
- Hooks: ResidualInjectionHook (sampling) and LogitBiasHook (sampling/beam); SoftPromptHook for training.
- Structured output enforcement: JSON-schema adherence, retry policies, semantic checks, and grammar stubs.
- Streaming with incremental validation: token callbacks, schema-aware early exits, and detailed result metadata.
- Batch helpers, JSONL adherence logging, and an eval harness for prompt suites.
- Training (MLX): `loss_forward`, `xent_loss` (label smoothing), mixed-precision compute (bf16) with fp32 master weights.
- Training utilities: `sequence_logprob`, `token_kl` for scoring and policy KL.
- Model helpers: `ema_update`, `build_action_mask`, `stable_softmax`; best-effort `clone_reference`.

Install
- From PyPI (recommended):
```
pip install mlx-genkit
```
- Dependencies (if not already installed):
```
pip install mlx mlx-lm transformers
```
- From source (editable):
```
pip install -e .
```

Models from Hugging Face
- If the repo provides MLX weights (e.g., in `mlx-community`), you can load directly: `load('mlx-community/<model>')`.
- For standard HF (PyTorch) repos, convert once using mlx-lm:
  - Python: `from mlx_genkit.interop import convert_hf_to_mlx; convert_hf_to_mlx('Qwen/Qwen3-0.6B', quantize=False, local_out='mlx_qwen3_0_6b')`
  - CLI: `mlx_lm.convert --hf-path Qwen/Qwen3-0.6B --mlx-path mlx_qwen3_0_6b`
  - Then load with `load('mlx_qwen3_0_6b')`.

Auto-convert loader
- You can pass either an HF repo id or a local MLX path to `auto_load`, which will convert once and cache under `./mlx_cache/<sanitized_repo_id>`:
```
from mlx_genkit.loader import auto_load
model, tokenizer, local_path = auto_load('Qwen/Qwen3-0.6B')
print('Loaded from', local_path)  # e.g., ./mlx_cache/Qwen_Qwen3-0.6B
```

Basic usage
```
from mlx_genkit import GenerationConfig, generate
from mlx_lm import load

model, tokenizer = load('mlx_qwen3_0_6b')
cfg = GenerationConfig(max_tokens=64, temperature=0.7, top_p=0.95, seed=17)
out = generate(model, tokenizer, 'Hello MLX parity', cfg)
print(out['text'])
```

Chat prompts (auto chat template)
```
# If you pass a list of HF-style messages, mlx-genkit will automatically
# apply the tokenizer's chat template when available.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize MLX in 3 bullets."},
]
cfg = GenerationConfig(max_tokens=64, temperature=0.7)
out = generate(model, tokenizer, messages, cfg)
print(out['text'])
```

Auto-apply chat template for plain prompts
```
# You can also provide a plain string and have mlx-genkit wrap it
# using the model's chat template when available. This is enabled
# automatically if the tokenizer defines a chat_template; you can
# force or disable it via GenerationConfig, or force with assume_user_chat.
cfg = GenerationConfig(max_tokens=64, temperature=0.7, auto_chat_template=True, system_prompt="You are helpful")
# Equivalent explicit flag:
# cfg = GenerationConfig(max_tokens=64, temperature=0.7, assume_user_chat=True, system_prompt="You are helpful")
out = generate(model, tokenizer, "Summarize MLX in 3 bullets.", cfg)
print(out['text'])
```

Beam and constraints
```
cfg = GenerationConfig(max_tokens=64, temperature=0.0, num_beams=4, early_stopping=True, length_penalty=0.2,
                       force_words_ids=[tokenizer.encode(' cat')], min_new_tokens=8,
                       bad_words_ids=[[tokenizer.eos_token_id]], suppress_tokens=[tokenizer.eos_token_id])
out = generate(model, tokenizer, 'The', cfg)
```

Speculative decoding
```
cfg = GenerationConfig(max_tokens=64, temperature=0.7, top_p=0.95,
                       use_speculative=True, draft_model_id='mlx_qwen3_0_6b', num_draft_tokens=3)
out = generate(model, tokenizer, 'Speculative test', cfg)
```

Structured JSON output
```
from mlx_genkit import GenerationConfig, JsonAdherence, generate

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["summary", "confidence"],
}

cfg = GenerationConfig(max_tokens=220, temperature=0.0)
result = generate(
    model,
    tokenizer,
    "Summarise the change log in JSON",
    cfg,
    json_schema=schema,
    adherence=JsonAdherence(retries=2, strict_only_json=True),
)
print(result.json)
```
- Structured adherence automatically strips common ```json fenced output and keeps
  retrying with escalating prompts when `JsonAdherence(retries=...)` is configured.
  Use `strict_only_json=True` to reject any non-JSON commentary.

Structured DSL
```
from mlx_genkit import StructuredSpec, generate_structured

spec = StructuredSpec(
    schema=schema,
    fields=["summary", "confidence"],
    examples=[{"input": "Bug fix", "output": {"summary": "...", "confidence": 0.9}}],
)
res = generate_structured(model, tokenizer, task="Summarise the diff", spec=spec)
print(res.json)
```

Streaming & incremental validation
```
from mlx_genkit import (
    GenerationConfig,
    JsonAdherence,
    StreamCallbacks,
    generate_stream,
)

def stream_json(model, tokenizer, prompt, schema):
    emitted = []

    def on_token(token, idx):
        emitted.append(token)
        piece = tokenizer.decode([token]) if isinstance(token, int) else str(token)
        if piece:
            print(piece, end="", flush=True)

    def on_invalid_path(info):
        print("\n[invalid path detected]", info["message"])

    callbacks = StreamCallbacks(on_token=on_token, on_invalid_path=on_invalid_path)
    result = generate_stream(
        model,
        tokenizer,
        prompt,
        GenerationConfig(max_tokens=128, temperature=0.0),
        json_schema=schema,
        adherence=JsonAdherence(strict_only_json=True, retries=1),
        on_token=callbacks.on_token,
        on_invalid_path=callbacks.on_invalid_path,
        stop_on_invalid=False,
    )
    print("\nAttempts:", result.attempts, " tokens:", len(emitted))
    return result
```
`StreamCallbacks` let you observe every token while the incremental monitor enforces the schema. Leave `stop_on_invalid=True` for hard stops, or set it to `False` to keep streaming after the warning while JSON retries repair the output.

Persona steering
```
import mlx.core as mx
from mlx_genkit import LogitBiasHook
H = model.args.hidden_size
model['_persona_v'] = mx.random.normal((H,)) * (1.0/(H**0.5))
cfg = GenerationConfig(max_tokens=64, temperature=0.7)
out = generate(model, tokenizer, 'Summarize MLX', cfg, hooks=[LogitBiasHook(param_key='_persona_v', alpha=1.2)])
```

Training (MLX)
```
from mlx_genkit import TrainingConfig, train_step, SoftPromptHook
from mlx.optimizers import AdamW
pad_id = getattr(tokenizer, 'pad_token_id', -100) or -100
opt = AdamW(learning_rate=2e-4)
batch = {'tokens': ...}  # mx.array [B, T]
cfg = TrainingConfig(dtype='bf16', loss_scale=1024.0)
loss = train_step(model, batch, opt, cfg, hooks=[SoftPromptHook(n_virtual=10, param_key='_soft_prompt')], pad_id=pad_id)
```

Utilities
```
from mlx_genkit import sequence_logprob, token_kl, ema_update, build_action_mask

# Per-sample mean log-prob on supervised positions (labels == -100 are ignored)
lp = sequence_logprob(model, batch_tokens, labels)  # [B]

# KL(pi || pref) averaged over supervised positions
kl = token_kl(model, ref_model, batch_tokens, labels)  # [B]

# EMA update of a target model from a source model
ema_update(target_model, model, decay=0.999)

# Supervised mask after prompt
mask = build_action_mask(prompt_lens=[12, 20], seq_len=T)  # [B, T] bool
```

Parity testing
- Torch vs MLX: `python -m mlx_genkit.tests.parity_hf --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b --prompt 'hello'`
- Suite (8 prompts): `python -m mlx_genkit.tests.parity_suite --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b`

CLI wrapper
```
mlxgk-generate \
  --model Qwen/Qwen3-0.6B \
  --prompt "Hello MLX" \
  --max-tokens 64 --temp 0.7 --top-p 0.95 \
  --num-beams 1 --no-repeat-ngram-size 2
```

Structured CLI flags
- `--json-schema schema.json` enable schema validation (optionally combine with `--retries 2`).
- `--strict-only-json` forbid commentary; `--stream` prints incremental output while validating.
- `--validator jsonschema --validator mypkg.validators:custom` layer pluggable validators.
- `--semantic-checks checks.json` applies `must_contain`, `enum_in`, and `regex_on_field` predicates.
- `--log-jsonl adherence.jsonl --log-raw-on-fail` capture adherence diagnostics per run.
- `--backend` selects the decoding backend (`mlx`, `transformers`, or `vllm`) and validates grammar support.

CLI chat and stop strings
- Chat: `--messages-json '[{"role":"user","content":"hi"}]'` (auto-applies template)
- Auto chat for plain prompts: add `--auto-chat` (or disable with `--no-auto-chat`); optional `--system "You are helpful"`
- Force treating plain prompts as user messages: `--assume-user-chat` (equivalent to `--auto-chat`)
- Stop strings: use `--stop` or the alias `--stop-strings` (comma-separated)

Defaults
- The CLI will, by default, auto-apply chat templates when the loaded tokenizer exposes a chat template (has `apply_chat_template` and a non-empty `chat_template`). Use `--no-auto-chat` to turn this off.

Prefetch and convert (download)
```
# Download an HF repo and convert to MLX format without loading into memory.
# Prints the local path, e.g., ./mlx_cache/Qwen_Qwen2-7B-Instruct
mlxgk-download --model Qwen/Qwen2-7B-Instruct

# Options
#  --cache-dir DIR           Cache location (default: ./mlx_cache)
#  --quantize                Quantize during conversion
#  --trust-remote-code       Allow custom code from the repo
#  --force                   Reconvert and overwrite existing cache
```

Adherence eval harness
```
mlxgk-eval --suite adherence_suite.yaml --markdown report.md --json report.json
```
Example suite (`adherence_suite.yaml`):
```
name: adherence_smoke
model: Qwen/Qwen3-0.6B
cases:
  - name: summary
    prompt: "Return a JSON object with summary and confidence"
    json_schema:
      type: object
      properties:
        summary: {type: string}
        confidence: {type: number}
      required: [summary, confidence]
    retries: 2
    strict_only_json: true
```

Performance bench
```
python -m mlx_genkit.tests.perf_bench --hf-model Qwen/Qwen3-0.6B --mlx-model ./mlx_qwen3_0_6b --prompt "Hello performance" --max-tokens 64
```

Releases
- Bump version across files (defaults to patch):
  - `make bump-version` (use `PART=minor` or `PART=major` to override)
- Create and push a git tag (vX.Y.Z):
  - `make git-release`
  - This tags and pushes the repo; PyPI packaging can be added later.

Notes
- Parity targets control‑surface equivalence: constraints, stops, finish reasons, determinism; token streams may differ across frameworks/devices.
- Sampling fast path reuses mlx-lm’s decoding loop and caches for best performance on Apple Silicon.

Known limitations
- Residual injection uses Python-level patching; highly optimized/compiled paths may bypass it. Use `forward_with_hidden(..., strict=True)` when you need deterministic capture/injection semantics.
- Some MLX model classes may not accept `input_embeddings` (used for soft prompts in training). In those cases, the library now falls back gracefully to standard token-only forward.
- Beam search applies processors on raw logits and then normalizes (HF behavior). Earlier parity reports in this repo may reflect the previous implementation on normalized logprobs.

Tips
- When running examples directly from the repo, make sure you’re using the local sources: `pip install -e .` or run with `PYTHONPATH=.`.
- Parity/perf harnesses will download HF models; ensure network access and sufficient disk space.
