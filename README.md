# CodonSeal

CodonSeal embeds a keyed signal in generated Python code. It seals a lock from a run of low-choice positions and uses it at the next eligible key position to bias sampling. Two variants share the lock construction and the frozen Qwen generator.

| Method | How it selects lock and key positions | Detection inputs |
|---|---|---|
| **CodonSeal** | A small WPN predictor estimates candidate richness and the expected next-token embedding from the two most recent answer tokens. The legacy Tiny model attends to a 16-slot sequence whose preceding 14 slots contain the fixed padding token. | Answer text, tokenizer, WPN weights, static token geometry and key. The original prompt and generator are unnecessary. |
| **CodonSeal-oracle** | The full generator provides candidate richness and token scores. The original semantic coefficient is 2.0. | Answer text, original prompt, original model and key. The detector uses the prompt-conditioned log-likelihood ratio. |
| **Unmarked** | The same generator samples without watermark bias. | Serves as the paired negative set for both detection methods. |

[Step-by-step reproduction guide](#reproduce-the-experiments) · [Benchmark and training logs](#published-logs)

## HumanEval results

<!-- RESULTS:START -->
### All five seeds

Mean ± sample standard deviation across the five generation seeds.

| Method | AUROC | TPR at ≤5% FPR | Actual FPR | Pass@1 | Self-PPL |
|---|---:|---:|---:|---:|---:|
| Unmarked | — | — | — | 50.98% ± 2.46% | 1.1159 ± 0.0080 |
| CodonSeal-oracle | 0.9208 ± 0.0139 | 65.37% ± 5.46% | 4.88% ± 0.00% | 49.63% ± 2.35% | 1.1359 ± 0.0083 |
| CodonSeal | 0.7207 ± 0.0052 | 27.32% ± 2.77% | 4.76% ± 0.27% | 47.80% ± 1.96% | 1.1777 ± 0.0086 |

### Per-seed results

| Seed | CodonSeal AUROC | Oracle AUROC | CodonSeal Pass@1 | Oracle Pass@1 | Unmarked Pass@1 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 0.7243 | 0.9299 | 49.39% | 46.34% | 51.83% |
| 2027 | 0.7235 | 0.8982 | 45.12% | 49.39% | 48.78% |
| 2028 | 0.7180 | 0.9191 | 46.95% | 52.44% | 50.00% |
| 2029 | 0.7248 | 0.9337 | 47.56% | 51.22% | 49.39% |
| 2030 | 0.7128 | 0.9232 | 50.00% | 48.78% | 54.88% |

<!-- RESULTS:END -->

See [per-seed summary](results/merix/summary.csv), [aggregate metrics](results/summary.json) and [per-task records for all five seeds](results/merix/). The per-task CSVs contain raw detector scores, sandbox pass flags, answer token counts and summed model NLL. These are sufficient to recompute the reported metrics.

### Protocol

- All 164 official HumanEval problems appear in each method and each seed. Each method generates one answer per question per seed. The methods share Qwen2.5-Coder-1.5B-Instruct (MLX 4-bit), temperature 0.8, an original pre-bias top-64 candidate set, a 256-token cap and the same per-question random stream definition.
- Generation seeds are 2026–2030. WPN training uses seed 42 and one checkpoint shared by all five generation seeds. The public benchmark key is derived from the label in [the configuration](configs/protocol.json). Use an external key file for private experiments.
- The WPN trains on 192 Open-R1 problems and validates on 48 other problems. The frozen [split file](configs/openr1_splits.json) also records the 64 external development problems on which the detector aggregation was selected. WPN fitting uses 13,577 distinct training token positions and 2,983 validation positions. The checkpoint minimizes the validation objective. The teacher conditions on the Open-R1 problem when constructing labels. The WPN consumes two answer tokens during generation and detection.
- HumanEval provides the final evaluation examples. The archived data preparation screened the Open-R1 candidates for high statement overlap and normalized AST overlap with HumanEval. The screen cannot guarantee identification of paraphrases or overlap in the base model's pretraining.
- AUROC uses raw scores, and curves group ties. TPR at 5% FPR is the maximum attainable empirical TPR with FPR at or below 5%. The table includes the observed FPR. Pass@1 is the fraction of submitted answers passing official tests in the OS sandbox. Self-PPL is computed from the original Qwen probabilities on generated answer tokens, conditioned on the prompt.

## Reproduce the experiments

Start with the pinned model and datasets, build the WPN once, run HumanEval for all five seeds, and export the metrics locally. Each step uses paths relative to the repository root. The checked-in results can be inspected directly in [results/merix](results/merix/).

### 1. Set up the environment

Use an **Apple silicon Mac** with native ARM64 Python 3.11, Metal, Git, curl, and `/usr/bin/sandbox-exec`. The reference host ran macOS 15.7.3. The full workflow uses MLX and a macOS sandbox to run generated Python against the official tests. Plan for at least 5 GB of free disk space for the environment, downloads, and intermediate files. The clone contains source and derived results. The setup below keeps weights, datasets, and new runs in the ignored `local/` directory.

If Python 3.11 is missing and you use Homebrew, install it with `brew install python@3.11`. In a Bash or Zsh terminal, run the following commands.

```bash
git clone https://github.com/huangjinghui317-art/CodonSeal.git
cd CodonSeal
uname -m
python3.11 --version
test -x /usr/bin/sandbox-exec
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
python -c 'import platform; import mlx.core as mx; assert platform.machine() == "arm64" and mx.metal.is_available(); print(mx.default_device())'
```

The first architecture check should print `arm64`. The last command should report an MLX GPU device. Use `source .venv/bin/activate` again when opening a fresh terminal.

If MLX reports that no Metal device is available, run the GPU check from a native Terminal session on the Mac. During the body-only demo, Transformers may report that PyTorch is unavailable. That command uses its tokenizer and runs the WPN with MLX.

### 2. Download and verify the model and datasets

The generator is the **MLX 4-bit** build of Qwen2.5-Coder-1.5B-Instruct. The model, tokenizer, and HumanEval total roughly 0.88 GB. The four Open-R1 training shards add roughly 0.53 GB. The listed assets are public.

| Asset | Pinned source | Local destination |
|---|---|---|
| Generator and tokenizer | [mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit), revision `b3252a4f97102b1fb1571fec2c9b27219a8536be` | `local/model/` |
| HumanEval | [openai/human-eval](https://github.com/openai/human-eval), commit `b9453e8f1865df0a0057a1383767caa400acfbcf` | `local/data/humaneval/HumanEval.jsonl.gz` |
| WPN training and validation data | [open-r1/verifiable-coding-problems-python](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python), revision `b761a24a95fa03289a231d2d31c183636ffb9833`, shards 00002 through 00005 | `local/data/openr1/` |

Run the downloader with `--with-training` for the full five-seed workflow. Running without that flag fetches the generator, tokenizer, and benchmark only.

```bash
mkdir -p local/logs
set -o pipefail
python -m codonseal.assets --with-training 2>&1 | tee -a local/logs/download.log
```

Wait for a `Verified` line for each file. The downloader checks SHA-256 against [the model and HumanEval manifest](configs/assets.json) and [the Open-R1 manifest](configs/openr1_assets.json). It writes the exact benchmark `config.json` from the manifest. If the connection drops, rerun the same command. Verified files are reused, and unfinished downloads can resume from their `.part` files if the server supports range requests. The downloader tries the listed mirror URLs after the primary URL. If the mirror is also rate-limited, retry later or download from a network that can reach the listed sources.

If you need to download the weights on another machine, the following command fetches the pinned model file. Copy it to `local/model/model.safetensors` on the benchmark Mac and run the asset command again to verify it and fetch the other files. The manifest lists the exact URLs, target paths, and checksums for manual transfer of the tokenizer and data as well.

```bash
mkdir -p local/model
curl --fail --location --retry 2 \
  --output local/model/model.safetensors \
  'https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit/resolve/b3252a2f97102b1fb1571fec2c9b27219a8536be/model.safetensors'
shasum -a 256 local/model/model.safetensors
```

The expected SHA-256 for `model.safetensors` is `daeab4764fb420d161721791cf2e509e2de81a7af4223646e7bed2bf82c57b58`. Use the asset command to install the pinned configuration even when you supply files manually.

If `raw.githubusercontent.com` is unreachable, fetch the same HumanEval file through the GitHub Contents API and verify it before retrying the main command.

```bash
mkdir -p local/data/humaneval
curl --fail --location --retry 2 \
  --header 'Accept: application/vnd.github.raw+json' \
  --output local/data/humaneval/HumanEval.jsonl.gz \
  'https://api.github.com/repos/openai/human-eval/contents/data/HumanEval.jsonl.gz?ref=b9453e8f1865df0a0057a1383767caa400acfbcf'
printf '%s  %s\n' \
  'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef' \
  'local/data/humaneval/HumanEval.jsonl.gz' | shasum -a 256 --check
```

### 3. Train the two-token WPN

The WPN fits on 192 fixed Open-R1 problems with a separate 48-problem validation split. Training uses seed 42 and the pinned Qwen model as a teacher. It builds the vocabulary geometry from the model embeddings, computes target candidate counts and next-token embedding forecasts on the reference Python solutions, then fits the WPN. HumanEval is reserved for the final benchmark.

```bash
set -o pipefail
python -m codonseal.train 2>&1 | tee -a local/logs/train.log
python -m json.tool local/wpn/training_result.json
```

| Output | Contents |
|---|---|
| `local/wpn/geometry.safetensors` | Frozen vocabulary geometry |
| `local/wpn/teacher.sqlite` | Cached labels for the external training and validation problems |
| `local/wpn/tiny2/training.json` | Loss history and last completed epoch |
| `local/wpn/tiny2/best.safetensors` | Checkpoint with the lowest validation loss, shared by all five generation seeds |
| `local/wpn/training_result.json` | Geometry and checkpoint hashes, selected epoch, and training coverage |

The rebuild selected epoch 6 and reports 13,577 training positions and 2,983 validation positions. Compare both hashes in `training_result.json` with [configs/protocol.json](configs/protocol.json). A fresh environment on the reference host rebuilt the WPN directly from the four verified Parquet shards and matched both archived hashes. Other hardware or numerical library builds can produce differences. If training is interrupted, rerun the same command with the same sources and output directory to resume teacher labeling at completed problems and WPN fitting at the last saved epoch.

### 4. Run the HumanEval benchmark

Run all five seeds using the default command. Each seed covers the same 164 tasks under Unmarked, CodonSeal-oracle, and CodonSeal. The configuration fixes the prompts, temperature, pre-bias top-k, answer budget, and the pairing of positives with unmarked negatives. Generation, detection, and sandbox grading each have a terminal progress bar.

```bash
set -o pipefail
python -m codonseal.evaluate --out local/run 2>&1 | tee -a local/logs/humaneval.log
```

While the job runs, view the auto-refreshing progress page from a second terminal in the repo.

```bash
open local/run/progress.html
tail -f local/logs/humaneval.log
```

The terminal prints `Complete seed` as each seed finishes. A complete five-seed run records 2,460 generations, 3,280 detector scores, and 2,460 test results in `local/run/run.sqlite`. The reference evaluation took about 32 minutes after model preparation and training, as recorded in the [final progress snapshot](results/logs/humaneval-progress.json). After an interruption, rerun the same evaluation command with the same `--out`, assets, and seed list. The runner reuses committed records and checks the configuration against `local/run/run.json`. Keep the SQLite database and the manifest together.

For a smaller trial, run `python -m codonseal.evaluate --seeds 2026 --out local/smoke` after WPN training. This still grades 164 tasks per method. Use `local/run` for the full five-seed run. For a private-key experiment, keep the binary key outside the repository, pass `--key-file /path/to/key`, and use a fresh `--out` directory.

### 5. Export and compare the metrics

The public exporter reads a completed five-seed database with the public benchmark key. Export to a new directory so you can compare against the checked-in results.

```bash
set -o pipefail
python -m codonseal.export --run local/run --output local/reproduced 2>&1 | tee -a local/logs/export.log
diff -u results/merix/summary.csv local/reproduced/merix/summary.csv
```

The export includes `merix/seed_2026.csv` through `merix/seed_2030.csv` with one record per task, `merix/summary.csv` with one record per seed and method, and `summary.json` with aggregate metrics, coverage counts, and asset checksums. If any values differ, compare the asset hashes in the two `summary.json` files and confirm the same Python and pinned package versions were used. To regenerate the tables in your local README as well, pass `--update-readme` to the exporter.

### Published logs

- [HumanEval benchmark log](results/logs/humaneval.log) and [final progress snapshot](results/logs/humaneval-progress.json) cover all five published seeds.
- [WPN rebuild log](results/logs/wpn-rebuild.log) and [epoch-level training record](results/logs/wpn-training.json) show the checkpoint hashes, losses, and selected validation epoch from the independent rebuild.

The published terminal logs retain the final progress-bar line for each phase and the completion messages. The commands above write your own full terminal logs under `local/logs/`.

### Generate or detect an individual answer

After downloading the model and training the WPN, try the released generation and detection commands on a single Python prompt.

```bash
mkdir -p local/demo
printf 'Write a Python function add(a, b) that returns their sum. Output Python code only.' > local/demo/prompt.txt
python scripts/demo.py generate --method CodonSeal --prompt local/demo/prompt.txt > local/demo/answer.py
python scripts/demo.py detect --method CodonSeal --answer local/demo/answer.py
python scripts/demo.py generate --method CodonSeal-oracle --prompt local/demo/prompt.txt > local/demo/oracle.py
python scripts/demo.py detect --method CodonSeal-oracle --prompt local/demo/prompt.txt --answer local/demo/oracle.py
python scripts/demo.py generate --method unmarked --prompt local/demo/prompt.txt > local/demo/unmarked.py
```

For the body-only detector, the required model-side files are `local/model/tokenizer.json`, `local/model/tokenizer_config.json`, `local/wpn/geometry.safetensors`, and `local/wpn/tiny2/best.safetensors`. The oracle detector uses the full generator and exact original prompt. Both commands return a raw `score` and the number of `candidate_keys`. The public benchmark evaluates the raw scores over all 164 positive-negative pairs per seed.

## Files and license

- `src/codonseal/core.py` defines the lock/key state and keyed features.
- `src/codonseal/backend.py` and `src/codonseal/watermark.py` implement generation and both detectors.
- `src/codonseal/train.py` rebuilds the frozen two-token WPN configuration.
- `src/codonseal/evaluate.py` and `src/codonseal/export.py` produce the benchmark and public tables.
- `configs/` records source revisions, hashes and the fixed experimental design.
- `results/` contains derived public metrics.

The source code is released under the [MIT License](LICENSE). See [NOTICE](NOTICE.md) for third-party sources. The paper-oriented experiments use a single generator, a single benchmark, one WPN training seed and no attacks.
