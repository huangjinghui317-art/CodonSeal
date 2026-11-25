# Third-party materials

The source release includes the CodonSeal implementation, pinned asset metadata, Open-R1 problem identifiers, and derived evaluation scores. Download the source data and checkpoints through their original providers. This license applies to the CodonSeal source code. The original licenses for all third-party assets continue to apply.

- [Qwen2.5-Coder-1.5B-Instruct, MLX 4-bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit), revision `b3252a2f97102b1fb1571fec2c9b27219a8536be`. The model architecture recipe and file hashes are in the asset registry.
- [HumanEval](https://github.com/openai/human-eval), revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, MIT. The evaluation runner downloads its official 164-question artifact.
- [Open-R1 Verifiable Coding Problems Python](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python), revision `b761a24a95fa03289a231d2d31c183636ffb9833`, a Python projection of PrimeIntellect's Verifiable Coding Problems dataset. The WPN reconstruction script retrieves the specified shards and uses the pinned train/validation IDs.

Original source notice: Copyright (c) 2026 CodonSeal contributors.
