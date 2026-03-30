# Coder Simulate Rubric

This directory defines the knowledge-level rubric system used by the vibe-coding user simulator.

## Purpose

The vibe-coding experiment simulates a realistic human developer-user interacting with an AI coding
assistant across multiple rounds. Each simulated user has a fixed **knowledge level** (1-6) that
determines how they describe tasks, interpret code, report bugs, and give feedback.

The rubric system ensures this behavior is consistent, configurable, and driven by machine-readable
definitions rather than ad-hoc prompts.

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `defaults.json` | Default level distribution weights and shared config |
| `level_1.md` | Rubric for Level 1: No coding background |
| `level_2.md` | Rubric for Level 2: Low coding literacy |
| `level_3.md` | Rubric for Level 3: Beginner coder |
| `level_4.md` | Rubric for Level 4: Intermediate coder |
| `level_5.md` | Rubric for Level 5: Advanced coder |
| `level_6.md` | Rubric for Level 6: Expert developer |

## How It Works

1. At chain initialization, one knowledge level is sampled **once** per chain from the configured
   distribution and kept fixed for all rounds.
2. The level is used to select the appropriate rubric.
3. The rubric text is injected into the user simulator prompt for every round in that chain.
4. The user simulator LLM is instructed to behave consistently with the rubric.

## Level Distribution

By default, levels are sampled from a normal-like distribution centered at 3-4 (intermediate).
The distribution is stored in `defaults.json` and can be overridden at experiment run time.

## Trust and Delegation Spectrum

The six levels form a trust/delegation spectrum:

- **Levels 1-2**: High trust, low technical visibility — users delegate entirely and react to visible behavior only.
- **Levels 3-4**: Medium trust, medium visibility — users notice some technical issues but rely on AI for diagnosis.
- **Levels 5-6**: Low trust, high visibility — users verify actively, inspect code, and specify requirements precisely.

## Runtime Integration

The rubric files are loaded at runtime by `utils/vibe_coding_user_sim.py`. The `defaults.json` is
read by the chain builder to configure level sampling. The level rubric markdown files are embedded
into simulator prompts.
