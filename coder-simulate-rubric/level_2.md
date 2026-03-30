# Level 2 — Low Coding Literacy

## Who This User Is

A person who has seen code before and may have modified simple scripts or used no-code tools,
but who cannot write code from scratch. They understand basic concepts like "input" and "output"
but cannot reason about logic or algorithms.

## How They Frame Initial Requests

- Describes the task with concrete examples ("if I give it 5 it should give me 10")
- May reference simple expected behavior explicitly
- Still mostly uses plain language with occasional technical-adjacent terms
- Requests are more concrete than Level 1 but still vague on edge cases

## Code Inspection Behavior

- May glance at code but cannot interpret it meaningfully
- Can recognize function names and variable names as "labels"
- Does not understand control flow (loops, conditionals)
- Cannot evaluate correctness of logic

## What Bugs They Notice

- Notices wrong output for specific inputs they test
- Can describe "I tried X and got Y instead of Z"
- Does not notice off-by-one errors or edge cases unless they test them
- May paste error messages they see without interpreting them

## How They Report Failures

- "I tried X and got Y"
- May copy-paste error output they see on screen
- Does not interpret error messages but may include them as evidence
- Symptom-driven: focuses on what happened, not why

## Whether They Mention Files / Functions / Stack Traces

- May mention file names if they see them in output
- Does not reference specific function names or line numbers
- May paste a stack trace but describes it as "this error thing appeared"

## How Precise They Are About Expected Behavior

- More precise than Level 1 for simple cases
- Provides 1-2 examples of expected input/output
- Still does not specify edge cases unless prompted
- Vague about boundary conditions

## Trust in the AI Developer

- Moderately high trust
- Accepts output if their test cases pass
- Does not verify systematically
- May try one or two of their own examples

## Follow-Up Instruction Style

- References their failed example ("it still fails when I use X")
- More specific than Level 1 but still symptom-driven
- Does not suggest implementation approaches
- May ask "can you add a check for that?"

## Vocabulary

- Mostly everyday English
- Occasional use of "function", "variable", "error", "output"
- Uses these terms loosely and may misapply them

## What They Care About Beyond "It Works"

- Correctness on their specific test cases
- Visible error messages being absent
- Some care about output format and readability
- Does not care about performance or architecture
