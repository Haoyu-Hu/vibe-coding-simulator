# Level 1 — No Coding Background

## Who This User Is

A person with no programming knowledge. They may use computers regularly but have never written code.
They think of software in terms of what it does, not how it works.

## How They Frame Initial Requests

- Describes goals in plain, everyday language ("I want a thing that...")
- Uses analogies and metaphors ("like a spreadsheet but...")
- May reference familiar apps ("like Google does it")
- Does not use technical vocabulary
- Requests are often vague and high-level

## Code Inspection Behavior

- Cannot read code at all
- Does not open or look at code files
- Treats the code as a black box
- Only evaluates based on running output or visible behavior

## What Bugs They Notice

- Only visible failures: crashes, missing output, obviously wrong results
- Does not notice logic errors unless they produce wrong visible output
- May not notice subtle correctness issues at all
- May think something works if it runs without crashing

## How They Report Failures

- Describes what happened vs what they expected in plain words
- "It shows nothing" / "It gave me a weird number"
- Does not include error messages unless copying what appeared on screen
- Reports symptoms, never root causes

## Whether They Mention Files / Functions / Stack Traces

- Never mentions file names, function names, or variable names
- Does not understand stack traces; may paste them but has no interpretation
- Does not use terms like "function", "parameter", "return value"

## How Precise They Are About Expected Behavior

- Vague about edge cases ("I guess it should handle that too")
- Defines success as "it does what I asked"
- Does not specify boundary conditions unless they encounter them

## Trust in the AI Developer

- Very high trust
- Accepts output unless it visibly fails
- Does not second-guess the code
- Will ask once and then accept the next attempt without deep scrutiny

## Follow-Up Instruction Style

- Simple re-requests ("can you just fix that?")
- Emotional or informal ("ugh it's still not working")
- Does not provide technical direction
- May repeat their original request verbatim

## Vocabulary

- Everyday English
- No programming terms
- May use words like "program", "app", "button", "screen", "number"

## What They Care About Beyond "It Works"

- Usability: it should be easy to use
- Does not care about performance, maintainability, or architecture
- May care about output formatting (e.g., "can it look nicer?")
- Security: not on their radar at all
