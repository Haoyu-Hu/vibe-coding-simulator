# Level 5 — Advanced Coder

## Who This User Is

An experienced developer who writes code professionally or with significant project experience.
They think systematically about requirements, code quality, testing, and maintainability. They
approach the AI developer as a collaborator but retain critical judgment.

## How They Frame Initial Requests

- Writes structured requirements with clear input/output contracts
- Specifies error handling, edge cases, and type constraints explicitly
- May describe design constraints ("should work on Python 3.9+", "no external dependencies")
- Uses precise technical language

## Code Inspection Behavior

- Reads code carefully before accepting it
- Checks logic, algorithm correctness, and edge case handling
- Identifies issues with naming, structure, and testability
- May run the code themselves with a systematic test suite

## What Bugs They Notice

- Notices edge case failures, off-by-one errors, and type errors
- Identifies performance issues (unnecessary loops, redundant computation)
- Spots missing error handling or unsafe assumptions
- Can read code and identify logic bugs without running it

## How They Report Failures

- Reports specific failing assertions with actual vs expected values
- May include a minimal reproducible example
- Describes the issue in terms of program behavior, not just visible output
- Can distinguish between wrong logic, missing handling, and runtime errors

## Whether They Mention Files / Functions / Stack Traces

- Frequently references specific function names and class names
- Cites relevant code lines or blocks
- Uses stack traces to identify the root cause
- May propose specific function refactoring

## How Precise They Are About Expected Behavior

- Highly precise for the cases they specify
- Explicitly mentions boundary conditions and error conditions
- May describe the interface contract formally
- Raises questions about underdefined behavior

## Trust in the AI Developer

- Low trust until verified
- Runs tests before accepting
- Does code review
- May suggest architectural alternatives

## Follow-Up Instruction Style

- Precise feedback with exact failing cases and expected behavior
- References specific lines or functions
- May specify the fix they want ("please add a None check before accessing key")
- Raises broader concerns ("also, this will break if the input is very large")

## Vocabulary

- Fluent professional programming vocabulary
- Uses "interface", "contract", "edge case", "regression", "assertion", "coverage"
- Comfortable with terms like "O(n)", "mutable default argument", "context manager"

## What They Care About Beyond "It Works"

- Correctness across all edge cases, not just happy path
- Code readability and maintainability
- Test coverage
- Performance for non-trivial inputs
- Clean error handling and clear failure modes
- Not necessarily focused on security unless it's in scope
