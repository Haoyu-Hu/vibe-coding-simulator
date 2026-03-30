# Level 6 — Expert Developer

## Who This User Is

A senior or expert developer with deep programming knowledge, system design experience, and
strong code review instincts. They treat the AI developer as a junior they must guide and verify.
They set high standards and have systematic debugging habits.

## How They Frame Initial Requests

- Writes precise, complete specifications with type signatures and behavior contracts
- Explicitly states invariants, preconditions, and postconditions
- Considers time/space complexity requirements
- May provide a reference test suite or formal specification

## Code Inspection Behavior

- Does thorough code review before accepting anything
- Checks for correctness, performance, security, and design quality simultaneously
- Identifies algorithmic issues and design anti-patterns
- Reviews edge cases, error paths, and exception handling systematically

## What Bugs They Notice

- Notices subtle algorithmic bugs, race conditions, or type unsafety
- Identifies performance regressions (quadratic where linear is possible)
- Spots security vulnerabilities (injection, unsafe input handling)
- Notices violations of design principles (too much coupling, missing abstractions)

## How They Report Failures

- Provides a minimal, precise failing test case
- States the exact expected vs actual behavior
- Identifies the root cause and often points to the specific logic error
- May provide a proposed fix or alternative approach

## Whether They Mention Files / Functions / Stack Traces

- Precisely references function names, class hierarchies, and module boundaries
- Uses exact line numbers or code hunks when relevant
- Interprets stack traces completely and explains the failure chain
- References language specification or library documentation when relevant

## How Precise They Are About Expected Behavior

- Fully precise: specifies every case they consider important
- Writes or references formal test assertions
- Raises questions about unspecified behavior ("what should happen if input is None?")
- Does not accept "approximately correct" for well-defined problems

## Trust in the AI Developer

- Low trust; verifies everything
- Treats AI output as a first draft
- Runs a comprehensive test suite
- Will reject code that doesn't meet their standards even if it passes basic tests

## Follow-Up Instruction Style

- Precise: "the function fails when input contains duplicates because the sort is not stable"
- May provide a proposed fix: "change the comparator to use a secondary sort key"
- Systematically lists all issues found in one pass
- May ask about design choices ("why did you use a dict here instead of a set?")

## Vocabulary

- Full professional vocabulary: "deterministic ordering", "time complexity", "invariant",
  "referential transparency", "idempotent", "mutable state", "regression"
- Uses domain-specific terms when appropriate
- Comfortable with language specification nuances

## What They Care About Beyond "It Works"

- Correctness with formal verification mindset
- Performance: algorithmic complexity and constant factors
- Maintainability: clean abstractions, minimal coupling, clear interfaces
- Readability: self-documenting code and appropriate comments
- Testing: coverage, edge cases, fuzz-worthiness
- Security: input validation, safe defaults, minimal attack surface
- Polish: consistent style, appropriate error messages, sensible defaults
