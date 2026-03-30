# Level 3 — Beginner Coder

## Who This User Is

A person who has written simple code (e.g., intro-level Python, small scripts) but has limited
experience with larger programs or debugging. They can follow code logic for simple functions but
struggle with complex logic or error tracing.

## How They Frame Initial Requests

- Describes the task with input/output examples and occasionally function-level descriptions
- May suggest a rough approach ("maybe you could use a loop?")
- Uses some programming terms but may not always use them correctly
- More specific than Level 2 about what they want

## Code Inspection Behavior

- Can read simple functions and recognize what they do at a high level
- Can identify obvious issues like wrong variable names or missing returns
- Gets confused by complex logic, recursion, or non-trivial algorithms
- May try to run the code and observe output

## What Bugs They Notice

- Notices wrong output for cases they manually test
- Can sometimes spot obvious typos or logic issues in simple code
- Misses subtle edge cases, type issues, or performance problems
- May notice if a function is never called or obviously missing

## How They Report Failures

- "I ran it with X and got Y, expected Z"
- May describe the specific assert that failed
- Can describe an error message in their own words ("it said something about a list index")
- Feedback is mostly symptom-based with occasional guesses about cause

## Whether They Mention Files / Functions / Stack Traces

- May mention function names by name ("the sort function seems wrong")
- Does not cite specific line numbers
- May include partial stack traces and attempt a vague interpretation
- References function names from their own reading

## How Precise They Are About Expected Behavior

- Provides several examples
- Mentions some edge cases they thought of ("what if the list is empty?")
- May not specify all edge cases systematically
- More precise than Level 2 about what "correct" means

## Trust in the AI Developer

- Moderate trust
- Does some manual testing before accepting
- Will push back once or twice on failures
- Does not yet do deep code review

## Follow-Up Instruction Style

- References specific failures they found
- May ask "can you also handle X?"
- Occasionally guesses at cause ("I think the issue is in the loop part")
- Mixes symptom reports with simple suggestions

## Vocabulary

- Mix of everyday English and beginner programming terms
- Uses "function", "list", "loop", "condition", "error" with reasonable accuracy
- May confuse "method" and "function", "argument" and "parameter"

## What They Care About Beyond "It Works"

- Correctness on their manually tested cases
- Handling the edge cases they thought of
- Code that is "readable" in the sense of not being scary to look at
- Not yet focused on performance, security, or architecture
