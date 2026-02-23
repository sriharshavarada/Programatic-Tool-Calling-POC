# Programmatic Tool Calling POC (Layman Guide)

This document explains what this POC is trying to teach, in plain English.

## What This POC Is About

Traditional AI usage is often:

- You ask a question in natural language
- The model thinks in natural language
- The model replies in natural language

That works for many cases, but it becomes weak when the task requires:

- repeated steps
- data fetching
- calculations
- retries
- structured workflows

This POC shows a different pattern:

- Let the model **write code**
- Run that code in a controlled environment
- Let that code call tools/data sources
- Trust the **executed result**, not just the model's words

In simple terms:

- Less "just trust what the model says"
- More "let the model produce a program and observe what it actually does"

## What We Built (3 Modes)

This project compares **three ways** to solve the same business task (healthcare denied-claims analysis):

### 1. Local (LangGraph)

- Claude generates Python code
- That code is executed on **your machine**
- The code calls your local tool (`healthcare_claims_get_denials`)
- The result is summarized

This shows:

- "Agent writes code -> I execute it locally"

### 2. Sandbox (LangGraph + Anthropic programmatic tool calling)

- Claude uses Anthropic-managed **code execution** (remote sandbox)
- The sandboxed code requests data through a tool call
- Your app runs the local tool and returns the result
- Sandbox execution resumes and computes the result
- Claude returns the final answer

This shows:

- "Code runs remotely, but my app still controls local tools"

### 3. Traditional (Classic tool calling)

- No generated Python code
- No code execution tool
- Claude directly calls the business tool (`healthcare_claims_get_denials`) in a classic tool loop
- Claude summarizes the results

This shows:

- "Agent tool calling without code generation"

## Why This Is Useful

The point is not only "which one is best."

The point is to understand tradeoffs:

- **Local code execution** gives maximum control and debugging
- **Sandbox code execution** gives isolation and managed runtime behavior
- **Traditional tool calling** is simpler and often enough for many workflows

## The Trust Shift (Most Important Idea)

The big shift is:

- old style: trust the model's natural-language reasoning
- new style: trust the code it wrote + runtime logs + tool outputs

This is why the UI shows:

- generated code
- prompts
- logs
- LLM message timeline (TX/RX)
- token usage

You are not just looking at the final answer.
You are inspecting the process.

## A Very Important Clarification (Our Sandbox POC vs "Ideal" Architecture)

In our sandbox POC, the local tool results are sent back using Anthropic's `tool_result` message flow.

That means:

- the raw tool payload can still enter the LLM/API conversation channel
- token usage may still increase if the payload is large

So this POC is:

- **protocol-correct** for Anthropic client-tool integration
- but **not automatically token-optimized**

Token savings only happen if you design the tool boundary carefully, for example:

- return aggregates instead of raw rows
- return references/handles instead of full data
- let computation happen closer to the data

## What To Look At in the UI

The swim lanes help you compare all three modes side by side.

For each lane, inspect:

- **Prompt Used**: what the model was asked
- **Integration Snippet**: how the API/tool loop is wired
- **Generated Code**: only for Local/Sandbox modes
- **Final Output**: what Claude finally says
- **Logs**: what actually happened during execution
- **LLM Conversation Timeline (TX/RX)**: exact data sent/received
- **Token Usage**: model usage from Anthropic API responses

This makes the POC useful as an educational tool, not just a demo.

## Why Inconsistencies Can Still Happen

Even with code execution and tools, the final narrative summary can still be wrong.

Example causes:

- incomplete tool coverage
- wrong field name assumptions
- summarization mismatch
- partial context in a final summarization step

That is why production systems add:

- host-side verification
- schema validation
- consistency checks
- structured outputs (JSON)

## Where the Data Is in This POC

The demo dataset is currently in code (in-memory), not in a database.

See:

- `shared_tool.py` (`_DENIALS_BY_DATE`, `_EXPANDED_DENIALS_BY_DATE`)

The tool:

- `healthcare_claims_get_denials(service_date)`

returns rows from that in-memory dataset.

## How This Relates to the Cloudflare "Code Mode MCP" Idea

Cloudflare's "Code Mode MCP" is **not the same implementation** as Anthropic programmatic tool calling, but conceptually it points in a very similar direction:

- give the model a constrained environment
- give it a small, reliable tool surface
- let it write code as the execution plan
- rely more on **code execution** than on natural-language reasoning alone

That is the broader trend:

- not "trust NLP only"
- but "let the model write code, run it, inspect it, and trust observed behavior"

## Useful Links

### Anthropic (Programmatic Tool Calling / Tools)

- Anthropic Programmatic Tool Calling: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
- Anthropic Tool Use (implementation patterns): <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use>
- Anthropic Code Execution Tool: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool>

### Cloudflare (Conceptually Related Direction)

- Cloudflare Code Mode MCP blog post: <https://blog.cloudflare.com/code-mode-mcp/>

## One-Line Summary

This POC demonstrates that modern AI agents are increasingly moving from "natural-language-only reasoning" toward "code-as-plan + tool execution + observable runtime behavior."
