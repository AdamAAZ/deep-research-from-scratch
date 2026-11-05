# Deep Research Multi-Agent

A LangGraph multi-agent research system that scopes a user request, delegates focused web research to parallel researcher agents, and synthesizes the findings into a cited report.

## Architecture

```text
User request
  → clarification and research brief
  → supervisor delegates focused topics
  → parallel researchers use Tavily and reflection
  → findings are compressed and aggregated
  → final report is generated
```

The runnable graphs are defined in `langgraph.json`:

- `scope_research` — clarification and brief generation
- `research_agent` — single Tavily researcher
- `research_agent_mcp` — local-filesystem MCP researcher
- `research_agent_supervisor` — supervisor plus parallel researchers
- `research_agent_full` — complete end-to-end agent

## Setup

Requirements: Python 3.11–3.13 and Node.js/npx (for the MCP graph).

On Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

The `.venv` directory is local to your machine and is ignored by Git. Choose one provider below. Both setups use `TAVILY_API_KEY` for web research; the local-file MCP graph does not need Tavily. LangSmith tracing is off by default. To enable it, add your `LANGSMITH_API_KEY` and set `LANGSMITH_TRACING=true` in `.env`.

### OpenAI setup (default)

In `.env`, set `MODEL_PROVIDER=openai`, `OPENAI_API_KEY`, and `TAVILY_API_KEY`. Every model call then uses OpenAI. The default roles are:

| Role | Model |
| --- | --- |
| Scoping, webpage summaries, research compression | `gpt-4.1-mini` |
| Research, supervisor, final report | `gpt-4.1` |

The smaller model handles routine extraction and summarization; the larger model handles tool use, delegation, and synthesis. Published standard API rates are $0.40 input / $1.60 output per million tokens for [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) and $2 / $8 for [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1). These are quality and cost starting points, not the absolute cheapest models or a measured optimum for this project.

### Anthropic setup

Install the optional Anthropic integration:

```powershell
.\.venv\Scripts\python -m pip install ".[anthropic]"  # Windows
```

```bash
./.venv/bin/python -m pip install '.[anthropic]'  # macOS/Linux
```

In `.env`, set `MODEL_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, and `TAVILY_API_KEY`. `OPENAI_API_KEY` is not needed. Every model call then uses Anthropic:

| Role | Model |
| --- | --- |
| Scoping, webpage summaries, research compression | `claude-haiku-4-5` |
| Research, supervisor, final report | `claude-sonnet-4-6` |

Published standard API rates are $1 input / $5 output per million tokens for [Claude Haiku 4.5](https://platform.claude.com/docs/en/models/haiku-4-5/overview) and $3 / $15 for [Claude Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview). Check provider pricing before use; total cost also depends on how many research loops and web searches a request needs.

You can override a role with `SCOPE_MODEL`, `RESEARCH_MODEL`, `SUPERVISOR_MODEL`, `SUMMARIZE_MODEL`, `COMPRESS_MODEL`, or `WRITER_MODEL` using a value such as `anthropic:claude-sonnet-4-6`. A cross-provider override also needs that provider's package and API key.

## Run with LangGraph

On Windows (PowerShell):

```powershell
.\.venv\Scripts\langgraph dev
```

On macOS/Linux:

```bash
./.venv/bin/langgraph dev
```

The command starts a local server and opens LangGraph Studio at `smith.langchain.com`. Seeing a graph there means the server and Studio connected; the graph is a view of the workflow, not the research result.

To run your first research request:

1. Keep the terminal running. In Studio, select `research_agent_full` from the graph selector.
2. Start a new thread and click add message ensuring it is set to `Human`.
3. Submit your research question and clarification if necessary.
4. Watch the nodes run. The final output appears under `final_report` (and in the final assistant message). If the agent asks a clarifying question, answer it in the same thread.

If the run fails, check the terminal for the error and confirm that `.env` has the selected provider's API key and `TAVILY_API_KEY`. A graph load error happens before you submit input; a model or search error appears during the run.

You can also import the compiled graphs from the `src` package in your own application or frontend backend (for example, `from src.research_agent_full import agent`). This tutorial uses `src` as its import package name because the source files live directly in that folder.

## Try the local-file researcher

In LangGraph Studio, select `research_agent_mcp` and run it with this input:

```json
{
  "researcher_messages": [
    {"role": "user", "content": "Using the local coffee shop research file, compare the listed San Francisco coffee shops by specialty and location."}
  ],
  "research_topic": "Compare the coffee shops in the local San Francisco research file"
}
```

The MCP filesystem server reads the sample document at `src/files/coffee_shops_sf.md`. The graph returns compressed research findings and raw notes from that file. This is a demonstration of local document research; the sample is not a current or independently verified guide to San Francisco coffee shops.

## Prompt injection evaluation

The `security_eval/` directory is an extension to this tutorial. It tests whether an instruction embedded in a retrieved search result can steer the research report away from the user's request. Two fictional research topics each have clean and adversarial search results. The harness runs each version with the baseline and an opt-in mitigation, for eight model trials. The mitigation extracts question-relevant evidence and citations from retrieved tool messages before writing the report; it is enabled only during the mitigated trials.

The harness supplies fixtures at the researcher tool-response boundary and then runs the real researcher subgraph, compression model, and writer model. It does not measure whether the model chooses to search. The recorded eight-run pass retained all expected facts in both conditions and had no adversarial marker hits in either condition, so it does not establish that the mitigation reduced attacks. See [the results and limitations](security_eval/result.md) and [the fixtures](security_eval/fixtures.json).

To check fixture setup without API calls, run `.\.venv\Scripts\python security_eval\run_eval.py --self-test` on Windows. To repeat all eight model trials, run `.\.venv\Scripts\python security_eval\run_eval.py --run` with your model API key configured. This replaces the recorded `result.md` with the new pass.

## Project layout

```text
src/
  research_agent_full.py       # end-to-end graph
  research_agent_scope.py      # clarification and brief generation
  multi_agent_supervisor.py    # delegation and parallel coordination
  research_agent.py             # Tavily research loop
  research_agent_mcp.py         # MCP filesystem research loop
  prompts.py                    # system and report prompts
  state_*.py                    # graph state and schemas
  utils.py                      # Tavily/search utilities and tools
  files/                        # local MCP research fixture
langgraph.json                  # graph/server entry points
pyproject.toml                  # package and dependencies
requirements.txt                # pip installation and LangGraph CLI
security_eval/                  # prompt injection harness, fixtures, and result
```

Never commit `.env` or real API keys.
