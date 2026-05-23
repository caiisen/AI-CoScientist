![AI-CoScientist](https://storage.googleapis.com/gweb-research2023-media/images/AICoScientist-1-Components.width-1250.png)

# AI-CoScientist

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/swarms-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A multi-agent AI framework for collaborative scientific research, implementing the "Towards an AI Co-Scientist" methodology with tournament-based hypothesis evolution, peer review systems, and intelligent agent orchestration.

## Features

🧠 **Multi-Agent Architecture**: Specialized agents for hypothesis generation, peer review, ranking, evolution, and meta-analysis  
🏆 **Tournament-Based Selection**: Elo rating system for hypothesis ranking through pairwise comparisons  
📊 **Comprehensive Review System**: Scientific soundness, novelty, testability, and impact assessment  
🔄 **Iterative Refinement**: Meta-review guided evolution with strategic hypothesis improvement  
🎯 **Diversity Control**: Proximity analysis to maintain hypothesis diversity and reduce redundancy  
📈 **Execution Metrics**: Detailed performance tracking and agent timing analytics  
💾 **State Persistence**: Save and resume research workflows with agent state management  
🛡️ **Robust Error Handling**: Graceful fallbacks and recovery mechanisms for production reliability
📚 **Optional Literature Grounding**: OpenAlex semantic search context for generation and review with DOI tracking

## Architecture

The AI-CoScientist framework consists of the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Generation      │    │ Reflection      │    │ Ranking         │
│ Agent           │───▶│ Agent           │───▶│ Agent           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Evolution       │    │ Meta-Review     │    │ Tournament      │
│ Agent           │◀───│ Agent           │    │ Agent           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Proximity       │    │ Supervisor      │    │ Conversation    │
│ Agent           │    │ Agent           │    │ Manager         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Workflow Process

1. **Generation Phase**: Create initial hypotheses based on research goal
2. **Reflection Phase**: Peer review each hypothesis for scientific merit
3. **Ranking Phase**: Order hypotheses by review scores
4. **Tournament Phase**: Pairwise comparisons with Elo rating updates
5. **Meta-Review Phase**: Synthesize insights across all reviews
6. **Evolution Phase**: Refine top hypotheses based on feedback
7. **Proximity Analysis**: Cluster similar hypotheses for diversity control
8. **Iteration**: Repeat refinement cycles for continuous improvement

## Installation

### Prerequisites

- Python 3.10 or higher
- API access to LLM providers (OpenAI, Anthropic, Google, etc.)

### Install from PyPI

```bash
pip3 install -U ai-coscientist
```

### Install from Source


```bash

git clone https://github.com/The-Swarm-Corporation/AI-CoScientist.git
cd AI-CoScientist
pip install -e .
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENALEX_EMAIL=your_email@example.com
```

### Supported model providers

AI-CoScientist routes models through `swarms.Agent` and LiteLLM, so any LiteLLM provider prefix works:

| Provider | `model_name` example | Env var |
|---|---|---|
| OpenAI | `gpt-4.1`, `openai/gpt-5` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-sonnet-4` | `ANTHROPIC_API_KEY` |
| Gemini | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| DeepSeek | `deepseek/deepseek-v4-pro` | `DEEPSEEK_API_KEY` |

For OpenAI- or Anthropic-compatible proxies, such as one-api, new-api, or litellm-proxy:

```python
AIScientistFramework(
    model_name="anthropic/mimo-v2.5-pro",
    llm_base_url="https://your-proxy.example.com/anthropic",
    llm_api_key="sk-...",
)
```

Notes for third-party APIs:

- Always include a LiteLLM provider prefix in `model_name`. Use
  `openai/...` for OpenAI-compatible endpoints and `anthropic/...` for
  Anthropic-compatible endpoints.
- Match the provider prefix to the endpoint protocol. For example,
  `model_name="anthropic/mimo-v2.5-pro"` should use an Anthropic
  endpoint, while `model_name="openai/mimo-v2.5-pro"` should use an
  OpenAI-compatible `/v1` endpoint.
- `llm_base_url` is the provider base URL. Anthropic-compatible
  clients append `/v1/messages` internally, so a base URL ending in
  `/anthropic` may still call `/anthropic/v1/messages`.
- Some proxies require provider-specific auth headers. Mimo's
  Anthropic-compatible endpoint uses `api-key`, so pass it with
  `llm_extra_headers`.
- `LLM_BASE_URL` and `LLM_API_KEY` in `.env` are not read
  automatically; pass them explicitly or load them with `os.getenv()`
  in your application code.

```python
AIScientistFramework(
    model_name="anthropic/mimo-v2.5-pro",
    llm_base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
    llm_api_key="tp-...",
    llm_extra_headers={"api-key": "tp-..."},
)
```


## Quick Start

```python
from ai_coscientist import AIScientistFramework

# Initialize the AI Co-scientist Framework
ai_coscientist = AIScientistFramework(
    model_name="gpt-4o-mini",
    max_iterations=3,
    hypotheses_per_generation=10,
    tournament_size=8,
    evolution_top_k=3,
    verbose=True
)

# Define your research goal
research_goal = "Develop novel approaches for improving reasoning capabilities in large language models"

# Run the research workflow
results = ai_coscientist.run_research_workflow(research_goal)

# Access the results
print(f"Generated {len(results['top_ranked_hypotheses'])} top hypotheses")
for i, hypothesis in enumerate(results['top_ranked_hypotheses'], 1):
    print(f"{i}. {hypothesis['text']}")
    print(f"   Elo Rating: {hypothesis['elo_rating']}")
    print(f"   Win Rate: {hypothesis['win_rate']}%")
```

### Literature Search

AI-CoScientist can optionally ground generation and reflection with OpenAlex
semantic search. It is disabled by default to keep the base workflow unchanged.

```python
import os
from ai_coscientist import AIScientistFramework

ai_coscientist = AIScientistFramework(
    model_name="gpt-4o-mini",
    enable_literature_search=True,
    literature_top_n=10,
    llm_max_tokens=8192,
    llm_context_length=64000,
    openalex_email=os.getenv("OPENALEX_EMAIL"),
)
```

When enabled, the framework retrieves OpenAlex papers for the research goal and
for each hypothesis review. Results are passed to the generation and reflection
agents as `title`, `abstract`, and `doi`, and final hypotheses include
`justification` and `citations` fields.

OpenAlex semantic search limits used by this integration:

- Query text is trimmed to 2000 characters before each request.
- `literature_top_n` is capped at 50 papers per query.
- Requests are limited to 1 request per second.
- Provide `OPENALEX_EMAIL` to use the OpenAlex polite pool. Current free usage
  is expected to be about 1000 papers per day for this workflow.

Model and literature sizing affect cost, latency, and JSON reliability:

- `llm_max_tokens` controls the maximum size of each agent response. The
  default is `8192`, which gives the reflection agent enough room to return
  review JSON with scores and detailed feedback. Higher values can reduce
  truncation but may be capped by the provider and can increase completion
  cost.
- `llm_context_length` is passed to `swarms.Agent` as the context window hint.
  The default is `64000`, which reduces unnecessary context compression when
  literature context is enabled. It does not increase output length; use
  `llm_max_tokens` for that.
- `literature_top_n` controls papers fetched for the research goal and each
  hypothesis review. Larger values add more grounding evidence but increase
  prompt tokens, OpenAlex requests, LLM latency, and the chance of context
  compression.
- `debug_failed_responses=False` by default. Set it to `True` to write failed
  reflection responses under `base_path` for diagnosing truncated or malformed
  JSON.

---

## Architecture

The AI-CoScientist framework consists of 8 specialized agents:

- **Generation Agent**: Creates novel research hypotheses
- **Reflection Agent**: Peer review and scientific critique
- **Ranking Agent**: Hypothesis ranking and selection
- **Evolution Agent**: Hypothesis refinement and improvement
- **Meta-Review Agent**: Cross-hypothesis insight synthesis
- **Proximity Agent**: Similarity analysis and diversity control
- **Tournament Agent**: Pairwise hypothesis comparison
- **Supervisor Agent**: Workflow orchestration and planning

## Advanced Usage

### Custom Configuration

```python
ai_coscientist = AIScientistFramework(
    model_name="claude-3-sonnet",
    max_iterations=5,
    base_path="./custom_states",
    verbose=True,
    tournament_size=12,
    hypotheses_per_generation=15,
    evolution_top_k=5,
    llm_max_tokens=8192,
    llm_context_length=64000,
    enable_literature_search=False,
)
```

### State Management

```python
# Save agent states
ai_coscientist.save_state()

# Load previous states
ai_coscientist.load_state()
```

### Results Analysis

```python
results = ai_coscientist.run_research_workflow(research_goal)

# Execution metrics
metrics = results['execution_metrics']
print(f"Total time: {results['total_workflow_time']:.2f}s")
print(f"Hypotheses generated: {metrics['hypothesis_count']}")
print(f"Reviews completed: {metrics['reviews_count']}")
print(f"Tournament rounds: {metrics['tournaments_count']}")

# Meta-review insights
insights = results['meta_review_insights']
print("Strategic recommendations:", insights.get('strategic_recommendations'))
```

## Documentation

For detailed documentation, see [DOCS.md](DOCS.md).


## 🤝 Contributing

We welcome contributions! Please feel free to open an issue or submit a pull request.
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite both the original paper and this software implementation.

```bibtex
@article{gottweis2024towards,
    title={Towards an AI co-scientist},
    author={Juraj Gottweis and Wei-Hung Weng and Alexander Daryin and Tao Tu and Anil Palepu and Petar Sirkovic and Artiom Myaskovsky and Felix Weissenberger and Keran Rong and Ryutaro Tanno and Khaled Saab and Dan Popovici and Jacob Blum and Fan Zhang and Katherine Chou and Avinatan Hassidim and Burak Gokturk and Amin Vahdat and Pushmeet Kohli and Yossi Matias and Andrew Carroll and Kavita Kulkarni and Nenad Tomasev and Vikram Dhillon and Eeshit Dhaval Vaishnav and Byron Lee and Tiago R D Costa and José R Penadés and Gary Peltz and Yunhan Xu and Annalisa Pawlosky and Alan Karthikesalingam and Vivek Natarajan},
    year={2024},
    institution={Google Cloud AI Research, Google Research, Google DeepMind, Houston Methodist, Sequome, Fleming Initiative and Imperial College London, Stanford University},
    url={https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf}
}

@software{ai_coscientist_framework,
    title={AI-CoScientist: A Multi-Agent Framework for Collaborative Scientific Research},
    author={The Swarm Corporation},
    year={2024},
    url={https://github.com/The-Swarm-Corporation/AI-CoScientist}
}
```

## 🔗 Related Work

- [Original Paper](https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf) - Towards an AI co-scientist
- [Swarms Framework](https://github.com/kyegomez/swarms) - Multi-agent AI orchestration
- [Google Research](https://research.google) - Original research institution

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/The-Swarm-Corporation/AI-CoScientist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/The-Swarm-Corporation/AI-CoScientist/discussions)
- **Email**: kye@swarms.world
- **Discord**: [Join our community](https://discord.gg/swarms-999382051935506503)

## 📝 TODO

- [ ] **Fix state saving**: Improve agent state persistence and resume functionality
- [ ] **Export hypothesis**: Add JSON/CSV export capabilities for generated hypotheses
- [ ] **Improve Elo rating**: Enhance tournament selection algorithm and rating calculations
- [ ] **Implement hypothesis validation**: Add automated testing framework for hypothesis quality
- [ ] **Enhance agent prompts**: Optimize system prompts for better scientific reasoning
- [x] **Add literature integration**: Connect with OpenAlex semantic search for knowledge grounding
- [ ] **Performance optimization**: Implement parallel agent execution and caching
- [ ] **Add visualization**: Create hypothesis evolution and tournament bracket visualizations
- [ ] **Extend model support**: Add support for more LLM providers and local models

---

<p align="center">
  <strong>Built with Swarms for advancing AI-powered scientific research</strong>
</p>
