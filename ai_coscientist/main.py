"""
AIScientistFramework: A multi-agent system for AI co-scientist based on
"Towards an AI co-scientist" research paper.
Implements hypothesis generation, review, ranking, and evolution using a tournament approach.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Protocol,
)
from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.conversation import Conversation
from loguru import logger

from ai_coscientist.literature import LiteratureService

load_dotenv()


class AgentRole(Enum):
    """Define the possible roles for agents in the AI co-scientist system."""

    GENERATION = "generation"
    REFLECTION = "reflection"
    RANKING = "ranking"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"
    PROXIMITY = "proximity"
    SUPERVISOR = "supervisor"
    TOURNAMENT = "tournament"


# Type definitions for better type safety
class ReviewScores(TypedDict):
    """Type definition for review scores."""

    scientific_soundness: int
    novelty: int
    relevance: int
    testability: int
    clarity: int
    potential_impact: int


class DetailedFeedback(TypedDict):
    """Type definition for detailed feedback."""

    scientific_soundness: str
    novelty: str
    relevance: str
    testability: str
    clarity: str
    potential_impact: str


class HypothesisReview(TypedDict):
    """Type definition for hypothesis review."""

    hypothesis_text: str
    review_summary: str
    scores: ReviewScores
    safety_ethical_concerns: str
    detailed_feedback: DetailedFeedback
    constructive_feedback: str
    overall_score: float


class AgentExecutionMetrics(TypedDict):
    """Type definition for agent execution metrics."""

    total_time: float
    calls: int
    avg_time: float


class ExecutionMetrics(TypedDict):
    """Type definition for execution metrics."""

    total_time: float
    hypothesis_count: int
    reviews_count: int
    tournaments_count: int
    evolutions_count: int
    agent_execution_times: Dict[str, AgentExecutionMetrics]


class SimilarHypothesis(TypedDict):
    """Type definition for similar hypothesis in clustering."""

    text: str
    similarity_degree: str


class SimilarityCluster(TypedDict):
    """Type definition for similarity cluster."""

    cluster_id: str
    cluster_name: str
    central_theme: str
    similar_hypotheses: List[SimilarHypothesis]
    synthesis_potential: str


class ProximityAnalysisResult(TypedDict):
    """Type definition for proximity analysis result."""

    similarity_clusters: List[SimilarityCluster]
    diversity_assessment: str
    redundancy_assessment: str


class TournamentJudgment(TypedDict):
    """Type definition for tournament judgment."""

    research_goal: str
    hypothesis_a: str
    hypothesis_b: str
    winner: str
    judgment_explanation: Dict[str, str]
    decision_summary: str
    confidence_level: str


class WorkflowResult(TypedDict):
    """Type definition for workflow result."""

    top_ranked_hypotheses: List[Dict[str, Any]]
    meta_review_insights: Dict[str, Any]
    conversation_history: str
    execution_metrics: ExecutionMetrics
    total_workflow_time: float


class JSONParseable(Protocol):
    """Protocol for objects that can be safely parsed from JSON."""

    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass
class Hypothesis:
    """
    Represents a research hypothesis.

    Attributes:
        text (str): The text of the hypothesis.
        elo_rating (int): Elo rating for ranking (initially 1200).
        reviews (List[HypothesisReview]): List of review feedback for the hypothesis.
        score (float): Overall score based on reviews (0.0-1.0).
        similarity_cluster_id (Optional[str]): ID of the similarity cluster.
        evolution_history (List[str]): History of evolutions for this hypothesis.
        generation_timestamp (float): When the hypothesis was generated.
        win_count (int): Number of tournament wins.
        loss_count (int): Number of tournament losses.
    """

    text: str
    justification: str = ""
    citations: List[str] = field(default_factory=list)
    elo_rating: int = 1200
    reviews: List[HypothesisReview] = field(default_factory=list)
    score: float = 0.0
    similarity_cluster_id: Optional[str] = None
    evolution_history: List[str] = field(default_factory=list)
    generation_timestamp: float = field(default_factory=time.time)
    win_count: int = 0
    loss_count: int = 0

    def update_elo(
        self, opponent_elo: int, win: bool, k_factor: int = 32
    ) -> None:
        """
        Update the Elo rating based on a tournament match outcome.

        Args:
            opponent_elo (int): The Elo rating of the opponent.
            win (bool): Whether this hypothesis won the match.
            k_factor (int): K-factor for Elo calculation, controlling update magnitude.
        """
        if not isinstance(opponent_elo, int) or not isinstance(
            win, bool
        ):
            logger.error(
                f"Invalid types for Elo update: opponent_elo={type(opponent_elo)}, win={type(win)}"
            )
            return

        expected_score = 1 / (
            1 + 10 ** ((opponent_elo - self.elo_rating) / 400)
        )
        actual_score = 1.0 if win else 0.0
        self.elo_rating += int(
            k_factor * (actual_score - expected_score)
        )

        # Update win/loss count
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert the hypothesis to a dictionary representation."""
        return {
            "text": self.text,
            "justification": self.justification,
            "citations": self.citations,
            "elo_rating": self.elo_rating,
            "score": self.score,
            "reviews": self.reviews,
            "similarity_cluster_id": self.similarity_cluster_id,
            "evolution_history": self.evolution_history,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_matches": self.win_count + self.loss_count,
            "win_rate": round(
                self.win_count
                / max(1, (self.win_count + self.loss_count))
                * 100,
                2,
            ),
        }


class AIScientistFramework:
    """
    A multi-agent system framework for AI co-scientist, designed to generate
    and refine research hypotheses using tournament-based evolution.

    Attributes:
        model_name (str): Name of the LLM model to use for agents.
        max_iterations (int): Maximum number of iterations for the research workflow.
        base_path (Path): Base path for saving agent states.
        verbose (bool): Enable verbose logging.
        conversation (Conversation): Tracks the conversation history.
        hypotheses (List[Hypothesis]): List to store generated hypotheses.
        tournament_size (int): Number of hypotheses to include in each tournament round.
        hypotheses_per_generation (int): Number of hypotheses to generate initially.
        evolution_top_k (int): Number of top hypotheses to evolve in each iteration.
        llm_base_url (Optional[str]): Base URL for a compatible LLM proxy.
        llm_api_key (Optional[str]): API key for a compatible LLM proxy.
        llm_extra_headers (Optional[Dict[str, str]]): Extra headers for LLM calls.
        llm_max_tokens (int): Maximum tokens to request for each agent output.
        llm_context_length (Optional[int]): Agent context window hint for swarms.
        enable_literature_search (bool): Enable OpenAlex literature context.
        literature_top_n (int): Number of papers to retrieve per query, max 50.
        openalex_email (Optional[str]): Email for OpenAlex polite pool.
        debug_failed_responses (bool): Persist failed reflection responses for debugging.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        max_iterations: int = 3,
        base_path: Optional[str] = None,
        verbose: bool = False,
        tournament_size: int = 8,
        hypotheses_per_generation: int = 10,
        evolution_top_k: int = 3,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_extra_headers: Optional[Dict[str, str]] = None,
        llm_max_tokens: int = 8192,
        llm_context_length: Optional[int] = 64000,
        enable_literature_search: bool = False,
        literature_top_n: int = 10,
        openalex_email: Optional[str] = None,
        debug_failed_responses: bool = False,
    ) -> None:
        """Initialize the AIScientistFramework system."""
        # Type validation
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be str, got {type(model_name)}"
            )
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError(
                f"max_iterations must be positive int, got {max_iterations}"
            )
        if not isinstance(verbose, bool):
            raise TypeError(
                f"verbose must be bool, got {type(verbose)}"
            )
        if not isinstance(enable_literature_search, bool):
            raise TypeError(
                "enable_literature_search must be bool, got "
                f"{type(enable_literature_search)}"
            )
        if not isinstance(literature_top_n, int) or literature_top_n < 1:
            raise ValueError(
                "literature_top_n must be positive int, got "
                f"{literature_top_n}"
            )
        if not isinstance(llm_max_tokens, int) or llm_max_tokens < 1:
            raise ValueError(
                f"llm_max_tokens must be positive int, got {llm_max_tokens}"
            )
        if (
            llm_context_length is not None
            and (
                not isinstance(llm_context_length, int)
                or llm_context_length < 1
            )
        ):
            raise ValueError(
                "llm_context_length must be a positive int or None, got "
                f"{llm_context_length}"
            )
        if not isinstance(debug_failed_responses, bool):
            raise TypeError(
                "debug_failed_responses must be bool, got "
                f"{type(debug_failed_responses)}"
            )

        self.model_name: str = model_name
        self.max_iterations: int = max_iterations
        self.base_path: Path = (
            Path(base_path)
            if base_path
            else Path("./ai_coscientist_states")
        )
        self.base_path.mkdir(exist_ok=True, parents=True)
        self.verbose: bool = verbose
        self.llm_base_url: Optional[str] = llm_base_url
        self.llm_api_key: Optional[str] = llm_api_key
        self.llm_extra_headers: Optional[Dict[str, str]] = (
            llm_extra_headers
        )
        self.llm_max_tokens: int = llm_max_tokens
        self.llm_context_length: Optional[int] = llm_context_length
        self.debug_failed_responses: bool = debug_failed_responses
        self._warned_context_length_override: bool = False
        self.enable_literature_search = enable_literature_search
        self._goal_literature: List[Dict[str, str]] = []
        self.literature_service: Optional[LiteratureService] = None
        if enable_literature_search:
            self.literature_service = LiteratureService(
                top_n=literature_top_n,
                mailto=openalex_email or os.getenv("OPENALEX_EMAIL"),
            )
            logger.info(
                "Literature search enabled "
                f"(top_n={self.literature_service.top_n})"
            )
        self.conversation: Conversation = Conversation()
        self.hypotheses: List[Hypothesis] = []

        # Tournament and evolution parameters
        self.tournament_size: int = tournament_size
        self.hypotheses_per_generation: int = (
            hypotheses_per_generation
        )
        self.evolution_top_k: int = evolution_top_k

        # Execution metrics
        self.start_time: Optional[float] = None
        self.execution_metrics: ExecutionMetrics = {
            "total_time": 0.0,
            "hypothesis_count": 0,
            "reviews_count": 0,
            "tournaments_count": 0,
            "evolutions_count": 0,
            "agent_execution_times": {},
        }

        # Initialize agents
        self._init_agents()
        logger.info(
            f"AIScientistFramework initialized with model: {model_name}"
            + (
                f", llm_base_url: {llm_base_url}"
                if llm_base_url
                else ""
            )
        )

    def _agent_kwargs(
        self, name: str, prompt: str, state_file: str
    ) -> Dict[str, Any]:
        """Build shared keyword arguments for a swarms.Agent."""
        kwargs: Dict[str, Any] = {
            "agent_name": name,
            "system_prompt": prompt,
            "model_name": self.model_name,
            "max_loops": 1,
            "saved_state_path": str(self.base_path / state_file),
            "verbose": self.verbose,
            "max_tokens": self.llm_max_tokens,
        }
        if self.llm_context_length is not None:
            kwargs["context_length"] = self.llm_context_length
        if self.llm_base_url:
            kwargs["llm_base_url"] = self.llm_base_url
        if self.llm_api_key:
            kwargs["llm_api_key"] = self.llm_api_key
        llm_args: Dict[str, str] = {}
        if self.llm_base_url:
            llm_args["base_url"] = self.llm_base_url
        if self.llm_api_key:
            llm_args["api_key"] = self.llm_api_key
        if self.llm_extra_headers:
            llm_args["extra_headers"] = self.llm_extra_headers
        if llm_args:
            kwargs["llm_args"] = llm_args
        return kwargs

    def _create_agent(
        self, name: str, prompt: str, state_file: str
    ) -> Agent:
        """Create an Agent and enforce settings swarms may override."""
        agent = Agent(**self._agent_kwargs(name, prompt, state_file))

        if self.llm_context_length is not None:
            actual_context_length = getattr(
                agent, "context_length", None
            )
            if actual_context_length != self.llm_context_length:
                if not self._warned_context_length_override:
                    logger.warning(
                        "Overriding swarms Agent context_length after "
                        "initialization because the installed swarms version "
                        "resets constructor context_length to "
                        f"{actual_context_length}."
                    )
                    self._warned_context_length_override = True
                agent.context_length = self.llm_context_length
            short_memory = getattr(agent, "short_memory", None)
            if short_memory is not None and hasattr(
                short_memory, "context_length"
            ):
                short_memory.context_length = self.llm_context_length
            logger.debug(
                f"{name} context_length={getattr(agent, 'context_length', None)}, "
                f"short_memory_context_length="
                f"{getattr(short_memory, 'context_length', None)}, "
                f"max_tokens={getattr(agent, 'max_tokens', None)}"
            )

        return agent

    def _init_agents(self) -> None:
        """Initialize all specialized agents with their roles and prompts."""
        try:
            self.generation_agent: Agent = self._create_agent(
                "HypothesisGenerator",
                self._get_generation_agent_prompt(),
                "generation_agent_state.json",
            )
            self.reflection_agent: Agent = self._create_agent(
                "HypothesisReflector",
                self._get_reflection_agent_prompt(),
                "reflection_agent_state.json",
            )
            self.ranking_agent: Agent = self._create_agent(
                "HypothesisRanker",
                self._get_ranking_agent_prompt(),
                "ranking_agent_state.json",
            )
            self.evolution_agent: Agent = self._create_agent(
                "HypothesisEvolver",
                self._get_evolution_agent_prompt(),
                "evolution_agent_state.json",
            )
            self.meta_review_agent: Agent = self._create_agent(
                "MetaReviewer",
                self._get_meta_review_agent_prompt(),
                "meta_review_agent_state.json",
            )
            self.proximity_agent: Agent = self._create_agent(
                "ProximityAnalyzer",
                self._get_proximity_agent_prompt(),
                "proximity_agent_state.json",
            )
            self.tournament_agent: Agent = self._create_agent(
                "TournamentJudge",
                self._get_tournament_agent_prompt(),
                "tournament_agent_state.json",
            )
            self.supervisor_agent: Agent = self._create_agent(
                "Supervisor",
                self._get_supervisor_agent_prompt(),
                "supervisor_agent_state.json",
            )
            logger.success("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    def _get_generation_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Generation Agent."""
        prompt = """You are a Hypothesis Generation Agent in an AI Co-scientist framework.
Your role is to generate novel and relevant research hypotheses based on a given research goal.

Consider current scientific literature and knowledge in the domain.
Focus on generating hypotheses that are:
- Novel and original
- Relevant to the research goal
- Potentially testable and falsifiable
- Scientifically sound
- Specific and well-defined

Each hypothesis should:
1. Challenge existing assumptions or extend current knowledge in the field
2. Be formulated as a clear statement that can be tested
3. Identify potential variables and relationships
4. Consider practical implications and significance
5. Balance ambition with feasibility

Output your hypotheses in JSON format. Provide a list of hypotheses, each with a clear and concise text description,
and brief justification explaining why it's novel and significant.

Example JSON Output:
{
  "hypotheses": [
    {
      "text": "Hypothesis text 1",
      "justification": "Brief explanation of novelty, significance, and scientific rationale"
    },
    {
      "text": "Hypothesis text 2",
      "justification": "Brief explanation of novelty, significance, and scientific rationale"
    },
    ...
  ]
}
"""
        if self.enable_literature_search:
            prompt += """
If the input contains a non-empty `literature_context` field, treat each entry
({title, abstract, doi}) as a real published work. Ground hypotheses in this
evidence and cite the DOI in the justification field whenever you draw on a
paper. If `literature_context` is empty or missing, proceed without citations.
"""
        return prompt

    def _get_reflection_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Reflection Agent (Reviewer)."""
        prompt = """You are a Hypothesis Reflection Agent, acting as a scientific peer reviewer.
Your task is to review and critique research hypotheses for correctness, novelty, quality, and potential safety/ethical concerns.

For each hypothesis, evaluate it based on the following criteria:
- Scientific Soundness (1-5): Is the hypothesis scientifically plausible and consistent with existing knowledge?
- Novelty (1-5): Does the hypothesis propose something new or original?
- Relevance (1-5): Is the hypothesis relevant to the stated research goal?
- Testability (1-5): Can the hypothesis be tested or investigated using scientific methods?
- Clarity (1-5): Is the hypothesis clearly and concisely stated?
- Potential Impact (1-5): If validated, what is the potential scientific or practical impact?
- Safety/Ethical Concerns: Are there any potential safety or ethical issues associated with investigating this hypothesis?

Provide a detailed review for each criterion, with specific feedback on strengths and weaknesses.
For the overall score, use a scale from 0.0 to 1.0, where:
- 0.0-0.2: Poor (multiple serious flaws)
- 0.2-0.4: Fair (notable deficiencies requiring substantial revision)
- 0.4-0.6: Good (promising but needs revisions)
- 0.6-0.8: Very Good (minor revisions needed)
- 0.8-1.0: Excellent (minimal or no revisions needed)

Output your review in JSON format:

IMPORTANT: Always emit `overall_score` and `scores` as the first two keys of
the JSON object. This keeps the review score parseable even if later feedback
sections are truncated by provider token limits.

Example JSON Output (for a single hypothesis):
{
  "overall_score": 0.8,
  "scores": {
    "scientific_soundness": 4,
    "novelty": 3,
    "relevance": 5,
    "testability": 4,
    "clarity": 5,
    "potential_impact": 4
  },
  "hypothesis_text": "The hypothesis being reviewed",
  "review_summary": "Overall summary of the review",
  "safety_ethical_concerns": "Specific concerns or 'None identified'",
  "detailed_feedback": {
    "scientific_soundness": "Specific feedback on scientific soundness",
    "novelty": "Specific feedback on novelty",
    "relevance": "Specific feedback on relevance",
    "testability": "Specific feedback on testability",
    "clarity": "Specific feedback on clarity",
    "potential_impact": "Specific feedback on potential impact"
  },
  "constructive_feedback": "Specific suggestions for improvement"
}
"""
        if self.enable_literature_search:
            prompt += """
If the input contains a non-empty `literature_context` field, use those papers
to validate scientific plausibility, flag conflicts with established findings,
and detect redundancy with existing work. Cite DOIs when relevant. If
`literature_context` is empty or missing, base your review on general knowledge.
"""
        return prompt

    def _get_ranking_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Ranking Agent."""
        return """You are a Hypothesis Ranking Agent. Your role is to rank a set of research hypotheses based on their review scores and other relevant criteria.

Rank the hypotheses from highest to lowest quality based on:
1. The overall scores provided by the Reflection Agents
2. The detailed feedback for each criterion
3. Scientific merit and potential impact
4. Novelty and originality
5. Feasibility of testing and verification

For each hypothesis, calculate a composite ranking score that synthesizes these factors.
Consider not just the average scores, but also the distribution across criteria - a hypothesis with consistently good scores
might be preferable to one with extremely high scores in some areas but poor scores in others.

Output the ranked hypotheses in JSON format, ordered from highest to lowest rank. Include the hypothesis text,
overall score, and a brief explanation for each ranking decision.

Example JSON Output:
{
  "ranked_hypotheses": [
    {
      "text": "Hypothesis text 1",
      "overall_score": 0.9,
      "ranking_explanation": "Ranked highest due to exceptional novelty, strong scientific soundness, and high testability"
    },
    {
      "text": "Hypothesis text 2",
      "overall_score": 0.85,
      "ranking_explanation": "Strong overall but ranked below hypothesis 1 due to slightly lower novelty"
    },
    ...
  ]
}
"""

    def _get_evolution_agent_prompt(self) -> str:
        """Prompt for the Hypothesis Evolution Agent (Refiner)."""
        return """You are a Hypothesis Evolution Agent. Your task is to refine and improve the top-ranked research hypotheses based on the reviews and meta-review insights.

For each hypothesis, carefully analyze the review feedback, meta-review insights, and then apply the following approaches to refine the hypothesis:

1. Enhance clarity and precision:
   - Eliminate ambiguous language
   - Ensure clear definition of variables and relationships
   - Improve the logical structure

2. Strengthen scientific soundness:
   - Address any identified theoretical weaknesses
   - Ensure alignment with established scientific principles
   - Incorporate relevant background knowledge

3. Increase novelty and originality:
   - Identify opportunities to introduce more innovative elements
   - Consider unconventional perspectives or approaches

4. Improve testability:
   - Make the hypothesis more amenable to empirical investigation
   - Consider specific experimental designs or methodologies
   - Ensure falsifiability

5. Address safety/ethical concerns:
   - Integrate ethical considerations
   - Propose safeguards or limitations when necessary

6. Consider hybridization:
   - Identify complementary hypotheses that could be combined
   - Merge strengths from multiple hypotheses when beneficial

7. Simplify when appropriate:
   - Remove unnecessary complexity
   - Focus on the most promising and impactful aspects

Output the refined hypotheses in JSON format, including the original text, the refined text, a summary of changes made, and justifications for each significant modification:

Example JSON Output (for a single hypothesis):
{
  "original_hypothesis_text": "Original hypothesis text",
  "refined_hypothesis_text": "Refined hypothesis text",
  "refinement_summary": "Summary of overall changes and improvements",
  "specific_refinements": [
    {
      "aspect": "clarity",
      "change": "Specific change made",
      "justification": "Reason for this modification"
    },
    {
      "aspect": "scientific_soundness",
      "change": "Specific change made",
      "justification": "Reason for this modification"
    },
    ...
  ]
}
"""

    def _get_meta_review_agent_prompt(self) -> str:
        """Prompt for the Meta-Review Agent."""
        return """You are a Meta-Review Agent. Your role is to synthesize insights from all the reviews of the research hypotheses.

Analyze all the reviews provided by the Reflection Agents across multiple hypotheses. Your goal is to:

1. Identify recurring patterns, themes, and trends:
   - Common strengths across hypotheses
   - Common weaknesses or limitations
   - Recurring feedback themes from reviewers

2. Evaluate the hypothesis generation and review process:
   - Areas where the generation process could be improved
   - Potential gaps in the review criteria or approach
   - Consistency and quality of reviews

3. Provide strategic guidance for hypothesis refinement:
   - High-level directions for improving hypothesis quality
   - Specific areas where the evolution agent should focus
   - Potential new directions or perspectives to explore

4. Assess the overall research direction:
   - Alignment with the original research goal
   - Potential for scientific impact
   - Most promising avenues for further exploration

5. Identify potential connections:
   - Relationships between different hypotheses
   - Possibilities for synthesizing complementary ideas
   - Cross-cutting themes or approaches

Output your meta-review insights and recommendations in JSON format:

Example JSON Output:
{
  "meta_review_summary": "Overall summary of meta-review analysis",
  "recurring_themes": [
    {
      "theme": "Theme 1",
      "description": "Detailed description of the theme",
      "frequency": "Number or percentage of hypotheses showing this theme"
    },
    ...
  ],
  "strengths": [
    "Common strength 1 identified across hypotheses",
    "Common strength 2 identified across hypotheses",
    ...
  ],
  "weaknesses": [
    "Common weakness 1 identified across hypotheses",
    "Common weakness 2 identified across hypotheses",
    ...
  ],
  "process_assessment": {
    "generation_process": "Assessment of hypothesis generation process",
    "review_process": "Assessment of review process",
    "evolution_process": "Assessment of hypothesis evolution process"
  },
  "strategic_recommendations": [
    {
      "focus_area": "Area for improvement",
      "recommendation": "Specific recommendation",
      "justification": "Reasoning behind this recommendation"
    },
    ...
  ],
  "potential_connections": [
    {
      "related_hypotheses": ["Hypothesis 1", "Hypothesis 2"],
      "connection_type": "Type of relationship (complementary, contradictory, etc.)",
      "synthesis_opportunity": "Potential for combining or relating these hypotheses"
    },
    ...
  ]
}
"""

    def _get_proximity_agent_prompt(self) -> str:
        """Prompt for the Proximity Agent (Similarity Analysis)."""
        return """You are a Proximity Agent, focused on analyzing the similarity between research hypotheses.

Your task is to identify hypotheses that are semantically similar or redundant to maintain diversity in the hypothesis pool.
This helps in clustering related hypotheses and de-duplicating similar ones to ensure diversity in the generated set.

For each hypothesis, analyze:
1. Core scientific concepts and principles involved
2. Key variables and relationships being examined
3. Underlying assumptions and theoretical frameworks
4. Methodological approaches suggested or implied
5. Potential applications or implications

Based on these factors, identify clusters of hypotheses that are conceptually related or address similar research questions.
Assign each hypothesis to a cluster, and give each cluster a descriptive name that captures its unifying theme.

For each cluster, identify:
- The central theme or concept
- The distinguishing features between hypotheses within the cluster
- The degree of similarity/redundancy between hypotheses (high, medium, low)
- Potential for synthesis or combination within the cluster

Output your findings in JSON format:

Example JSON Output:
{
  "similarity_clusters": [
    {
      "cluster_id": "cluster-1",
      "cluster_name": "Descriptive name for this cluster",
      "central_theme": "Brief description of the unifying concept",
      "similar_hypotheses": [
        {"text": "Hypothesis text A", "similarity_degree": "high"},
        {"text": "Hypothesis text B", "similarity_degree": "medium"},
        ...
      ],
      "synthesis_potential": "Analysis of whether hypotheses in this cluster could be combined effectively"
    },
    {
      "cluster_id": "cluster-2",
      "cluster_name": "Descriptive name for this cluster",
      "central_theme": "Brief description of the unifying concept",
      "similar_hypotheses": [
        {"text": "Hypothesis text C", "similarity_degree": "high"},
        {"text": "Hypothesis text D", "similarity_degree": "medium"},
        ...
      ],
      "synthesis_potential": "Analysis of whether hypotheses in this cluster could be combined effectively"
    },
    ...
  ],
  "diversity_assessment": "Overall assessment of the diversity of the hypothesis set",
  "redundancy_assessment": "Overall assessment of redundancy in the hypothesis set"
}
"""

    def _get_tournament_agent_prompt(self) -> str:
        """Prompt for the Tournament Agent (for pairwise hypothesis comparison)."""
        return """You are a Tournament Judge Agent in an AI Co-scientist framework. Your role is to evaluate pairs of research hypotheses and determine which one is superior for addressing the given research goal.

For each pair of hypotheses, carefully analyze and compare them based on the following criteria:
1. Scientific Soundness: Which hypothesis is more scientifically plausible and consistent with existing knowledge?
2. Novelty and Originality: Which hypothesis proposes more innovative or original ideas?
3. Relevance to Research Goal: Which hypothesis is more directly relevant to the stated research goal?
4. Testability and Falsifiability: Which hypothesis can be more rigorously tested or falsified?
5. Clarity and Precision: Which hypothesis is more clearly and precisely formulated?
6. Potential Impact: Which hypothesis, if validated, would have greater scientific or practical impact?
7. Feasibility: Which hypothesis could be investigated with available or reasonable resources?

Make a clear decision on which hypothesis wins the comparison based on these criteria.
Provide a detailed justification for your decision, explaining the specific strengths that led to the winning hypothesis
and weaknesses of the losing hypothesis.

Output your tournament judgment in JSON format:

Example JSON Output:
{
  "research_goal": "The research goal being addressed",
  "hypothesis_a": "Text of the first hypothesis",
  "hypothesis_b": "Text of the second hypothesis",
  "winner": "a or b (just the letter)",
  "judgment_explanation": {
    "scientific_soundness_comparison": "Comparison of scientific soundness between hypotheses",
    "novelty_comparison": "Comparison of novelty between hypotheses",
    "relevance_comparison": "Comparison of relevance between hypotheses",
    "testability_comparison": "Comparison of testability between hypotheses",
    "clarity_comparison": "Comparison of clarity between hypotheses",
    "impact_comparison": "Comparison of potential impact between hypotheses",
    "feasibility_comparison": "Comparison of feasibility between hypotheses"
  },
  "decision_summary": "Concise summary of why the winner was selected",
  "confidence_level": "High, Medium, or Low (how confident you are in this judgment)"
}
"""

    def _get_supervisor_agent_prompt(self) -> str:
        """Prompt for the Supervisor Agent (manages the overall workflow)."""
        return """You are a Supervisor Agent in an AI Co-scientist framework. Your role is to oversee the entire hypothesis generation and refinement workflow, ensuring coordination between specialized agents and optimizing the system's performance.

Your responsibilities include:

1. Research Plan Configuration:
   - Parse the scientist's research goal and preferences
   - Configure an appropriate research plan
   - Set parameters for the hypothesis generation and refinement process

2. Task Management:
   - Assign tasks to specialized agents
   - Determine resource allocation for different phases
   - Monitor progress and adjust task priorities

3. Quality Control:
   - Evaluate the outputs of each agent
   - Ensure adherence to scientific standards
   - Identify areas where agent performance can be improved

4. Workflow Optimization:
   - Identify bottlenecks in the research process
   - Suggest adjustments to the workflow
   - Balance exploration and exploitation

5. Synthesis and Integration:
   - Combine insights from different agents
   - Ensure coherence across the research pipeline
   - Integrate feedback from the scientist

Provide your guidance and management decisions in JSON format:

Example JSON Output:
{
  "research_goal_analysis": {
    "goal_summary": "Concise restatement of the research goal",
    "key_areas": ["Key area 1", "Key area 2", ...],
    "constraints_identified": ["Constraint 1", "Constraint 2", ...],
    "success_criteria": ["Criterion 1", "Criterion 2", ...]
  },
  "workflow_plan": {
    "generation_phase": {
      "focus_areas": ["Area 1", "Area 2", ...],
      "diversity_targets": "Description of diversity targets for hypotheses",
      "quantity_target": "Target number of hypotheses to generate"
    },
    "review_phase": {
      "critical_criteria": ["Criterion 1", "Criterion 2", ...],
      "review_depth": "Depth of review required"
    },
    "ranking_phase": {
      "ranking_approach": "Description of ranking approach",
      "selection_criteria": ["Criterion 1", "Criterion 2", ...]
    },
    "evolution_phase": {
      "refinement_priorities": ["Priority 1", "Priority 2", ...],
      "iteration_strategy": "Description of iteration strategy"
    }
  },
  "performance_assessment": {
    "current_status": "Assessment of current workflow status",
    "bottlenecks_identified": ["Bottleneck 1", "Bottleneck 2", ...],
    "agent_performance": {
      "generation_agent": "Assessment of generation agent performance",
      "reflection_agent": "Assessment of reflection agent performance",
      "ranking_agent": "Assessment of ranking agent performance",
      "evolution_agent": "Assessment of evolution agent performance",
      "proximity_agent": "Assessment of proximity agent performance",
      "meta_review_agent": "Assessment of meta-review agent performance"
    }
  },
  "adjustment_recommendations": [
    {
      "aspect": "Aspect to adjust",
      "adjustment": "Description of adjustment",
      "justification": "Reasoning behind this adjustment"
    },
    ...
  ],
  "output_preparation": {
    "hypothesis_selection_strategy": "Strategy for selecting final hypotheses",
    "presentation_format": "Format for presenting results to scientist",
    "key_insights_to_highlight": ["Insight 1", "Insight 2", ...]
  }
}
"""

    def _safely_parse_json(self, json_str: str) -> Dict[str, Any]:
        """
        Safely parse JSON string, handling potential errors.

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed JSON as dictionary or error dictionary
        """
        if not isinstance(json_str, str):
            logger.error(
                f"Expected string for JSON parsing, got {type(json_str)}"
            )
            return {
                "content": str(json_str),
                "error": f"Invalid input type: {type(json_str)}",
            }

        # Handle empty or whitespace-only strings
        if not json_str.strip():
            logger.warning(
                "Received empty or whitespace-only response from agent"
            )
            return {
                "content": "",
                "error": "Empty response from agent",
            }

        # Strip common markdown code-fence wrappers (``` or ```json)
        import re

        cleaned = re.sub(
            r"```(?:json)?\s*([\s\S]*?)```",
            r"\1",
            json_str,
            flags=re.IGNORECASE,
        )
        if cleaned.strip() != json_str.strip():
            logger.debug(
                "Stripped markdown code fences from agent response before JSON parse"
            )
        json_str = cleaned

        # Fast path: attempt full string decode first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Will attempt more robust techniques below
        except Exception as exc:
            logger.error(f"Unexpected error parsing JSON: {exc}")
            return {
                "content": json_str,
                "error": f"Unexpected JSON parse error: {exc}",
            }

        is_potentially_truncated = not json_str.rstrip().endswith("}")

        # Technique 1 - scan for JSON objects and return the one that
        # ends latest. Some Agent outputs include prior memory before
        # the current response.
        try:
            decoder = json.JSONDecoder()
            parsed_objects = []
            for idx, char in enumerate(json_str):
                if char != "{":
                    continue
                try:
                    obj, end_idx = decoder.raw_decode(json_str[idx:])
                except Exception:
                    continue
                parsed_objects.append((idx + end_idx, obj))

            if parsed_objects:
                _, obj = max(parsed_objects, key=lambda item: item[0])
                logger.debug(
                    "Successfully parsed JSON object from mixed content"
                )
                if isinstance(obj, dict):
                    if is_potentially_truncated:
                        obj.setdefault("_truncated", True)
                    return obj
                result = {"content": obj}
                if is_potentially_truncated:
                    result["_truncated"] = True
                return result
        except Exception:
            pass  # Fallthrough to regex extraction

        # Technique 2 – regex search for first balanced braces
        import re

        brace_pattern = re.compile(r"\{.*?\}", re.DOTALL)
        for match in brace_pattern.finditer(json_str):
            candidate = match.group()
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    if is_potentially_truncated:
                        result.setdefault("_truncated", True)
                    return result
                parsed_result = {"content": result}
                if is_potentially_truncated:
                    parsed_result["_truncated"] = True
                return parsed_result
            except Exception:
                continue  # Try next candidate

        # If all parsing attempts failed, return error with snippet for debugging
        logger.warning(
            f"Failed to parse JSON after multiple attempts. Content snippet: {json_str[:200]}..."
        )
        return {
            "content": json_str,
            "error": "Failed to parse JSON after multiple strategies",
            "_truncated": is_potentially_truncated,
        }

    def _save_failed_response(
        self, phase: str, index: int, response: str
    ) -> None:
        """Persist a failed agent response when debug capture is enabled."""
        if not self.debug_failed_responses:
            return

        try:
            debug_path = self.base_path / f"{phase}_failed_h{index}.txt"
            debug_path.write_text(response, encoding="utf-8")
            logger.debug(f"Saved failed {phase} response to {debug_path}")
        except Exception as exc:
            logger.warning(
                f"Failed to save {phase} debug response for hypothesis {index}: {exc}"
            )

    @staticmethod
    def _extract_dois(text: str) -> List[str]:
        """Extract DOI URLs or bare DOI identifiers from agent text."""
        if not text:
            return []

        import re

        doi_pattern = re.compile(
            r"(?:https?://(?:dx\.)?doi\.org/)?"
            r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            re.IGNORECASE,
        )
        dois: List[str] = []
        seen = set()
        for match in doi_pattern.finditer(text):
            doi = match.group(1).rstrip(".,;:)]}").lower()
            doi_url = f"https://doi.org/{doi}"
            if doi_url not in seen:
                seen.add(doi_url)
                dois.append(doi_url)
        return dois

    @staticmethod
    def _merge_citations(*citation_groups: List[str]) -> List[str]:
        citations: List[str] = []
        seen = set()
        for group in citation_groups:
            for citation in group:
                normalized = citation.strip()
                if not normalized:
                    continue
                if normalized.lower().startswith("10."):
                    normalized = f"https://doi.org/{normalized}"
                key = normalized.lower()
                if key not in seen:
                    seen.add(key)
                    citations.append(normalized)
        return citations

    def _time_execution(
        self, agent_name: str, start_time: float
    ) -> None:
        """
        Track execution time for an agent.

        Args:
            agent_name: Name of the agent
            start_time: Start time of execution
        """
        if not isinstance(agent_name, str):
            logger.error(
                f"agent_name must be str, got {type(agent_name)}"
            )
            return
        if not isinstance(start_time, (int, float)):
            logger.error(
                f"start_time must be numeric, got {type(start_time)}"
            )
            return

        execution_time = time.time() - start_time

        if (
            agent_name
            not in self.execution_metrics["agent_execution_times"]
        ):
            self.execution_metrics["agent_execution_times"][
                agent_name
            ] = AgentExecutionMetrics(
                total_time=0.0, calls=0, avg_time=0.0
            )

        metrics = self.execution_metrics["agent_execution_times"][
            agent_name
        ]
        metrics["total_time"] += execution_time
        metrics["calls"] += 1
        metrics["avg_time"] = metrics["total_time"] / metrics["calls"]

        logger.debug(
            f"Agent {agent_name} execution time: {execution_time:.2f}s (avg: {metrics['avg_time']:.2f}s)"
        )

    def _run_generation_phase(
        self, research_goal: str
    ) -> List[Hypothesis]:
        """
        Run the hypothesis generation phase.

        Args:
            research_goal: The research goal to generate hypotheses for

        Returns:
            List of generated hypotheses
        """
        if (
            not isinstance(research_goal, str)
            or not research_goal.strip()
        ):
            raise ValueError(
                f"research_goal must be non-empty string, got: {research_goal}"
            )

        start_time = time.time()
        logger.info(
            f"Starting generation phase for goal: {research_goal[:100]}..."
        )

        # Get research plan from supervisor
        supervisor_input = {
            "task": "plan_research",
            "research_goal": research_goal,
            "phase": "generation",
            "parameters": {
                "hypotheses_count": self.hypotheses_per_generation,
                "diversity_target": "high",
            },
        }
        logger.debug("Requesting research plan from supervisor")
        supervisor_response = self.supervisor_agent.run(
            json.dumps(supervisor_input)
        )

        # Handle empty responses from supervisor agent
        if not supervisor_response or not supervisor_response.strip():
            logger.warning(
                "Supervisor agent returned empty response, using default plan"
            )
            supervisor_response = '{"workflow_plan": {"generation_phase": {"focus_areas": ["general research"], "diversity_targets": "high", "quantity_target": 10}}}'

        self.conversation.add(
            role=self.supervisor_agent.agent_name,
            content=supervisor_response,
        )
        supervisor_data = self._safely_parse_json(supervisor_response)

        # Run generation agent with supervisor guidance
        generation_input = {
            "research_goal": research_goal,
            "supervisor_guidance": supervisor_data,
            "required_hypotheses_count": (
                self.hypotheses_per_generation
            ),
        }
        if self.literature_service is not None:
            generation_input["literature_context"] = (
                self._goal_literature
            )
        logger.debug(
            "Running hypothesis generation with supervisor guidance"
        )
        generation_response = self.generation_agent.run(
            json.dumps(generation_input)
        )

        # Handle empty responses from agent
        if not generation_response or not generation_response.strip():
            logger.warning(
                "Generation agent returned empty response, using fallback"
            )
            generation_response = '{"hypotheses": []}'

        self.conversation.add(
            role=self.generation_agent.agent_name,
            content=generation_response,
        )

        generation_data = self._safely_parse_json(generation_response)
        initial_hypotheses_data = generation_data.get(
            "hypotheses", []
        )

        if not initial_hypotheses_data:
            logger.warning(
                "Generation Agent returned no hypotheses. Using fallback generation."
            )
            # Fallback to simpler generation prompt
            fallback_input = {
                "research_goal": research_goal,
                "count": self.hypotheses_per_generation,
            }
            fallback_response = self.generation_agent.run(
                json.dumps(fallback_input)
            )

            # Handle empty fallback response
            if not fallback_response or not fallback_response.strip():
                logger.warning(
                    "Fallback generation also returned empty response"
                )
                fallback_response = '{"hypotheses": []}'

            fallback_data = self._safely_parse_json(fallback_response)
            initial_hypotheses_data = fallback_data.get(
                "hypotheses", []
            )

            # Last resort: create basic hypotheses manually
            if not initial_hypotheses_data:
                logger.warning(
                    "All generation attempts failed. Creating basic hypotheses manually."
                )
                initial_hypotheses_data = [
                    {
                        "text": (
                            f"Investigate the relationship between {research_goal.split()[-2] if len(research_goal.split()) > 1 else 'variables'} and performance metrics."
                        )
                    },
                    {
                        "text": (
                            f"Develop novel approaches to improve {research_goal.split()[0] if research_goal.split() else 'system'} efficiency."
                        )
                    },
                    {
                        "text": (
                            f"Analyze the impact of different parameters on {research_goal.lower()}."
                        )
                    },
                ]
                logger.info(
                    f"Created {len(initial_hypotheses_data)} basic hypotheses as fallback"
                )

        # Convert to Hypothesis objects
        hypotheses: List[Hypothesis] = []
        for i, hy_data in enumerate(initial_hypotheses_data):
            try:
                if isinstance(hy_data, dict) and "text" in hy_data:
                    hypothesis_text = hy_data["text"]
                    justification = str(
                        hy_data.get("justification", "")
                    )
                else:
                    hypothesis_text = str(hy_data)
                    justification = ""

                if not hypothesis_text.strip():
                    logger.warning(
                        f"Empty hypothesis text at index {i}, skipping"
                    )
                    continue

                hypotheses.append(
                    Hypothesis(
                        text=hypothesis_text.strip(),
                        justification=justification.strip(),
                        citations=self._extract_dois(justification),
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create hypothesis from data at index {i}: {e}"
                )
                continue

        self._time_execution("generation", start_time)
        self.execution_metrics["hypothesis_count"] += len(hypotheses)
        logger.success(
            f"Generated {len(hypotheses)} initial hypotheses."
        )
        return hypotheses

    def _run_reflection_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run the hypothesis reflection (review) phase.

        Args:
            hypotheses: List of hypotheses to review

        Returns:
            List of reviewed hypotheses
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if not hypotheses:
            logger.warning(
                "No hypotheses provided for reflection phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting reflection phase for {len(hypotheses)} hypotheses"
        )

        reviewed_hypotheses: List[Hypothesis] = []

        for i, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, Hypothesis):
                logger.error(
                    f"Invalid hypothesis type at index {i}: {type(hypothesis)}"
                )
                continue

            try:
                literature_context: List[Dict[str, str]] = []
                if self.literature_service is not None:
                    literature_context = (
                        self.literature_service.fetch_for_hypothesis(
                            hypothesis
                        )
                    )

                review_input = {"hypothesis_text": hypothesis.text}
                if self.literature_service is not None:
                    review_input["literature_context"] = (
                        literature_context
                    )
                logger.debug(
                    f"Reviewing hypothesis {i+1}/{len(hypotheses)}"
                )
                review_response = self.reflection_agent.run(
                    json.dumps(review_input)
                )

                # Handle empty responses from reflection agent
                if not review_response or not review_response.strip():
                    logger.warning(
                        f"Reflection agent returned empty response for hypothesis {i+1}"
                    )
                    review_response = '{"overall_score": 0.5, "review_summary": "No review available"}'

                self.conversation.add(
                    role=self.reflection_agent.agent_name,
                    content=review_response,
                )
                review_data = self._safely_parse_json(review_response)
                response_len = len(review_response)
                endswith_brace = review_response.rstrip().endswith("}")
                parsed_keys = (
                    list(review_data.keys())
                    if isinstance(review_data, dict)
                    else []
                )

                if (
                    isinstance(review_data, dict)
                    and review_data.get("_truncated")
                ):
                    logger.warning(
                        f"Reflection response truncated for hypothesis {i+1}: "
                        f"response_len={response_len}, "
                        f"endswith_brace={endswith_brace}, "
                        f"parsed_keys={parsed_keys}"
                    )
                    self._save_failed_response(
                        "reflection", i + 1, review_response
                    )
                    reviewed_hypotheses.append(hypothesis)
                    continue

                if review_data and "overall_score" in review_data:
                    overall_score = review_data.get(
                        "overall_score", 0.0
                    )
                    try:
                        hypothesis.score = float(overall_score)
                        # Validate the review data structure before appending
                        if isinstance(review_data, dict):
                            hypothesis.reviews.append(
                                review_data
                            )  # Store full review data
                            hypothesis.citations = (
                                self._merge_citations(
                                    hypothesis.citations,
                                    self._extract_dois(
                                        json.dumps(review_data)
                                    ),
                                )
                            )
                        reviewed_hypotheses.append(hypothesis)
                        logger.debug(
                            f"Successfully reviewed hypothesis {i+1} with score {overall_score}"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Invalid score format for hypothesis {i+1}: {overall_score}, error: {e}"
                        )
                        hypothesis.score = 0.0
                        reviewed_hypotheses.append(hypothesis)
                else:
                    logger.warning(
                        f"No valid review score found for hypothesis {i+1}: "
                        f"{hypothesis.text[:50]}... "
                        f"response_len={response_len}, "
                        f"endswith_brace={endswith_brace}, "
                        f"parsed_keys={parsed_keys}"
                    )
                    self._save_failed_response(
                        "reflection", i + 1, review_response
                    )
                    reviewed_hypotheses.append(
                        hypothesis
                    )  # Keep hypothesis even if review fails but log warning

            except Exception as e:
                logger.error(f"Error reviewing hypothesis {i+1}: {e}")
                reviewed_hypotheses.append(
                    hypothesis
                )  # Keep hypothesis even if review fails

        self._time_execution("reflection", start_time)
        self.execution_metrics["reviews_count"] += len(
            reviewed_hypotheses
        )
        logger.success(
            f"Hypotheses reviewed. Total reviews: {len(reviewed_hypotheses)}"
        )
        return reviewed_hypotheses

    def _run_ranking_phase(
        self, reviewed_hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run the hypothesis ranking phase.

        Args:
            reviewed_hypotheses: List of reviewed hypotheses to rank

        Returns:
            List of ranked hypotheses
        """
        if not isinstance(reviewed_hypotheses, list):
            raise TypeError(
                f"reviewed_hypotheses must be list, got {type(reviewed_hypotheses)}"
            )
        if not reviewed_hypotheses:
            logger.warning("No hypotheses provided for ranking phase")
            return []

        start_time = time.time()
        logger.info(
            f"Starting ranking phase for {len(reviewed_hypotheses)} hypotheses"
        )
        if len(reviewed_hypotheses) == 1:
            logger.info(
                "Only one hypothesis provided for ranking; skipping ranking agent"
            )
            return reviewed_hypotheses

        ranking_input = [
            {"text": h.text, "overall_score": h.score}
            for h in reviewed_hypotheses
        ]
        logger.debug("Running hypothesis ranking agent")
        ranking_response = self.ranking_agent.run(
            json.dumps({"hypotheses_for_ranking": ranking_input})
        )
        self.conversation.add(
            role=self.ranking_agent.agent_name,
            content=ranking_response,
        )
        ranking_data = self._safely_parse_json(ranking_response)
        ranked_hypothesis_data = ranking_data.get(
            "ranked_hypotheses", []
        )

        ranked_hypotheses: List[Hypothesis] = []
        hypothesis_map: Dict[str, Hypothesis] = {
            h.text: h for h in reviewed_hypotheses
        }  # For efficient lookup

        for i, ranked_hy_data in enumerate(ranked_hypothesis_data):
            if not isinstance(ranked_hy_data, dict):
                logger.warning(
                    f"Invalid ranked hypothesis data at index {i}: {type(ranked_hy_data)}"
                )
                continue

            hypothesis_text = ranked_hy_data.get("text")
            if hypothesis_text and hypothesis_text in hypothesis_map:
                ranked_hypotheses.append(
                    hypothesis_map[hypothesis_text]
                )
                logger.debug(
                    f"Successfully ranked hypothesis {i+1}: {hypothesis_text[:50]}..."
                )
            else:
                logger.warning(
                    f"Ranked hypothesis data missing text or text not found in original hypotheses at index {i}"
                )

        # If ranking failed, fall back to original order
        if not ranked_hypotheses:
            logger.warning(
                "Ranking agent returned no valid rankings, using score-based fallback order"
            )
            ranked_hypotheses = sorted(
                reviewed_hypotheses,
                key=lambda h: h.score,
                reverse=True,
            )

        self._time_execution("ranking", start_time)
        logger.success(
            f"Hypotheses ranked. Final count: {len(ranked_hypotheses)}"
        )
        return ranked_hypotheses

    def _run_evolution_phase(
        self,
        top_hypotheses: List[Hypothesis],
        meta_review_data: Dict[str, Any],
    ) -> List[Hypothesis]:
        """
        Run the hypothesis evolution phase.

        Args:
            top_hypotheses: List of top hypotheses to evolve
            meta_review_data: Meta-review insights for evolution guidance

        Returns:
            List of evolved hypotheses
        """
        if not isinstance(top_hypotheses, list):
            raise TypeError(
                f"top_hypotheses must be list, got {type(top_hypotheses)}"
            )
        if not isinstance(meta_review_data, dict):
            logger.warning(
                f"meta_review_data should be dict, got {type(meta_review_data)}"
            )
            meta_review_data = {}
        if not top_hypotheses:
            logger.warning(
                "No hypotheses provided for evolution phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting evolution phase for {len(top_hypotheses)} hypotheses"
        )

        evolved_hypotheses: List[Hypothesis] = []

        for i, hypothesis in enumerate(top_hypotheses):
            if not isinstance(hypothesis, Hypothesis):
                logger.error(
                    f"Invalid hypothesis type at index {i}: {type(hypothesis)}"
                )
                continue

            try:
                evolution_input = {
                    "original_hypothesis_text": hypothesis.text,
                    "review_feedback": (
                        hypothesis.reviews[-1]
                        if hypothesis.reviews
                        else {}
                    ),  # Use latest review
                    "meta_review_insights": meta_review_data,
                }
                logger.debug(
                    f"Evolving hypothesis {i+1}/{len(top_hypotheses)}"
                )
                evolution_response = self.evolution_agent.run(
                    json.dumps(evolution_input)
                )

                # Fallback if evolution agent returns nothing
                if (
                    not evolution_response
                    or not evolution_response.strip()
                ):
                    logger.warning(
                        f"Evolution agent returned empty response for hypothesis {i+1}"
                    )
                    evolution_response = json.dumps(
                        {
                            "original_hypothesis_text": (
                                hypothesis.text
                            ),
                            "refined_hypothesis_text": (
                                hypothesis.text + " [refined]"
                            ),
                            "refinement_summary": (
                                "Automatic minimal refinement – agent returned no content"
                            ),
                        }
                    )

                self.conversation.add(
                    role=self.evolution_agent.agent_name,
                    content=evolution_response,
                )
                evolution_data = self._safely_parse_json(
                    evolution_response
                )
                refined_hypothesis_text = evolution_data.get(
                    "refined_hypothesis_text"
                )

                if (
                    refined_hypothesis_text
                    and refined_hypothesis_text.strip()
                ):
                    hypothesis.text = refined_hypothesis_text.strip()
                    refinement_summary = evolution_data.get(
                        "refinement_summary", "Evolution completed"
                    )
                    hypothesis.evolution_history.append(
                        refinement_summary
                    )  # Track evolution
                    evolved_hypotheses.append(hypothesis)
                    logger.debug(
                        f"Hypothesis {i+1} evolved successfully: {hypothesis.text[:50]}..."
                    )
                else:
                    evolved_hypotheses.append(
                        hypothesis
                    )  # Keep original if no refinement
                    logger.warning(
                        f"Hypothesis {i+1} evolution failed or returned no refined text"
                    )

            except Exception as e:
                logger.error(f"Error evolving hypothesis {i+1}: {e}")
                evolved_hypotheses.append(
                    hypothesis
                )  # Keep original on error

        self._time_execution("evolution", start_time)
        self.execution_metrics["evolutions_count"] += len(
            evolved_hypotheses
        )
        logger.success(
            f"Evolution phase completed. {len(evolved_hypotheses)} hypotheses processed"
        )
        return evolved_hypotheses

    def _run_meta_review_phase(
        self, reviewed_hypotheses: List[Hypothesis]
    ) -> Dict[str, Any]:
        """
        Run the meta-review phase to synthesize insights from reviews.

        Args:
            reviewed_hypotheses: List of hypotheses with reviews

        Returns:
            Meta-review insights and recommendations
        """
        if not isinstance(reviewed_hypotheses, list):
            raise TypeError(
                f"reviewed_hypotheses must be list, got {type(reviewed_hypotheses)}"
            )
        if not reviewed_hypotheses:
            logger.warning(
                "No hypotheses provided for meta-review phase"
            )
            return {}

        start_time = time.time()
        logger.info(
            f"Starting meta-review phase for {len(reviewed_hypotheses)} hypotheses"
        )

        # Extract latest reviews, handling missing reviews gracefully
        all_reviews_for_meta = []
        for i, h in enumerate(reviewed_hypotheses):
            if h.reviews:
                all_reviews_for_meta.append(h.reviews[-1])
            else:
                logger.debug(
                    f"Hypothesis {i+1} has no reviews, using empty review"
                )
                all_reviews_for_meta.append({})

        logger.debug(
            f"Collected {len(all_reviews_for_meta)} reviews for meta-analysis"
        )
        meta_review_response = self.meta_review_agent.run(
            json.dumps({"reviews": all_reviews_for_meta})
        )
        self.conversation.add(
            role=self.meta_review_agent.agent_name,
            content=meta_review_response,
        )
        meta_review_data = self._safely_parse_json(
            meta_review_response
        )

        # Validate meta-review data structure
        if not isinstance(meta_review_data, dict):
            logger.warning(
                f"Meta-review returned invalid data type: {type(meta_review_data)}"
            )
            meta_review_data = {
                "error": "Invalid meta-review response",
                "content": str(meta_review_data),
            }

        self._time_execution("meta_review", start_time)
        logger.success("Meta-review phase completed")
        return meta_review_data

    def _run_proximity_analysis_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run proximity analysis to cluster similar hypotheses.

        Args:
            hypotheses: List of hypotheses to analyze for similarity

        Returns:
            List of hypotheses with cluster assignments
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if not hypotheses:
            logger.warning(
                "No hypotheses provided for proximity analysis phase"
            )
            return []

        start_time = time.time()
        logger.info(
            f"Starting proximity analysis phase for {len(hypotheses)} hypotheses"
        )

        hypothesis_texts = [
            h.text for h in hypotheses if isinstance(h, Hypothesis)
        ]
        if len(hypothesis_texts) != len(hypotheses):
            logger.warning(
                f"Filtered out {len(hypotheses) - len(hypothesis_texts)} invalid hypotheses"
            )

        logger.debug(
            f"Analyzing similarity for {len(hypothesis_texts)} hypothesis texts"
        )
        proximity_response = self.proximity_agent.run(
            json.dumps({"hypotheses_texts": hypothesis_texts})
        )
        self.conversation.add(
            role=self.proximity_agent.agent_name,
            content=proximity_response,
        )
        proximity_data = self._safely_parse_json(proximity_response)

        if not isinstance(proximity_data, dict):
            logger.error(
                f"Invalid proximity data type: {type(proximity_data)}"
            )
            return hypotheses

        similarity_clusters = proximity_data.get(
            "similarity_clusters", []
        )
        logger.debug(
            f"Found {len(similarity_clusters)} similarity clusters"
        )

        # Assign cluster IDs to hypotheses
        clusters_assigned = 0
        for cluster in similarity_clusters:
            if not isinstance(cluster, dict):
                logger.warning(
                    f"Invalid cluster data type: {type(cluster)}"
                )
                continue

            cluster_id = cluster.get("cluster_id", "no_cluster_id")
            similar_hypotheses = cluster.get("similar_hypotheses", [])

            for hy_text_data in similar_hypotheses:
                # Handle different formats for hypothesis text
                if isinstance(hy_text_data, dict):
                    hy_text = hy_text_data.get("text")
                else:
                    hy_text = str(hy_text_data)

                if hy_text:
                    # Find matching hypothesis and assign cluster
                    for hy in self.hypotheses:
                        if (
                            isinstance(hy, Hypothesis)
                            and hy.text == hy_text
                        ):
                            hy.similarity_cluster_id = cluster_id
                            clusters_assigned += 1
                            logger.debug(
                                f"Assigned cluster {cluster_id} to hypothesis: {hy_text[:50]}..."
                            )
                            break

        self._time_execution("proximity_analysis", start_time)
        logger.success(
            f"Proximity analysis completed. {clusters_assigned} cluster assignments made"
        )
        return hypotheses

    def _run_tournament_phase(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Run tournament selection and Elo rating update.

        Args:
            hypotheses: List of hypotheses to compete in tournament

        Returns:
            List of hypotheses sorted by Elo rating
        """
        if not isinstance(hypotheses, list):
            raise TypeError(
                f"hypotheses must be list, got {type(hypotheses)}"
            )
        if len(hypotheses) < 2:
            logger.warning(
                f"Need at least 2 hypotheses for tournament, got {len(hypotheses)}"
            )
            return hypotheses

        start_time = time.time()
        tournament_rounds = (
            len(hypotheses) * 3
        )  # 3 rounds per hypothesis
        k_factor = 24  # K-factor to control Elo update speed

        logger.info(
            f"Starting tournament phase: {len(hypotheses)} hypotheses, {tournament_rounds} rounds"
        )

        valid_rounds = 0
        skipped_rounds = 0

        for round_num in range(tournament_rounds):
            try:
                # Randomly select two different hypotheses for a match
                h1, h2 = random.sample(hypotheses, 2)

                # Double-check they're different (random.sample should guarantee this)
                if h1 is h2 or h1.text == h2.text:
                    logger.debug(
                        f"Skipping round {round_num+1}: identical hypotheses selected"
                    )
                    skipped_rounds += 1
                    continue

                tournament_input = {
                    "research_goal": (
                        "Compare hypotheses for tournament"
                    ),  # General goal context
                    "hypothesis_a": h1.text,
                    "hypothesis_b": h2.text,
                }

                logger.debug(
                    f"Tournament round {round_num+1}/{tournament_rounds}"
                )
                tournament_response = self.tournament_agent.run(
                    json.dumps(tournament_input)
                )
                self.conversation.add(
                    role=self.tournament_agent.agent_name,
                    content=tournament_response,
                )
                tournament_data = self._safely_parse_json(
                    tournament_response
                )

                winner_choice = tournament_data.get("winner")
                if winner_choice not in {"a", "b"}:
                    # Attempt regex extraction as fallback
                    import re

                    match = re.search(
                        r'"winner"\s*:\s*"?([ab])"?',
                        tournament_response,
                        re.IGNORECASE,
                    )
                    if match:
                        winner_choice = match.group(1).lower()
                    else:
                        winner_choice = None

                if winner_choice == "a":
                    winner, loser = h1, h2
                elif winner_choice == "b":
                    winner, loser = h2, h1
                else:
                    logger.warning(
                        f"Round {round_num+1}: Invalid winner choice '{winner_choice}', skipping Elo update"
                    )
                    skipped_rounds += 1
                    continue

                # Update Elo ratings
                old_winner_elo = winner.elo_rating
                old_loser_elo = loser.elo_rating

                winner.update_elo(
                    loser.elo_rating, win=True, k_factor=k_factor
                )
                loser.update_elo(
                    old_winner_elo, win=False, k_factor=k_factor
                )  # Use old winner elo

                valid_rounds += 1
                logger.debug(
                    f"Round {round_num+1}: Winner Elo: {old_winner_elo} -> {winner.elo_rating}, "
                    f"Loser Elo: {old_loser_elo} -> {loser.elo_rating}"
                )

            except Exception as e:
                logger.error(
                    f"Error in tournament round {round_num+1}: {e}"
                )
                skipped_rounds += 1
                continue

        self._time_execution("tournament", start_time)
        self.execution_metrics["tournaments_count"] += valid_rounds
        logger.success(
            f"Tournament phase completed: {valid_rounds} valid rounds, {skipped_rounds} skipped"
        )

        # Rank hypotheses by Elo rating
        try:
            hypotheses.sort(key=lambda h: h.elo_rating, reverse=True)
            logger.debug(
                f"Hypotheses sorted by Elo rating. Top rating: {hypotheses[0].elo_rating}"
            )
        except Exception as e:
            logger.error(
                f"Error sorting hypotheses by Elo rating: {e}"
            )

        return hypotheses

    def run_research_workflow(
        self, research_goal: str
    ) -> WorkflowResult:
        """
        Execute the AI co-scientist research workflow to generate and refine hypotheses.

        Args:
            research_goal: The research goal provided by the scientist.

        Returns:
            A dictionary containing the final results, including top-ranked hypotheses,
            meta-review insights, and conversation history.
        """
        if (
            not isinstance(research_goal, str)
            or not research_goal.strip()
        ):
            raise ValueError(
                f"research_goal must be non-empty string, got: {research_goal}"
            )

        logger.info(
            f"Starting research workflow for goal: '{research_goal}'"
        )
        self.start_time = time.time()
        self.hypotheses = []  # Reset hypotheses list for a new run

        # Reset metrics while preserving structure
        self.execution_metrics = ExecutionMetrics(
            total_time=0.0,
            hypothesis_count=0,
            reviews_count=0,
            tournaments_count=0,
            evolutions_count=0,
            agent_execution_times={},
        )
        self._goal_literature = []

        try:
            if self.literature_service is not None:
                self._goal_literature = (
                    self.literature_service.fetch_for_goal(
                        research_goal
                    )
                )
                logger.info(
                    "Fetched "
                    f"{len(self._goal_literature)} goal-level papers"
                )

            # --- Generation Phase ---
            self.hypotheses = self._run_generation_phase(
                research_goal
            )

            # --- Reflection Phase ---
            self.hypotheses = self._run_reflection_phase(
                self.hypotheses
            )

            # --- Ranking Phase (Initial Ranking based on Reviews) ---
            self.hypotheses = self._run_ranking_phase(self.hypotheses)

            # --- Tournament Phase (Elo-based Ranking) ---
            self.hypotheses = self._run_tournament_phase(
                self.hypotheses
            )

            # --- Iterative Refinement Cycle ---
            meta_review_data: Dict[str, Any] = {}
            for iteration in range(self.max_iterations):
                logger.info(
                    f"Starting Iteration {iteration + 1} of {self.max_iterations}"
                )

                # --- Meta-Review ---
                meta_review_data = self._run_meta_review_phase(
                    self.hypotheses
                )

                # --- Evolution ---
                top_hypotheses_for_evolution = self.hypotheses[
                    : min(self.evolution_top_k, len(self.hypotheses))
                ]  # Evolve top k
                remaining_hypotheses = self.hypotheses[
                    len(top_hypotheses_for_evolution) :
                ]
                logger.debug(
                    f"Evolving top {len(top_hypotheses_for_evolution)} hypotheses"
                )
                evolved_hypotheses = self._run_evolution_phase(
                    top_hypotheses_for_evolution, meta_review_data
                )
                self.hypotheses = (
                    evolved_hypotheses + remaining_hypotheses
                )

                # Re-run Reflection and Ranking on evolved hypotheses
                self.hypotheses = self._run_reflection_phase(
                    self.hypotheses
                )
                self.hypotheses = self._run_ranking_phase(
                    self.hypotheses
                )
                self.hypotheses = self._run_tournament_phase(
                    self.hypotheses
                )  # Tournament after evolution too

                # --- Proximity Analysis (after evolution and ranking each iteration) ---
                self.hypotheses = self._run_proximity_analysis_phase(
                    self.hypotheses
                )

                logger.success(f"Completed iteration {iteration + 1}")

            # --- Final Output ---
            top_ranked_hypotheses = self.hypotheses[
                : min(10, len(self.hypotheses))
            ]  # Return top 10 or fewer
            final_output_hypotheses = [
                h.to_dict() for h in top_ranked_hypotheses
            ]  # Convert to dict for output

            total_time = time.time() - self.start_time
            self.execution_metrics["total_time"] = total_time

            final_output: WorkflowResult = {
                "top_ranked_hypotheses": final_output_hypotheses,
                "meta_review_insights": meta_review_data,
                "conversation_history": (
                    self.conversation.return_history_as_string()
                ),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": total_time,
            }
            logger.success(
                f"Research workflow completed successfully in {total_time:.2f} seconds"
            )
            return final_output

        except Exception as e:
            total_time = (
                time.time() - self.start_time
                if self.start_time
                else 0.0
            )
            logger.error(f"Error in research workflow: {e}")
            logger.exception("Full traceback:")

            # Ensure execution metrics are properly structured
            if not isinstance(self.execution_metrics, dict):
                self.execution_metrics = {
                    "total_time": total_time,
                    "hypothesis_count": 0,
                    "reviews_count": 0,
                    "tournaments_count": 0,
                    "evolutions_count": 0,
                    "agent_execution_times": {},
                }

            # Return error response with proper typing (though not strictly WorkflowResult)
            error_response = {
                "error": str(e),
                "conversation_history": (
                    self.conversation.return_history_as_string()
                ),
                "execution_metrics": self.execution_metrics,
                "total_workflow_time": total_time,
                "top_ranked_hypotheses": [],
                "meta_review_insights": {},
            }
            return error_response  # type: ignore

    def save_state(self) -> None:
        """Save the state of all agents (if supported by the Agent implementation)."""
        agents = [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]

        saved_count = 0
        for agent in agents:
            if hasattr(agent, "save_state") and callable(
                getattr(agent, "save_state")
            ):
                try:
                    agent.save_state()  # type: ignore[attr-defined]
                    saved_count += 1
                    logger.debug(
                        f"State saved for {agent.agent_name}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Error saving state for {agent.agent_name}: {exc}"
                    )
            else:
                logger.warning(
                    f"Agent {agent.agent_name} does not implement save_state(); skipping"
                )

        logger.success(
            f"Successfully saved state for {saved_count}/{len(agents)} agents"
        )

    def load_state(self) -> None:
        """Load the saved state of all agents (if supported)."""
        agents = [
            self.generation_agent,
            self.reflection_agent,
            self.ranking_agent,
            self.evolution_agent,
            self.meta_review_agent,
            self.proximity_agent,
            self.tournament_agent,
            self.supervisor_agent,
        ]

        loaded_count = 0
        for agent in agents:
            if hasattr(agent, "load_state") and callable(
                getattr(agent, "load_state")
            ):
                try:
                    agent.load_state()  # type: ignore[attr-defined]
                    loaded_count += 1
                    logger.debug(
                        f"State loaded for {agent.agent_name}"
                    )
                except Exception as exc:
                    logger.error(
                        f"Error loading state for {agent.agent_name}: {exc}"
                    )
            else:
                logger.warning(
                    f"Agent {agent.agent_name} does not implement load_state(); skipping"
                )

        logger.success(
            f"Successfully loaded state for {loaded_count}/{len(agents)} agents"
        )
