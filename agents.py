from dataclasses import dataclass
from typing import Dict

from models import ExperienceEntry, Interpretation


RUN_1 = "run_1_divergent_memory"
RUN_2 = "run_2_shared_memory"


@dataclass(frozen=True)
class Agent:
    agent_id: str

    def interpret(self, entry: ExperienceEntry, context: Dict[str, str]) -> Interpretation:
        raise NotImplementedError


@dataclass(frozen=True)
class NPCAgentSymbolic(Agent):
    agent_id: str = "NPC_A"

    def interpret(self, entry: ExperienceEntry, context: Dict[str, str]) -> Interpretation:
        if context["scenario_id"] == RUN_1:
            return Interpretation(
                agent_id=self.agent_id,
                mode="symbolic",
                meaning_label="symbolic_risk_inference",
                feature_vector=(0.95, 0.10, 0.15, 0.85, 0.60),
                memory_reference="ASSOCIATIVE_EVENT_12 -> risk_warning",
                notes=(
                    f"{entry.object_id} is interpreted through associative symbolic context"
                    " as a risk signal."
                ),
            )

        return Interpretation(
            agent_id=self.agent_id,
            mode="symbolic",
            meaning_label="symbolic_caution_inference",
            feature_vector=(0.70, 0.45, 0.20, 0.55, 0.40),
            memory_reference="ALIGNED_CONTEXT_PROFILE_1 -> caution",
            notes=(
                f"{entry.object_id} remains symbolically interpreted, with inference"
                " shifted by aligned context."
            ),
        )


@dataclass(frozen=True)
class NPCAgentLiteral(Agent):
    agent_id: str = "NPC_B"

    def interpret(self, entry: ExperienceEntry, context: Dict[str, str]) -> Interpretation:
        if context["scenario_id"] == RUN_1:
            return Interpretation(
                agent_id=self.agent_id,
                mode="literal",
                meaning_label="literal_signal_inference",
                feature_vector=(0.10, 0.95, 0.05, 0.20, 0.25),
                memory_reference=None,
                notes=(
                    f"{entry.object_id} is interpreted from direct signal structure with"
                    " minimal contextual shaping."
                ),
            )

        return Interpretation(
            agent_id=self.agent_id,
            mode="literal",
            meaning_label="literal_caution_inference",
            feature_vector=(0.45, 0.70, 0.15, 0.50, 0.35),
            memory_reference="ALIGNED_CONTEXT_PROFILE_1 -> caution",
            notes=(
                f"{entry.object_id} remains literal, but aligned context reduces semantic"
                " distance from NPC_A."
            ),
        )


@dataclass(frozen=True)
class NPCAgentMetaObserver(Agent):
    agent_id: str = "NPC_C"

    def interpret(self, entry: ExperienceEntry, context: Dict[str, str]) -> Interpretation:
        if context["scenario_id"] == RUN_1:
            return Interpretation(
                agent_id=self.agent_id,
                mode="meta_observer",
                meaning_label="meta_interpretive_divergence",
                feature_vector=(0.35, 0.35, 0.95, 0.60, 0.80),
                memory_reference="MODEL_A_B_DISAGREEMENT",
                notes=(
                    f"{entry.object_id} is modeled second-order as interpretive divergence"
                    " between NPC_A and NPC_B."
                ),
            )

        return Interpretation(
            agent_id=self.agent_id,
            mode="meta_observer",
            meaning_label="meta_interpretive_convergence",
            feature_vector=(0.35, 0.35, 0.95, 0.45, 0.30),
            memory_reference="MODEL_A_B_ALIGNMENT",
            notes=(
                f"{entry.object_id} is modeled second-order as partial convergence"
                " between NPC_A and NPC_B."
            ),
        )


def build_agents() -> Dict[str, Agent]:
    return {
        "NPC_A": NPCAgentSymbolic(),
        "NPC_B": NPCAgentLiteral(),
        "NPC_C": NPCAgentMetaObserver(),
    }
