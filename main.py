from typing import Dict, List

from agents import RUN_1, RUN_2, build_agents
from divergence import mean_pairwise_divergence, pairwise_divergence_matrix
from models import ExperienceEntry, Interpretation, TVS


FEATURE_SPACE_LABEL = "[symbolic, literal, social_modeling, threat, ambiguity]"


def build_shared_experience_entry() -> ExperienceEntry:
    return ExperienceEntry(
        object_id="WORLD_OBJECT_A",
        tvs=TVS(axis_3=0.82, axis_7=-0.12, axis_14=0.43),
    )


def run_scenario(
    scenario_id: str,
    scenario_label: str,
    causal_frame: str,
    entry: ExperienceEntry,
) -> float:
    agents = build_agents()
    context: Dict[str, str] = {"scenario_id": scenario_id}
    interpretations: List[Interpretation] = [
        agents["NPC_A"].interpret(entry, context),
        agents["NPC_B"].interpret(entry, context),
        agents["NPC_C"].interpret(entry, context),
    ]

    print(f"===== {scenario_label} =====")
    print(f"Causal framing: {causal_frame}")

    print("\n--- Shared Representation ---")
    print("Input identity: same ExperienceEntry and TVS for all agents and both runs")
    print(f"ExperienceEntry.object_id: {entry.object_id}")
    print(
        "TVS: "
        f"axis_3={entry.tvs.axis_3}, "
        f"axis_7={entry.tvs.axis_7}, "
        f"axis_14={entry.tvs.axis_14}"
    )

    print("\n--- Agent Interpretations ---")
    for interpretation in interpretations:
        print(f"[{interpretation.agent_id}] mode={interpretation.mode}")
        print(f"  meaning_label: {interpretation.meaning_label}")
        print(f"  feature_space: {FEATURE_SPACE_LABEL}")
        print(f"  feature_vector: {interpretation.feature_vector}")
        print(f"  memory_reference: {interpretation.memory_reference}")
        print(f"  notes: {interpretation.notes}")

    matrix = pairwise_divergence_matrix(interpretations)
    print("\n--- Divergence Matrix (Euclidean Distance) ---")
    ids = ["NPC_A", "NPC_B", "NPC_C"]
    print("        " + "  ".join(f"{agent_id:>8}" for agent_id in ids))
    for left in ids:
        row = "  ".join(f"{matrix[left][right]:8.3f}" for right in ids)
        print(f"{left:>8}  {row}")

    avg = mean_pairwise_divergence(matrix)
    print(f"\nAverage pairwise interpretation divergence: {avg:.3f}")
    print()
    return avg


def main() -> None:
    entry = build_shared_experience_entry()

    run_1_avg = run_scenario(
        scenario_id=RUN_1,
        scenario_label="Run 1: Divergent Memory Context",
        causal_frame="identical representation, divergent memory context",
        entry=entry,
    )

    run_2_avg = run_scenario(
        scenario_id=RUN_2,
        scenario_label="Run 2: Shared Memory Context",
        causal_frame="identical representation, aligned memory context",
        entry=entry,
    )

    print("===== Comparison Summary =====")
    print(f"Run 1 average divergence: {run_1_avg:.3f}")
    print(f"Run 2 average divergence: {run_2_avg:.3f}")
    if run_1_avg > run_2_avg:
        print("Result: Run 1 divergence > Run 2 divergence (as expected).")
    else:
        print("Result: Divergence ordering did not match expectation.")
    print("Conclusion: perceptual agreement does not guarantee semantic agreement.")


if __name__ == "__main__":
    main()
