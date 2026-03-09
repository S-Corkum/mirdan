"""Multi-agent quality coordination.

Coordinates quality validation work across multiple AI agents
in a multi-agent coding session. Uses asyncio.Lock for safe
concurrent access (mirdan is an async MCP server).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    agent_id: str
    capabilities: list[str] = field(default_factory=list)
    claimed_files: set[str] = field(default_factory=set)
    completed_files: set[str] = field(default_factory=set)


@dataclass
class CoordinationResult:
    """Result of a coordination action."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            **self.data,
        }


class QualityCoordinator:
    """Coordinates quality validation across multiple AI agents.

    Prevents duplicate work by tracking which agents are validating
    which files. Uses asyncio.Lock for thread-safe async operations.

    Usage:
        coordinator = QualityCoordinator()
        coordinator.register_agent("agent-1", ["python", "security"])
        if await coordinator.claim_file("agent-1", "src/app.py"):
            # validate the file...
            await coordinator.release_file("agent-1", "src/app.py", result)
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._agents: dict[str, AgentInfo] = {}
        self._file_claims: dict[str, str] = {}  # file_path → agent_id
        self._results: dict[str, dict[str, Any]] = {}  # file_path → validation result

    def register_agent(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
    ) -> CoordinationResult:
        """Register an agent for quality coordination.

        Args:
            agent_id: Unique identifier for the agent.
            capabilities: List of capability strings (e.g., ["python", "security"]).

        Returns:
            CoordinationResult indicating success.
        """
        if agent_id in self._agents:
            return CoordinationResult(
                success=True,
                message=f"Agent {agent_id} already registered",
                data={"agent_count": len(self._agents)},
            )

        self._agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            capabilities=capabilities or [],
        )
        return CoordinationResult(
            success=True,
            message=f"Agent {agent_id} registered",
            data={"agent_count": len(self._agents)},
        )

    def unregister_agent(self, agent_id: str) -> CoordinationResult:
        """Unregister an agent and release all its file claims.

        Args:
            agent_id: The agent to unregister.

        Returns:
            CoordinationResult with released file count.
        """
        agent = self._agents.pop(agent_id, None)
        if not agent:
            return CoordinationResult(
                success=False,
                message=f"Agent {agent_id} not found",
            )

        # Release all file claims
        released = []
        for fpath, claimer in list(self._file_claims.items()):
            if claimer == agent_id:
                del self._file_claims[fpath]
                released.append(fpath)

        return CoordinationResult(
            success=True,
            message=f"Agent {agent_id} unregistered, {len(released)} files released",
            data={"released_files": released},
        )

    async def claim_file(
        self,
        agent_id: str,
        file_path: str,
    ) -> bool:
        """Attempt to claim a file for validation.

        Args:
            agent_id: The agent attempting to claim.
            file_path: The file to claim.

        Returns:
            True if the claim succeeded, False if already claimed.
        """
        async with self._lock:
            if file_path in self._file_claims:
                return False

            if agent_id not in self._agents:
                return False

            self._file_claims[file_path] = agent_id
            self._agents[agent_id].claimed_files.add(file_path)
            return True

    async def release_file(
        self,
        agent_id: str,
        file_path: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Release a file claim after validation.

        Args:
            agent_id: The agent releasing the claim.
            file_path: The file being released.
            result: Optional validation result to store.
        """
        async with self._lock:
            if self._file_claims.get(file_path) == agent_id:
                del self._file_claims[file_path]

                agent = self._agents.get(agent_id)
                if agent:
                    agent.claimed_files.discard(file_path)
                    agent.completed_files.add(file_path)

                if result is not None:
                    self._results[file_path] = result

    def get_unassigned_files(
        self,
        changed_files: list[str],
    ) -> list[str]:
        """Get files not yet claimed or completed by any agent.

        Args:
            changed_files: List of file paths that need validation.

        Returns:
            Files that haven't been claimed or had results submitted.
        """
        return [f for f in changed_files if f not in self._file_claims and f not in self._results]

    def aggregate_results(self) -> dict[str, Any]:
        """Aggregate all completed validation results.

        Returns:
            Summary dict with per-file results and aggregate stats.
        """
        if not self._results:
            return {
                "files_validated": 0,
                "avg_score": 0.0,
                "results": {},
            }

        scores = [r.get("score", 0.0) for r in self._results.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        total_violations = sum(len(r.get("violations", [])) for r in self._results.values())

        return {
            "files_validated": len(self._results),
            "avg_score": round(avg_score, 3),
            "total_violations": total_violations,
            "results": dict(self._results),
            "agents": {
                aid: {
                    "completed": len(a.completed_files),
                    "claimed": len(a.claimed_files),
                }
                for aid, a in self._agents.items()
            },
        }

    def get_status(self) -> dict[str, Any]:
        """Get current coordination status.

        Returns:
            Dict with agents, claims, and results summary.
        """
        return {
            "agent_count": len(self._agents),
            "agents": [
                {
                    "id": a.agent_id,
                    "capabilities": a.capabilities,
                    "claimed_files": sorted(a.claimed_files),
                    "completed_files": sorted(a.completed_files),
                }
                for a in self._agents.values()
            ],
            "active_claims": dict(self._file_claims),
            "completed_files": len(self._results),
        }
