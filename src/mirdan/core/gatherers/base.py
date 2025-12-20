"""Base protocol and types for context gatherers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mirdan.models import ContextBundle, Intent

if TYPE_CHECKING:
    from mirdan.core.client_registry import MCPClientRegistry

logger = logging.getLogger(__name__)


@dataclass
class GathererResult:
    """Result from a context gatherer operation.

    Attributes:
        success: Whether the gathering succeeded
        context: Partial ContextBundle with gathered data
        error: Error message if gathering failed
        metadata: Additional metadata about the gathering
    """

    success: bool
    context: ContextBundle
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ContextGatherer(Protocol):
    """Protocol for context gathering implementations.

    Each gatherer is responsible for populating specific fields
    of ContextBundle by calling external MCP tools.
    """

    @property
    def name(self) -> str:
        """Human-readable name of this gatherer."""
        ...

    @property
    def required_mcp(self) -> str:
        """Name of the MCP this gatherer requires."""
        ...

    async def is_available(self) -> bool:
        """Check if this gatherer can operate.

        Returns:
            True if the required MCP is configured and accessible
        """
        ...

    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Gather context relevant to the given intent.

        Args:
            intent: Analyzed intent from the user's prompt
            depth: How much context to gather ('minimal', 'auto', 'comprehensive')

        Returns:
            GathererResult with success status and partial ContextBundle
        """
        ...


class BaseGatherer(ABC):
    """Abstract base class for context gatherers.

    Provides common boilerplate for MCP-based gatherers including:
    - Client registry access
    - Timeout configuration
    - is_available() implementation

    Subclasses must implement:
    - name property
    - required_mcp property
    - gather() method
    """

    def __init__(self, registry: "MCPClientRegistry", timeout: float = 3.0) -> None:
        """Initialize gatherer with client registry.

        Args:
            registry: MCP client registry for connections
            timeout: Timeout for MCP operations (default: 3.0 seconds)
        """
        self._registry = registry
        self._timeout = timeout

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this gatherer."""
        ...

    @property
    @abstractmethod
    def required_mcp(self) -> str:
        """Name of the MCP this gatherer requires."""
        ...

    async def is_available(self) -> bool:
        """Check if this gatherer can operate.

        Returns:
            True if the required MCP is configured
        """
        return self._registry.is_configured(self.required_mcp)

    @abstractmethod
    async def gather(
        self,
        intent: Intent,
        depth: str = "auto",
    ) -> GathererResult:
        """Gather context relevant to the given intent.

        Args:
            intent: Analyzed intent from the user's prompt
            depth: How much context to gather ('minimal', 'auto', 'comprehensive')

        Returns:
            GathererResult with success status and partial ContextBundle
        """
        ...
