"""Quality Standards - Repository of coding standards by language and framework."""

from pathlib import Path
from typing import Any

import yaml

from mirdan.config import QualityConfig
from mirdan.models import Intent


class QualityStandards:
    """Repository of quality standards by language and framework."""

    def __init__(
        self,
        standards_dir: Path | None = None,
        config: QualityConfig | None = None,
    ):
        """Initialize with optional custom standards directory and quality config.

        Args:
            standards_dir: Directory with custom YAML standards
            config: Quality config for stringency levels
        """
        self._config = config
        self.standards_dir = standards_dir
        self.standards = self._load_default_standards()
        if standards_dir and standards_dir.exists():
            self._load_custom_standards(standards_dir)

    def _get_stringency_count(self, category: str) -> int:
        """Get the number of standards to include based on stringency level.

        Args:
            category: Category name (security, architecture, documentation, testing)

        Returns:
            Number of standards to include (5 for strict, 3 for moderate, 1 for permissive)
        """
        if not self._config:
            return 3  # Default: moderate

        level = getattr(self._config, category, "moderate")
        stringency_map = {"strict": 5, "moderate": 3, "permissive": 1}
        return stringency_map.get(level, 3)

    def _load_yaml_file(self, file_traversable: Any, category: str) -> dict[str, Any]:
        """Load a single YAML file with error handling.

        Args:
            file_traversable: Traversable object pointing to YAML file
            category: Category name for logging purposes

        Returns:
            Parsed YAML content as dict, or empty dict on error
        """
        import logging

        logger = logging.getLogger(__name__)
        try:
            content = file_traversable.read_text()
            parsed = yaml.safe_load(content)
            return parsed if parsed else {}
        except FileNotFoundError:
            logger.warning(f"Standards file not found for category: {category}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in standards file {category}: {e}")
            return {}

    def _load_default_standards(self) -> dict[str, Any]:
        """Load built-in quality standards from YAML files."""
        import logging
        from importlib.resources import files

        logger = logging.getLogger(__name__)
        standards: dict[str, Any] = {}

        try:
            standards_pkg = files("mirdan.standards")
        except ModuleNotFoundError:
            logger.warning("mirdan.standards package not found, using empty standards")
            return {}

        # Load language standards
        languages = ["typescript", "python", "javascript", "rust", "go", "java"]
        for lang in languages:
            lang_file = standards_pkg.joinpath("languages", f"{lang}.yaml")
            standards[lang] = self._load_yaml_file(lang_file, lang)

        # Load all framework standards dynamically from the frameworks directory
        # Map filename to standard name (e.g., nextjs.yaml -> next.js)
        filename_to_name = {
            "nextjs": "next.js",
            "springboot": "spring-boot",
        }
        frameworks_dir = standards_pkg.joinpath("frameworks")
        try:
            for item in frameworks_dir.iterdir():
                if item.name.endswith(".yaml"):
                    # Remove .yaml extension to get base name
                    base_name = item.name[:-5]
                    # Apply name mapping for backwards compatibility
                    framework_name = filename_to_name.get(base_name, base_name)
                    standards[framework_name] = self._load_yaml_file(item, framework_name)
        except (FileNotFoundError, TypeError):
            logger.warning("frameworks directory not found or not iterable")

        # Load security standards (root level)
        security_file = standards_pkg.joinpath("security.yaml")
        standards["security"] = self._load_yaml_file(security_file, "security")

        # Load architecture standards (root level)
        arch_file = standards_pkg.joinpath("architecture.yaml")
        standards["architecture"] = self._load_yaml_file(arch_file, "architecture")

        # Load RAG pipeline standards (domain-level, cross-cutting)
        rag_file = standards_pkg.joinpath("rag_pipelines.yaml")
        standards["rag_pipelines"] = self._load_yaml_file(rag_file, "rag_pipelines")

        # Load knowledge graph standards (domain-level, cross-cutting)
        kg_file = standards_pkg.joinpath("knowledge_graphs.yaml")
        standards["knowledge_graphs"] = self._load_yaml_file(kg_file, "knowledge_graphs")

        return standards

    def _load_custom_standards(self, standards_dir: Path) -> None:
        """Load custom standards from YAML files."""
        for yaml_file in standards_dir.rglob("*.yaml"):
            with yaml_file.open() as f:
                custom = yaml.safe_load(f)
                if custom:
                    # Merge custom standards
                    for key, value in custom.items():
                        if key in self.standards:
                            self.standards[key].update(value)
                        else:
                            self.standards[key] = value

    def get_for_language(self, language: str) -> dict[str, Any]:
        """Get standards for a specific language."""
        result: dict[str, Any] = self.standards.get(language, {})
        return result

    def get_for_framework(self, framework: str) -> dict[str, Any]:
        """Get standards for a specific framework."""
        result: dict[str, Any] = self.standards.get(framework, {})
        return result

    def get_security_standards(self) -> dict[str, Any]:
        """Get security-related standards."""
        result: dict[str, Any] = self.standards.get("security", {})
        return result

    def get_architecture_standards(self) -> dict[str, Any]:
        """Get architecture standards."""
        result: dict[str, Any] = self.standards.get("architecture", {})
        return result

    def render_for_intent(self, intent: Intent) -> list[str]:
        """Render relevant standards for a given intent."""
        requirements: list[str] = []

        # Add language-specific standards
        if intent.primary_language:
            lang_standards = self.get_for_language(intent.primary_language)
            if "principles" in lang_standards:
                # Use moderate (3) for language principles - not category-specific
                requirements.extend(lang_standards["principles"][:3])

        # Add framework-specific standards
        if intent.frameworks:
            fw_count = self._get_stringency_count("framework")
            for framework in intent.frameworks:
                fw_standards = self.get_for_framework(framework)
                if "principles" in fw_standards:
                    requirements.extend(fw_standards["principles"][:fw_count])

        # Add security standards if relevant (use security stringency)
        if intent.touches_security:
            sec_count = self._get_stringency_count("security")
            sec_standards = self.get_security_standards()
            if "authentication" in sec_standards:
                requirements.extend(sec_standards["authentication"][:sec_count])
            if "input_validation" in sec_standards:
                requirements.extend(sec_standards["input_validation"][:sec_count])

        # Add RAG pipeline standards if relevant
        if intent.touches_rag:
            rag_count = self._get_stringency_count("framework")
            rag_standards = self.standards.get("rag_pipelines", {})
            if "principles" in rag_standards:
                requirements.extend(rag_standards["principles"][:rag_count])

            # Add knowledge graph standards if neo4j is detected
            if "neo4j" in intent.frameworks:
                kg_standards = self.standards.get("knowledge_graphs", {})
                if "principles" in kg_standards:
                    requirements.extend(kg_standards["principles"][:rag_count])

        # Add architecture standards (use architecture stringency)
        arch_count = self._get_stringency_count("architecture")
        arch_standards = self.get_architecture_standards()
        if "general" in arch_standards:
            requirements.extend(arch_standards["general"][:arch_count])

        return requirements

    def get_all_standards(
        self,
        language: str | None = None,
        framework: str | None = None,
        category: str = "all",
    ) -> dict[str, Any]:
        """Get standards filtered by language, framework, and category."""
        result: dict[str, Any] = {}

        # Get language standards
        if language:
            lang_standards = self.get_for_language(language.lower())
            if lang_standards:
                result["language_standards"] = lang_standards

        # Get framework standards if requested
        if framework:
            fw_standards = self.get_for_framework(framework.lower())
            if fw_standards:
                result["framework_standards"] = fw_standards

        # Get security standards if requested
        if category in ["all", "security"]:
            result["security_standards"] = self.get_security_standards()

        # Get architecture standards if requested
        if category in ["all", "architecture"]:
            result["architecture_standards"] = self.get_architecture_standards()

        # Get RAG and knowledge graph standards if requested
        if category in ["all", "rag"]:
            rag_standards = self.standards.get("rag_pipelines", {})
            if rag_standards:
                result["rag_standards"] = rag_standards
            kg_standards = self.standards.get("knowledge_graphs", {})
            if kg_standards:
                result["knowledge_graph_standards"] = kg_standards

        return result
