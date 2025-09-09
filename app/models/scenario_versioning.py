"""
Simple scenario versioning service that stores full copies.

This implementation prioritizes simplicity and maintainability over storage efficiency.
Full scenario copies are stored for each version, with DeepDiff used only for
comparison and display purposes.

Future optimization: Implement base + diff storage when storage becomes a concern.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from deepdiff import DeepDiff

from app.database.models import VersionedScenario, User
from app.models.scenario import Scenario


class SimpleScenarioVersioningService:
    """Simple scenario versioning that stores full copies."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_version(
        self, 
        user_id: int, 
        scenario_data: Scenario, 
        name: str, 
        description: Optional[str] = None,
        parent_version_id: Optional[int] = None
    ) -> str:
        """
        Create a new scenario version (full copy).
        
        Args:
            user_id: ID of the user creating the version
            scenario_data: The scenario data to store
            name: Human-readable name for this version
            description: Optional description
            parent_version_id: ID of parent version (for branching)
            
        Returns:
            scenario_id: Unique identifier for the new version
        """
        # Generate unique scenario ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Include microseconds
        scenario_id = f"scenario_{timestamp}_{user_id}"
        
        # Determine version number
        version = self._get_next_version(user_id, parent_version_id)
        
        # Create versioned scenario
        versioned_scenario = VersionedScenario(
            scenario_id=scenario_id,
            user_id=user_id,
            name=name,
            description=description,
            scenario_data=scenario_data.model_dump(mode='json'),  # Full copy with JSON serialization
            parent_version_id=parent_version_id,
            version=version,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(versioned_scenario)
        self.db.commit()
        
        return scenario_id
    
    def get_scenario(self, scenario_id: str) -> Scenario:
        """
        Get a scenario version (direct load).
        
        Args:
            scenario_id: Unique identifier for the scenario version
            
        Returns:
            Scenario: The scenario data as a Pydantic model
            
        Raises:
            ValueError: If scenario_id not found
        """
        versioned = self.db.query(VersionedScenario).filter(
            VersionedScenario.scenario_id == scenario_id
        ).first()
        
        if not versioned:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        return Scenario(**versioned.scenario_data)
    
    def get_version_info(self, scenario_id: str) -> Dict[str, Any]:
        """
        Get metadata about a scenario version.
        
        Args:
            scenario_id: Unique identifier for the scenario version
            
        Returns:
            Dict with version metadata
        """
        versioned = self.db.query(VersionedScenario).filter(
            VersionedScenario.scenario_id == scenario_id
        ).first()
        
        if not versioned:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        return {
            "scenario_id": versioned.scenario_id,
            "name": versioned.name,
            "description": versioned.description,
            "version": versioned.version,
            "parent_version_id": versioned.parent_version_id,
            "created_at": versioned.created_at,
            "updated_at": versioned.updated_at,
            "user_id": versioned.user_id
        }
    
    def list_user_scenarios(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all scenario versions for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of scenario metadata dictionaries
        """
        scenarios = self.db.query(VersionedScenario).filter(
            VersionedScenario.user_id == user_id
        ).order_by(VersionedScenario.created_at.desc()).all()
        
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "parent_version_id": s.parent_version_id,
                "created_at": s.created_at,
                "updated_at": s.updated_at
            }
            for s in scenarios
        ]
    
    def compare_versions(self, scenario_id_1: str, scenario_id_2: str) -> Dict[str, Any]:
        """
        Compare two scenario versions using DeepDiff.
        
        Args:
            scenario_id_1: First scenario version ID
            scenario_id_2: Second scenario version ID
            
        Returns:
            Dict with comparison results
        """
        scenario_1 = self.get_scenario(scenario_id_1)
        scenario_2 = self.get_scenario(scenario_id_2)
        
        # Use DeepDiff for comparison
        diff = DeepDiff(
            scenario_1.model_dump(), 
            scenario_2.model_dump(),
            ignore_order=True,
            exclude_paths=["root['metadata']['created_at']", "root['metadata']['updated_at']"]
        )
        
        return {
            "scenario_1": scenario_id_1,
            "scenario_2": scenario_id_2,
            "changes": diff,
            "has_changes": bool(diff)
        }
    
    def get_version_history(self, scenario_id: str) -> List[Dict[str, Any]]:
        """
        Get the version history for a scenario (parent chain).
        
        Args:
            scenario_id: Starting scenario version ID
            
        Returns:
            List of version metadata in chronological order
        """
        # Find the root version (no parent)
        current = self.db.query(VersionedScenario).filter(
            VersionedScenario.scenario_id == scenario_id
        ).first()
        
        if not current:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        # Walk up to find root
        while current.parent_version_id:
            current = self.db.query(VersionedScenario).filter(
                VersionedScenario.id == current.parent_version_id
            ).first()
        
        # Walk down to build history
        history = []
        while current:
            history.append({
                "scenario_id": current.scenario_id,
                "name": current.name,
                "description": current.description,
                "version": current.version,
                "created_at": current.created_at,
                "updated_at": current.updated_at
            })
            
            # Find next child version
            current = self.db.query(VersionedScenario).filter(
                VersionedScenario.parent_version_id == current.id
            ).first()
        
        return history
    
    def _get_next_version(self, user_id: int, parent_version_id: Optional[int] = None) -> str:
        """Get the next version number for a user's scenario."""
        if parent_version_id:
            # This is a branch from an existing version
            parent = self.db.query(VersionedScenario).filter(
                VersionedScenario.id == parent_version_id
            ).first()
            if parent:
                base_version = parent.version
                # Find existing branches from this parent
                existing_branches = self.db.query(VersionedScenario).filter(
                    VersionedScenario.parent_version_id == parent_version_id
                ).count()
                return f"{base_version}.{existing_branches + 1}"
        
        # This is a new scenario
        user_scenario_count = self.db.query(VersionedScenario).filter(
            VersionedScenario.user_id == user_id
        ).count()
        
        return f"v{user_scenario_count + 1}"