"""
Orchestration blueprint for retirement planning simulation runs.

This module provides API endpoints for orchestrating simulation runs,
including starting runs, checking status, and retrieving results.
"""

from datetime import datetime
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.database.models import Run, VersionedScenario
from app.services.orchestration_service import OrchestrationService

orchestration_bp = Blueprint("orchestration", __name__, url_prefix="/api")


@orchestration_bp.route("/scenarios/<int:scenario_id>/runs", methods=["POST"])
def start_run(scenario_id: int) -> Any:
    """Start a new simulation run for a scenario.

    Args:
        scenario_id: ID of the scenario to run

    Returns:
        JSON response with run_id and status
    """
    try:
        # Get request data
        data = request.get_json() or {}
        run_type = data.get("run_type", "monte_carlo")
        num_simulations = data.get("num_simulations", 1000)

        # Validate run_type
        if run_type not in ["monte_carlo", "deterministic", "historical"]:
            return jsonify({"error": "Invalid run_type"}), 400

        # Validate num_simulations
        if not isinstance(num_simulations, int) or num_simulations <= 0:
            return jsonify({"error": "num_simulations must be a positive integer"}), 400

        # Get database session
        db: Session = next(get_db())

        # Check if scenario exists
        scenario = (
            db.query(VersionedScenario)
            .filter(VersionedScenario.id == scenario_id)
            .first()
        )

        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404

        # Create new run record
        run = Run(
            versioned_scenario_id=scenario_id,
            status="pending",
            run_type=run_type,
            num_simulations=num_simulations,
            created_at=datetime.utcnow(),
        )

        db.add(run)
        db.commit()
        db.refresh(run)

        # Start orchestration service
        orchestration_service = OrchestrationService()

        # Start the run asynchronously (for now, we'll run synchronously)
        # In a production system, this would be queued with Celery or similar
        try:
            orchestration_service.run_simulation(
                int(run.id), scenario.scenario_config, run_type, num_simulations
            )

            # Update run status to completed
            run.status = "completed"  # type: ignore
            run.completed_at = datetime.utcnow()  # type: ignore
            db.commit()

        except Exception as e:
            # Update run status to failed
            run.status = "failed"  # type: ignore
            run.error_message = str(e)  # type: ignore
            run.completed_at = datetime.utcnow()  # type: ignore
            db.commit()

            return jsonify({"error": "Simulation failed", "message": str(e)}), 500

        return (
            jsonify(
                {
                    "run_id": run.id,
                    "status": run.status,
                    "run_type": run.run_type,
                    "num_simulations": run.num_simulations,
                    "created_at": run.created_at.isoformat(),
                    "completed_at": (
                        run.completed_at.isoformat() if run.completed_at else None
                    ),
                }
            ),
            201,
        )

    except Exception as e:
        current_app.logger.error(f"Error starting run: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@orchestration_bp.route("/runs/<int:run_id>", methods=["GET"])
def get_run_status(run_id: int) -> Any:
    """Get the status and results of a simulation run.

    Args:
        run_id: ID of the run to check

    Returns:
        JSON response with run status and results
    """
    try:
        # Get database session
        db: Session = next(get_db())

        # Get run record
        run = db.query(Run).filter(Run.id == run_id).first()

        if not run:
            return jsonify({"error": "Run not found"}), 404

        response_data = {
            "run_id": run.id,
            "scenario_id": run.versioned_scenario_id,
            "status": run.status,
            "run_type": run.run_type,
            "num_simulations": run.num_simulations,
            "created_at": run.created_at.isoformat(),
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

        # Include results if completed
        if run.status == "completed" and run.results:
            response_data["results"] = run.results

        # Include error message if failed
        if run.status == "failed" and run.error_message:
            response_data["error_message"] = run.error_message

        return jsonify(response_data), 200

    except Exception as e:
        current_app.logger.error(f"Error getting run status: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@orchestration_bp.route("/runs/<int:run_id>/results", methods=["GET"])
def get_run_results(run_id: int) -> Any:
    """Get the results of a completed simulation run.

    Args:
        run_id: ID of the run to get results for

    Returns:
        JSON response with run results
    """
    try:
        # Get database session
        db: Session = next(get_db())

        # Get run record
        run = db.query(Run).filter(Run.id == run_id).first()

        if not run:
            return jsonify({"error": "Run not found"}), 404

        if run.status != "completed":
            return jsonify({"error": "Run not completed"}), 400

        if not run.results:
            return jsonify({"error": "No results available"}), 404

        return jsonify({"run_id": run.id, "results": run.results}), 200

    except Exception as e:
        current_app.logger.error(f"Error getting run results: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@orchestration_bp.route("/scenarios/<int:scenario_id>/runs", methods=["GET"])
def list_scenario_runs(scenario_id: int) -> Any:
    """List all runs for a scenario.

    Args:
        scenario_id: ID of the scenario

    Returns:
        JSON response with list of runs
    """
    try:
        # Get database session
        db: Session = next(get_db())

        # Check if scenario exists
        scenario = (
            db.query(VersionedScenario)
            .filter(VersionedScenario.id == scenario_id)
            .first()
        )

        if not scenario:
            return jsonify({"error": "Scenario not found"}), 404

        # Get runs for scenario
        runs = (
            db.query(Run)
            .filter(Run.versioned_scenario_id == scenario_id)
            .order_by(Run.created_at.desc())
            .all()
        )

        runs_data = []
        for run in runs:
            run_data = {
                "run_id": run.id,
                "status": run.status,
                "run_type": run.run_type,
                "num_simulations": run.num_simulations,
                "created_at": run.created_at.isoformat(),
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": (
                    run.completed_at.isoformat() if run.completed_at else None
                ),
            }

            if run.status == "failed" and run.error_message:
                run_data["error_message"] = run.error_message

            runs_data.append(run_data)

        return jsonify({"scenario_id": scenario_id, "runs": runs_data}), 200

    except Exception as e:
        current_app.logger.error(f"Error listing scenario runs: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
