# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project initialization with comprehensive system design blueprint
- GitHub repository setup with 16 epics and 59 individual tickets
- Agent workflow framework for systematic development
- Issue tracking with phase-based sequencing (Phase 1-5)
- Comprehensive labeling system for tickets and epics
- [EP-1-T1] Flask application scaffold with health endpoint (v0.1.0)
- [EP-1-T2] Development tooling (pytest, coverage, black, isort, flake8, mypy) (v0.2.0)
- [EP-1-T4] Makefile & pre-commit hooks (v0.3.0)
- [EP-1-T3] Config management via .env and Pydantic Settings (v0.4.0)
- [EP-1-T5] Dockerfile & dev docker-compose (v0.5.0)
- [EP-2-T1] Define JSON schema v0.1 for Scenario (v0.6.0)
- [EP-2-T2] SQLAlchemy models - User, Scenario, Run, LedgerRow (v0.7.0)
- [EP-2-T4] S3/local storage for exports & run artifacts (v0.8.0)
- [EP-16-T1] GitHub Actions CI workflow for automated testing (v0.9.0)
- [EP-5-T1] Cursor rules for automated agent guidance (v0.10.0)
- [EP-16-T1] Improved CI workflow to enforce quality gates (v0.11.0)
- [EP-3-T1] Time grid & unit system (annual; real vs nominal) (v0.12.0)
- [EP-3-T2] Mortgage amortization module (v0.13.0)
- [EP-3-T3] Baseline expenses & lumpy events module (v0.14.0)
- [EP-3-T4] Account balance evolution module (v0.15.0)

### Changed

### Deprecated

### Removed

### Fixed

### Security

---

## Development Notes

This changelog will be updated by agents as they implement tickets. Each ticket implementation should add an entry under the `[Unreleased]` section with the format:

- `[EP-X-TY] Brief description of what was implemented (vX.Y.Z)`

**Version Management:**
- Each major feature implementation should include a version number
- Version numbers follow semantic versioning (MAJOR.MINOR.PATCH)
- When releases are created, the `[Unreleased]` section will be moved to a version number (e.g., `[1.0.0]`)

**Development Environment:**
- EP-1-T1: Work locally to establish foundation ✅ **COMPLETED**
- EP-1-T5: Set up Docker environment ✅ **COMPLETED**
- EP-1-T2: Add local development tooling (pytest, coverage, black, isort, flake8/ruff, mypy) ✅ **COMPLETED**
- EP-1-T4: Makefile & pre-commit hooks ✅ **COMPLETED**
- EP-1-T3: Config management via `.env` and Pydantic Settings ✅ **COMPLETED**
- **Phase 1 Foundation Complete** 🎉
- EP-2-T1: Define JSON schema v0.1 for Scenario ✅ **COMPLETED** (PR #87)
- EP-2-T2: SQLAlchemy models - User, Scenario, Run, LedgerRow ✅ **COMPLETED** (PR #101)
- EP-2-T4: S3/local storage for exports & run artifacts 🔄 **IN REVIEW** (PR #100)
- EP-16-T1: GitHub Actions CI workflow ✅ **COMPLETED** (automated testing on push/PR)
- EP-2-T3: Scenario versioning & immutable base+diff 🔄 **BLOCKED** (low priority)
- EP-3-T1: Time grid & unit system ✅ **COMPLETED**
- EP-3-T2: Mortgage amortization module ✅ **COMPLETED**
- EP-3-T3: Baseline expenses & lumpy events module ✅ **COMPLETED**
- Starting with EP-1-T5: All development should use Docker (`docker compose up -d`)
- All testing, linting, and type checking should be done in the Docker environment

**Updated Sequencing Strategy:**
- **Phase 1**: Foundation (EP-1-T1, EP-1-T5, EP-1-T2, EP-1-T4, EP-1-T3)
- **Phase 2-4**: Core functionality and features
- **Phase 6**: CI/CD & DevOps (moved from Phase 5 to after basic functionality)
