# Agent Workflow Framework

## ⚠️ CRITICAL: READ THIS FIRST

**FOR ALL AGENTS:** You MUST read this entire document before starting any work. This workflow is mandatory and includes:
- Creating pull requests for every ticket (NEVER merge directly to main)
- Following the exact ticket sequencing
- Using Docker for development
- Proper testing and quality gates

**FOR HUMAN USERS:** Always tell new agents to read this document first.

## 🎯 Cursor Rules Integration

**NEW:** This project now includes Cursor rules (`.cursor/rules/`) that automatically guide AI agents. These rules are automatically applied based on context and provide structured guidance for all development work.

### Available Cursor Rules
- **001-foundation-workflow.mdc**: Core workflow requirements (always applied)
- **002-development-process.mdc**: Implementation workflow (auto-attached to Python files)
- **003-ticket-management.mdc**: GitHub issue workflow (agent requested)
- **004-quality-gates.mdc**: Testing and validation (auto-attached to test files)
- **005-documentation-standards.mdc**: Documentation maintenance (manual invocation)

### How Cursor Rules Work
- **Always Rules**: Foundation workflow is automatically included in every agent session
- **Auto Attached**: Development and quality rules are applied when working on relevant files
- **Agent Requested**: Ticket management rules are available for on-demand inclusion
- **Manual**: Documentation rules are applied when explicitly invoked with `@ruleName`

### Benefits
- **Consistent behavior** across all agent interactions
- **Reduced manual instruction** repetition
- **Automatic context awareness** based on files being worked on
- **Structured guidance** for complex workflows

**Note**: While Cursor rules provide automatic guidance, this document remains the authoritative source for complete workflow understanding and reference.

## Overview
This document outlines how Cursor agents can systematically work through the retirement planner tickets in the correct order, creating pull requests for each implementation.

## Current Status (Updated: Latest)
- **Phase 1 (Foundation)**: ✅ **COMPLETED** - All infrastructure tickets done
- **Phase 2 (Core Data & Engine)**: ✅ **COMPLETED** - All tickets completed
  - ✅ **COMPLETED**: EP-2-T1 (JSON schema v0.1) - PR #87
  - ✅ **COMPLETED**: EP-2-T2 (SQLAlchemy models) - PR #101
  - ✅ **COMPLETED**: EP-2-T4 (Storage backend) - PR #100
  - 🔄 **BLOCKED**: EP-2-T3 (Scenario versioning) - low priority
- **Phase 3 (Simulation Engines)**: ✅ **COMPLETED** - All tickets completed
  - ✅ **COMPLETED**: EP-3-T1 (Time grid & unit system) - ready for PR
  - ✅ **COMPLETED**: EP-3-T2 (Mortgage amortization module) - ready for PR
  - ✅ **COMPLETED**: EP-3-T3 (Baseline expenses & lumpy events) - ready for PR
  - ✅ **COMPLETED**: EP-3-T4 (Account balance evolution) - ready for PR
  - ✅ **COMPLETED**: EP-3-T5 (Social Security stub) - ready for PR
- **Phase 4 (Monte Carlo Engine)**: 🔄 **IN PROGRESS** - 0/5 tickets completed
  - Next: EP-4-T1 (Random returns generator)
- **Phase 6 (CI/CD & DevOps)**: 🔄 **IN PROGRESS** - 1/4 tickets completed
  - ✅ **COMPLETED**: EP-16-T1 (Improved CI workflow) - quality gates enforced
  - Next: EP-16-T2 (Containerized deploy)

## Ticket Sequencing Strategy

### Phase 1: Foundation (Must be completed first) ✅ **COMPLETED**
**Order 1-5**: Core infrastructure that everything else depends on
- EP-1-T1: Create repo scaffold (Flask app factory, blueprints, config) ✅ **COMPLETED**
- EP-1-T5: Dockerfile & dev docker-compose ✅ **COMPLETED**
- EP-1-T2: Add tooling (pytest, coverage, black, isort, flake8/ruff, mypy) ✅ **COMPLETED**
- EP-1-T4: Makefile & pre-commit hooks ✅ **COMPLETED**
- EP-1-T3: Config management via `.env` and Pydantic Settings ✅ **COMPLETED**

### Phase 2: Core Data & Engine (Parallel after foundation) ✅ **COMPLETED**
**Order 6-12**: Data models and basic engine components
- EP-2-T1: Define JSON schema v0.1 for Scenario ✅ **COMPLETED**
- EP-2-T2: SQLAlchemy models (User, Scenario, Run, LedgerRow) ✅ **COMPLETED**
- EP-2-T3: Scenario versioning & immutable base+diff 🔄 **BLOCKED** (low priority)
- EP-2-T4: Storage backend ✅ **COMPLETED** (PR #100)

### Phase 3: Simulation Engines ✅ **COMPLETED**
**Order 13-19**: Core simulation logic
- EP-3-T1: Time grid & unit system (annual; real vs nominal) ✅ **COMPLETED**
- EP-3-T2: Mortgage amortization module ✅ **COMPLETED**
- EP-3-T3: Baseline expenses & lumpy events ✅ **COMPLETED**
- EP-3-T4: Account balance evolution (taxable/trad/Roth) ✅ **COMPLETED**
- EP-3-T5: Social Security stub (fixed input benefit) ✅ **COMPLETED**
- EP-4-T1: Random returns generator (normal/lognormal selectable)
- EP-4-T2: Correlated draws via covariance (Cholesky)
- EP-4-T3: Portfolio evolution w/ annual rebalance
- EP-4-T4: Withdrawal rules (4% real, fixed %, VPW scaffold)
- EP-4-T5: Success metrics & percentiles

### Phase 4: API and UI
**Order 20-27**: User-facing interfaces
- EP-11-T1: Auth (session or token) + RBAC
- EP-11-T2: CRUD endpoints for Scenario
- EP-11-T3: Run orchestration endpoint
- EP-11-T4: Export endpoints (CSV ledger, PDF report stub)
- EP-12-T1: Scenario builder form (Quick Start)
- EP-12-T2: Scenario list + clone/diff
- EP-12-T3: Results dashboard
- EP-12-T4: Compare view (A/B)

### Phase 5: Domain Modules (Can be parallel)
**Order 28-60**: Specialized features
- Tax engine, withdrawal strategies, housing, college, benefits, etc.

### Phase 6: CI/CD & DevOps (After basic functionality)
**Order 61+**: Production infrastructure
- EP-16-T1: GitHub Actions CI (test/lint/type/perf) ✅ **COMPLETED**
- EP-16-T2: Containerized deploy (Render/Fly/Heroku)
- EP-16-T3: Basic security hardening
- EP-16-T4: Error monitoring & logging

## Agent Workflow Process

### 1. Agent Initialization
When a fresh agent starts, it should:

```bash
# 1. Read the project documentation for context
cat README.md
cat AGENT_WORKFLOW.md

# 2. Check current repository state
git status
git log --oneline -5

# 3. Check if changelog exists, create if needed
ls -la CHANGELOG.md || echo "Changelog not found - will create one"

# 4. Find the next ticket to work on
# First, check Phase 1 tickets (foundation)
gh issue list --label "phase:1" --state open

# If Phase 1 is complete, check Phase 2
gh issue list --label "phase:2" --state open

# Continue through phases until you find open tickets
gh issue list --label "phase:3" --state open
gh issue list --label "phase:4" --state open
gh issue list --label "phase:5" --state open
```

### 2. Ticket Selection Logic
The agent should:
1. **Check phase completion**: Ensure all previous phase tickets are closed
2. **Find next available ticket**: Look for the lowest order number in current phase
3. **Verify dependencies**: Check that all dependency tickets are completed
4. **Assign ticket**: Assign the ticket to itself

**Simple Command Sequence:**
```bash
# Check if Phase 1 has any open tickets
gh issue list --label "phase:1" --state open

# If Phase 1 is empty, check Phase 2
gh issue list --label "phase:2" --state open

# Once you find a phase with open tickets, pick the first one
# (they should be in order based on the implementation order comments)

# Assign the ticket to yourself
gh issue edit [ISSUE_NUMBER] --assignee @me
```

### 3. Implementation Process
For each ticket:

1. **Create feature branch**:
   ```bash
   git checkout -b feature/EP-X-TY-short-description
   ```

2. **Implement the ticket**:
   - **Start Docker environment**: `docker compose up -d` (if not already running)
   - Follow the acceptance criteria exactly
   - Write tests as specified
   - **Run tests in container**: `docker compose exec app make test`
   - **Run linting in container**: `docker compose exec app make lint`
   - **Run type checking in container**: `docker compose exec app make typecheck`

3. **Update documentation** (REQUIRED - DO BEFORE PR):
   
   **A. Update CHANGELOG.md:**
   ```bash
   # Add entry to CHANGELOG.md under [Unreleased] section
   # Format: - [EP-X-TY] Brief description of what was implemented (vX.Y.Z)
   # Example: - [EP-2-T2] SQLAlchemy models - User, Scenario, Run, LedgerRow (v0.7.0)
   ```
   
   **B. Update AGENT_WORKFLOW.md:**
   ```bash
   # 1. Mark the completed ticket with ✅ **COMPLETED**
   # 2. Update "Next" indicators to point to the next available ticket
   # 3. Update current status section at the top
   # 4. Update development notes in CHANGELOG.md if needed
   ```
   
   **Documentation Update Checklist:**
   - [ ] Added changelog entry with proper format
   - [ ] Marked ticket as completed in workflow document
   - [ ] Updated "Next" indicators
   - [ ] Updated current status section
   - [ ] Updated development notes if applicable

4. **Create pull request** (REQUIRED - DO NOT SKIP):
   ```bash
   git add .
   git commit -m "feat: implement EP-X-TY - brief description

   - Implemented all acceptance criteria
   - Added comprehensive tests
   - Updated changelog and workflow documentation
   - All quality gates passed"
   git push origin feature/EP-X-TY-short-description
   gh pr create --title "EP-X-TY: Ticket Title" --body "Implements EP-X-TY

   **Acceptance Criteria**: [Copy from ticket]
   **Tests**: [Copy from ticket]
   **Dependencies**: [Copy from ticket]

   **Documentation Updates**:
   - Updated CHANGELOG.md with new entry
   - Updated AGENT_WORKFLOW.md status and next indicators

   Closes #[issue-number]"
   ```
   
   **⚠️ CRITICAL: You MUST create a PR for every ticket. Do not merge directly to main.**

5. **Verify GitHub Actions CI passes**:
   ```bash
   # Check that the CI workflow passes before marking as ready for review
   gh run list --limit 5
   gh run watch [LATEST_RUN_ID]
   
   # If CI fails, fix issues and push updates
   # Only proceed to step 6 when CI shows green/checkmark
   ```

6. **Update status and add progress comment** (ONLY after CI passes):
   ```bash
   # Remove in-progress label and add review label
   gh issue edit [ISSUE_NUMBER] --remove-label "status:in-progress" --add-label "status:review"

   # Add progress comment
   gh issue comment [ISSUE_NUMBER] --body "✅ **Implementation complete**

   - Created feature branch: feature/EP-X-TY-short-description
   - Implemented all acceptance criteria
   - Added tests as specified
   - Updated changelog and workflow documentation
   - Created PR: #[PR_NUMBER]
   - ✅ **GitHub Actions CI passing** - All tests, linting, and checks pass

   🤖 Agent waiting for human review and merge."
   ```

7. **Wait for review**: Agent should not proceed until PR is merged

### 4. Post-Merge Process
After PR is merged:

1. **Update local repository**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Update changelog with version**:
   ```bash
   # Update the changelog to move from [Unreleased] to a version
   # This should be done when creating a release, but for now just note completion
   ```

3. **Mark as completed and close the ticket**:
   ```bash
   # Update status label
   gh issue edit [issue-number] --remove-label "status:review" --add-label "status:completed"

   # Close with detailed comment
   gh issue close [issue-number] --comment "✅ **COMPLETED**

   **Implementation Summary**:
   - All acceptance criteria met
   - Tests implemented and passing
   - Code reviewed and merged in PR #[pr-number]
   - Changelog and workflow documentation updated

   🤖 **Agent Status**: Ready for next ticket in sequence."
   ```

4. **Find next ticket**:
   ```bash
   # Check next phase or continue with current phase
   gh issue list --label "phase:1" --state open
   gh issue list --label "phase:2" --state open
   # etc.
   ```

## Ensuring Agents Read This Workflow

**For Human Users:** When starting a new agent session, you MUST provide these instructions to ensure the agent follows the proper workflow:

1. **Always start with**: "Please read the AGENT_WORKFLOW.md document first before starting any work"
2. **Verify the agent has read it**: Ask the agent to summarize the key steps
3. **Reference the workflow**: Throughout the session, remind the agent to follow the workflow steps

**For Agents:** You MUST read this entire document before starting any work. This is not optional.

## Agent Instructions Template

When starting a new agent session, provide these instructions:

```
You are working on the retirement planner project. Your task is to:

1. Find the next available ticket to implement
2. Implement it according to the acceptance criteria
3. Create a pull request
4. Wait for human review and merge
5. Close the ticket and move to the next one

Current repository: https://github.com/jacobbaron/retirement-planner

**IMPORTANT: Start by reading the project documentation:**
```bash
# Read the README to understand the project
cat README.md

# Read this workflow document
cat AGENT_WORKFLOW.md

# Check if changelog exists
ls -la CHANGELOG.md

# Check if Docker is available (required for development)
docker --version
docker compose --version

# Check Cursor rules are available
ls -la .cursor/rules/
```

**CURSOR RULES:** This project includes Cursor rules that automatically guide your work:
- Foundation workflow rules are always applied
- Development process rules activate when working on Python files
- Quality gates rules activate when working on test files
- Use `@ruleName` to invoke specific rules manually

To find the next ticket, run these commands in order:
```bash
# Phase 1 is complete, so check Phase 2 first
gh issue list --label "phase:2" --state open --exclude-label "status:in-progress" --exclude-label "status:review" --exclude-label "status:completed" --exclude-label "status:blocked"

# Current Phase 2 status:
# - EP-2-T1: ✅ COMPLETED (PR #87)
# - EP-2-T2: ✅ COMPLETED (PR #101) 
# - EP-2-T3: 🔄 NEXT (ready for implementation)
# - EP-2-T4: 🔄 IN REVIEW (PR #100)

# If Phase 2 is complete, check Phase 3
gh issue list --label "phase:3" --state open --exclude-label "status:in-progress" --exclude-label "status:review" --exclude-label "status:completed" --exclude-label "status:blocked"
gh issue list --label "phase:4" --state open --exclude-label "status:in-progress" --exclude-label "status:review" --exclude-label "status:completed" --exclude-label "status:blocked"
gh issue list --label "phase:5" --state open --exclude-label "status:in-progress" --exclude-label "status:review" --exclude-label "status:completed" --exclude-label "status:blocked"
```

**Alternative: Check for blocked tickets that might be ready:**
```bash
# Check if any blocked tickets are now unblocked
gh issue list --label "status:blocked" --state open
```

Once you find a phase with open tickets:
1. Pick the first ticket in the list (they should be in order)
2. **Read the ticket thoroughly**: `gh issue view [ISSUE_NUMBER]`
3. **Read all comments**: Check the full issue page for any additional context
4. **Mark as in-progress**: `gh issue edit [ISSUE_NUMBER] --add-label "status:in-progress"`
5. **Add a comment** indicating you're starting work: `gh issue comment [ISSUE_NUMBER] --body "🤖 Agent starting work on this ticket"`

To implement a ticket:
- Create a feature branch: `git checkout -b feature/EP-X-TY-description`
- **Use Docker for development**: `docker compose up -d` (after EP-1-T5 is complete)
- **Note**: For EP-1-T1 through EP-1-T4, work locally first, then ensure Docker compatibility in EP-1-T5
- Follow the acceptance criteria exactly
- Write the specified tests
- Ensure all tests pass: `docker compose exec app make test` (or `make test` for early tickets)
- Run linting: `docker compose exec app make lint` (or `make lint` for early tickets)
- **Update documentation BEFORE creating PR**:
  - Add changelog entry to CHANGELOG.md
  - Update AGENT_WORKFLOW.md status and "Next" indicators
- Create a pull request

Do not proceed to the next ticket until the current PR is merged.
```

## Quality Gates

Each implementation must pass:
- [ ] All tests pass (`docker compose exec app make test`)
- [ ] Linting passes (`docker compose exec app make lint`)
- [ ] Type checking passes (`docker compose exec app make typecheck`)
- [ ] Coverage ≥80% (`docker compose exec app make coverage`)
- [ ] Acceptance criteria met
- [ ] Dependencies satisfied
- [ ] Docker environment working (`docker compose up -d` and health check passes)
- [ ] **Documentation updated** (CHANGELOG.md and AGENT_WORKFLOW.md)
- [ ] **GitHub Actions CI passing** (all workflow jobs must show green/checkmark)

## Error Handling

If a ticket cannot be implemented:
1. Add a comment explaining the issue
2. Update status label: `gh issue edit [ISSUE_NUMBER] --remove-label "status:in-progress" --add-label "status:blocked"`
3. Move to the next available ticket
4. Return to blocked tickets later

## Progress Tracking

The agent should maintain a simple log:
```
[Date] EP-1-T1: ✅ Completed (PR #X)
[Date] EP-1-T2: ✅ Completed (PR #Y)
[Date] EP-1-T3: 🔄 In Progress (PR #Z)
[Date] EP-1-T4: ⏳ Waiting for dependencies
```

## Documentation Maintenance

**CRITICAL**: Agents must keep documentation up to date as part of every implementation. This ensures the workflow remains accurate and helpful for future agents.

### Required Documentation Updates

**Before creating any PR, agents must:**

1. **Update CHANGELOG.md**:
   - Add entry under `[Unreleased]` section
   - Use format: `- [EP-X-TY] Brief description (vX.Y.Z)`
   - Increment version number appropriately

2. **Update AGENT_WORKFLOW.md**:
   - Mark completed ticket with ✅ **COMPLETED**
   - Update "Next" indicators to point to next available ticket
   - Update current status section at the top
   - Update development notes if applicable

3. **Verify accuracy**:
   - Ensure all status indicators are correct
   - Check that dependencies are properly marked
   - Confirm phase completion status

### Documentation Update Process

```bash
# 1. Update CHANGELOG.md
# Add entry under [Unreleased] section with proper format

# 2. Update AGENT_WORKFLOW.md
# Mark ticket as completed and update status indicators

# 3. Commit documentation changes with implementation
git add CHANGELOG.md AGENT_WORKFLOW.md
git commit -m "docs: update changelog and workflow status for EP-X-TY"
```

## Changelog Management

The agent should maintain a `CHANGELOG.md` file following the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- [EP-1-T1] Flask application scaffold with health endpoint
- [EP-1-T2] Development tooling (pytest, coverage, black, isort, flake8, mypy)

### Changed

### Deprecated

### Removed

### Fixed

### Security
```

**Changelog Update Process:**
1. Add new entries under `[Unreleased]` section
2. Use ticket ID format: `[EP-X-TY] Brief description`
3. Categorize changes appropriately (Added, Changed, Fixed, etc.)
4. When creating releases, move `[Unreleased]` to a version number

This framework ensures systematic, dependency-aware development with proper quality gates, documentation, and human oversight.

## Cursor Rules Troubleshooting

### Rule Activation Issues
If Cursor rules are not being applied automatically:

1. **Check rule file format**:
   ```bash
   # Verify .mdc files are properly formatted
   cat .cursor/rules/001-foundation-workflow.mdc
   ```

2. **Verify file paths**:
   ```bash
   # Ensure rules directory exists
   ls -la .cursor/rules/
   ```

3. **Check glob patterns**:
   - Rules with `globs` patterns only activate when working on matching files
   - Rules with `alwaysApply: true` should always be active
   - Manual rules require explicit invocation with `@ruleName`

### Rule Conflicts
If rules provide conflicting guidance:

1. **Priority order**: Foundation workflow > Development process > Quality gates > Documentation
2. **Always rules** take precedence over auto-attached rules
3. **Manual rules** can override automatic rules when explicitly invoked

### Missing Rules
If you need guidance not covered by existing rules:

1. **Check this document** for complete workflow details
2. **Use manual rule invocation** for specific guidance
3. **Request new rules** by creating an issue with the enhancement label

### Rule Updates
When updating rules:

1. **Test rule changes** with different file types
2. **Verify rule activation** works as expected
3. **Update documentation** to reflect changes
4. **Commit rule changes** with proper documentation updates
