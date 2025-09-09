# Agent Workflow Framework

## Overview
This document outlines how Cursor agents can systematically work through the retirement planner tickets in the correct order, creating pull requests for each implementation.

## Ticket Sequencing Strategy

### Phase 1: Foundation (Must be completed first)
**Order 1-5**: Core infrastructure that everything else depends on
- EP-1-T1: Create repo scaffold (Flask app factory, blueprints, config)
- EP-1-T2: Add tooling (pytest, coverage, black, isort, flake8/ruff, mypy)
- EP-1-T3: Config management via `.env` and Pydantic Settings
- EP-1-T4: Makefile & pre-commit hooks
- EP-1-T5: Dockerfile & dev docker-compose

### Phase 2: Core Data & Engine (Parallel after foundation)
**Order 6-12**: Data models and basic engine components
- EP-2-T1: Define JSON schema v0.1 for Scenario
- EP-2-T2: SQLAlchemy models (User, Scenario, Run, LedgerRow)
- EP-2-T3: Scenario versioning & immutable base+diff
- EP-2-T4: Storage backend
- EP-3-T1: Time grid & unit system (annual; real vs nominal)
- EP-3-T2: Mortgage amortization module
- EP-3-T3: Baseline expenses & lumpy events

### Phase 3: Simulation Engines
**Order 13-19**: Core simulation logic
- EP-3-T4: Account balance evolution (taxable/trad/Roth)
- EP-3-T5: Social Security stub (fixed input benefit)
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

## Agent Workflow Process

### 1. Agent Initialization
When a fresh agent starts, it should:

```bash
# 1. Check current repository state
git status
git log --oneline -5

# 2. Find the next ticket to work on
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
   - Follow the acceptance criteria exactly
   - Write tests as specified
   - Ensure all tests pass
   - Run linting and type checking

3. **Create pull request**:
   ```bash
   git push origin feature/EP-X-TY-short-description
   gh pr create --title "EP-X-TY: Ticket Title" --body "Implements EP-X-TY

   **Acceptance Criteria**: [Copy from ticket]
   **Tests**: [Copy from ticket]
   **Dependencies**: [Copy from ticket]

   Closes #[issue-number]"
   ```

4. **Wait for review**: Agent should not proceed until PR is merged

### 4. Post-Merge Process
After PR is merged:

1. **Update local repository**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Close the ticket**:
   ```bash
   gh issue close [issue-number] --comment "Implemented in PR #[pr-number]"
   ```

3. **Find next ticket**:
   ```bash
   gh issue list --state open --json number,title,labels | jq '.[0]'
   ```

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

To find the next ticket, run these commands in order:
```bash
# Check Phase 1 (foundation) first
gh issue list --label "phase:1" --state open

# If Phase 1 is empty, check Phase 2
gh issue list --label "phase:2" --state open

# Continue through phases until you find open tickets
gh issue list --label "phase:3" --state open
gh issue list --label "phase:4" --state open
gh issue list --label "phase:5" --state open
```

Once you find a phase with open tickets:
1. Pick the first ticket in the list (they should be in order)
2. Assign it to yourself: `gh issue edit [ISSUE_NUMBER] --assignee @me`
3. View the ticket details: `gh issue view [ISSUE_NUMBER]`

To implement a ticket:
- Create a feature branch: `git checkout -b feature/EP-X-TY-description`
- Follow the acceptance criteria exactly
- Write the specified tests
- Ensure all tests pass
- Create a pull request

Do not proceed to the next ticket until the current PR is merged.
```

## Quality Gates

Each implementation must pass:
- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Coverage ≥80% (`make coverage`)
- [ ] Acceptance criteria met
- [ ] Dependencies satisfied

## Error Handling

If a ticket cannot be implemented:
1. Add a comment explaining the issue
2. Tag the ticket with `blocked` label
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

This framework ensures systematic, dependency-aware development with proper quality gates and human oversight.
