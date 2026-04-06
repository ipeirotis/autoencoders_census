# GitHub Collaboration Workflow

**Purpose:** Establish consistent Git and GitHub practices for maintainer and IliasTriant collaboration on v1.0 production features.

**Audience:** Project maintainer (Aaron), collaborator (IliasTriant)

---

## Branch Strategy

### Branch Types

**feature/** - New features, enhancements
```bash
git checkout -b feature/per-column-contribution-ui
git checkout -b feature/csv-export-with-sanitization
```

**bugfix/** - Non-critical bug fixes
```bash
git checkout -b bugfix/fix-polling-memory-leak
git checkout -b bugfix/handle-expired-jobs
```

**fix/** - Bug fixes (alternative to bugfix/)
```bash
git checkout -b fix/testing-and-ci
git checkout -b fix/unify-data-loading-format
```

**hotfix/** - Critical production fixes (rare in pre-production v1.0)
```bash
git checkout -b hotfix/security-cors-bypass
```

### Naming Conventions

- Use kebab-case (lowercase with hyphens)
- Be descriptive but concise (3-5 words)
- Reference issue number if applicable: `feature/123-user-authentication`

**Examples from Repository:**
- ✅ `fix/testing-and-ci` (PR #10 - IliasTriant)
- ✅ `fix/unify-data-loading-format` (PR #11 - IliasTriant)
- ✅ `paper_additions` (PR #9 - IliasTriant)
- ❌ `feature/stuff` (too vague)
- ❌ `Feature/GCS-Lifecycle` (wrong case)

### Base Branch

All branches created from and merged into `main`.

No long-lived development branches. Short-lived feature branches (1-3 days max).

---

## Commit Message Conventions

### Format

```
type(scope): description

Optional body explaining context and rationale.

Optional footer with breaking changes or issue references.
```

### Types

| Type | Use For | Example |
|------|---------|---------|
| `feat` | New feature or enhancement | `feat(04-01): add CSV export endpoint` |
| `fix` | Bug fix | `fix(worker): handle encoding detection errors` |
| `docs` | Documentation only | `docs(readme): update installation instructions` |
| `test` | Tests only | `test(04-02): add job cancellation integration tests` |
| `chore` | Maintenance, deps, config | `chore(deps): update @google-cloud/storage to 7.18.0` |
| `refactor` | Code restructuring (no behavior change) | `refactor(auth): extract validation to middleware` |
| `style` | Formatting, whitespace | `style(client): fix linter warnings` |
| `perf` | Performance improvement | `perf(csv): use streaming for large file parsing` |

### Scopes

- Phase number: `(01)`, `(02)`, `(03)`, `(04)`
- Plan ID: `(01-01)`, `(03-03A)`, `(04-02)`
- Subsystem: `(worker)`, `(frontend)`, `(api)`, `(tests)`
- Feature: `(auth)`, `(csv-export)`, `(progress)`

Use most specific scope available. If unclear, use feature or subsystem.

### Description

- Start with lowercase (unless proper noun)
- No period at end
- Imperative mood ("add" not "added" or "adds")
- Max 72 characters
- Describe WHAT changed, not HOW

**Good:**
- `feat(04-01): add CSV export endpoint with formula injection protection`
- `fix(worker): prevent duplicate processing with transaction check`

**Bad:**
- `feat(04-01): Added CSV export.` (capitalized, period, vague)
- `fix: fixed bug` (no scope, uninformative)

### Examples from Repository

```bash
# Claude-created commits (conventional format with phase scopes)
feat(03-03A): create useJobPolling hook with TanStack Query
feat(01-01): implement Winston logger with Cloud Logging
fix(03-01): add missing @radix-ui/react-collapsible dependency
docs(04): capture phase context
chore(03-04A): enable noImplicitAny in tsconfig.json
test(01-01): add failing tests for env validation

# IliasTriant commits (descriptive, mixed format)
Fix integration test: remove filtered_list arg not in committed outliers.py
Fix columns_of_interest test to use string names (matches loader API, documents bug 1.8)
Document Chow-Liu tree outlier scoring in CLAUDE.md and TASKS.md
Fix broken test suite, add integration tests, and set up GitHub Actions CI
```

**Note:** Phase 4 work follows conventional commits format consistently. IliasTriant's prior commits use descriptive format without type prefixes - both styles are acceptable, but conventional commits preferred going forward for tooling compatibility.

---

## Pull Request Workflow

### 1. Create Feature Branch

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature
```

### 2. Make Changes

```bash
# Work on feature
# Make atomic commits (one logical change per commit)

git add file1.ts file2.ts
git commit -m "feat(scope): add feature component"

# Continue iterating
git add test.ts
git commit -m "test(scope): add feature tests"
```

### 3. Keep Branch Updated

```bash
# Regularly sync with main (if long-lived branch)
git checkout main
git pull origin main
git checkout feature/my-feature
git rebase main  # Preferred over merge for clean history

# Resolve conflicts if any
# Test after rebase
```

### 4. Push to Remote

```bash
# First push
git push -u origin feature/my-feature

# Subsequent pushes
git push
```

### 5. Create Pull Request

Via GitHub UI or gh CLI:

```bash
# Using gh CLI
gh pr create \
  --title "feat(04-01): add CSV export with formula injection protection" \
  --body "$(cat <<'EOF'
## Summary

Implements OPS-01, OPS-02, OPS-03 requirements for CSV export.

- New GET /jobs/:id/export endpoint streams sanitized CSV
- Formula injection prevention via single-quote prefix (OWASP standard)
- fast-csv library for performance on large datasets

## Changes

- Added `frontend/server/utils/csvSanitization.ts` with sanitizeFormulaInjection function
- Added GET /jobs/:id/export route to jobs.ts
- Installed fast-csv dependency
- 18 tests added (8 sanitization + 10 export integration)

## Test Plan

- [ ] Unit tests pass: `npm test -- --testPathPattern=csvSanitization`
- [ ] Integration tests pass: `npm test -- --testPathPattern=csvExport`
- [ ] Manual test: Downloaded CSV with =SUM(A1:A10) displays as text in Excel
- [ ] Manual test: Rate limiting enforced (10 downloads per hour)

## Dependencies

None - standalone feature.

## Breaking Changes

None.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### PR Title Format

Use descriptive format that clearly states what the PR does:

**Examples from IliasTriant PRs:**
- "Fix test suite, add integration tests, and set up CI" (PR #10)
- "Stabilize core pipeline: fix data loading, configs, and CLI bugs" (PR #11)
- "Paper additions" (PR #9)

**Conventional commit format also acceptable:**
- `feat(04-02): implement job cancellation with GCS cleanup`
- `fix(worker): handle chardet encoding detection errors`

Use whichever format best communicates the PR's purpose to reviewers.

### PR Description Structure

**Required sections:**

1. **Summary** - What problem does this solve? (bullet points or short paragraphs)
2. **Test Plan** - How was this tested? (checklist format)

**Optional but recommended:**
- **Changes** - Bullet list of key changes
- **Dependencies** - Does this depend on other PRs?
- **Breaking Changes** - Any breaking changes? (or "None")
- **Screenshots** - For UI changes
- **Performance Impact** - For perf-critical changes

**Example from IliasTriant PR #10:**
```markdown
## Summary
- Fix broken test_loader.py (stale API references) and test_loss.py (nonexistent import)
- Add end-to-end integration tests using synthetic data
- Add GitHub Actions CI workflow to run tests on every push/PR
- Fix loss.py get_config() to include percentile parameter

## Test plan
- [ ] Verify python -m pytest tests/ -v passes locally
- [ ] Verify GitHub Actions CI runs successfully on this PR
- [ ] Confirm test_integration_pipeline.py tests run without requiring external data files
```

**Tip:** Use checkboxes `- [ ]` for test plan items.

### 6. Request Review

```bash
# Using gh CLI
gh pr view  # Opens PR in browser

# Request review from collaborator
gh pr edit --add-reviewer IliasTriant  # or maintainer Aaron
```

### 7. Address Review Feedback

```bash
# Make changes based on feedback
git add updated-files
git commit -m "fix(scope): address review feedback - clarify error message"

# Push updates
git push  # Automatically updates PR
```

**Communication:**
- Respond to all review comments (even if just "Done ✓")
- Ask questions if feedback unclear
- Push commits addressing feedback promptly (within 24 hours if possible)

### 8. Merge

Once approved:

```bash
# Squash merge preferred (clean main history)
gh pr merge --squash --delete-branch

# Or via GitHub UI: "Squash and merge" button
```

**Merge commit message:** GitHub auto-generates from PR title + description.

---

## Code Review Process

### For Authors (Creating PR)

**Before requesting review:**
- [ ] All tests pass locally
- [ ] Code follows project conventions (TypeScript strict mode, ESLint rules)
- [ ] No console.log/console.error (use logger)
- [ ] No hardcoded values (use env vars)
- [ ] PR description complete with test plan
- [ ] Self-review: read your own diff in GitHub UI

**During review:**
- Respond within 24 hours to review comments
- Be open to feedback and alternative approaches
- Ask clarifying questions if reviewer's intent unclear
- Mark conversations as resolved after addressing

**After approval:**
- Merge promptly (don't leave approved PRs open)
- Delete branch after merge (GitHub auto-deletes if configured)

### For Reviewers

**Review checklist:**
- [ ] PR title/description clear and complete
- [ ] Changes match described intent
- [ ] Tests added for new functionality
- [ ] Error handling appropriate
- [ ] No obvious security issues (SQL injection, XSS, exposed secrets)
- [ ] Performance impact acceptable (no obvious N+1 queries, large loops)
- [ ] Code readable and maintainable

**Review tone:**
- Constructive and specific ("Consider using X instead of Y because...")
- Ask questions rather than demand ("Why did you choose approach X?")
- Approve if good enough, not perfect (velocity matters)
- Use "nit:" prefix for minor/optional suggestions

**Review response time:**
- Target: Within 24 hours on weekdays
- Urgent/blocking PRs: Tag with "urgent" label, notify directly

---

## IliasTriant Collaboration Patterns

**Historical Contributions:**

IliasTriant has extensive history in this repository (40+ commits, 5 PRs). Analysis of recent PRs reveals:

### PR Patterns (from PRs #9, #10, #11)

**PR Size:**
- PR #10: +374/-91 lines (medium, well-scoped)
- Focused on single logical changes (testing infrastructure, bug fixes)

**Commit Style:**
- Descriptive commit messages focusing on WHAT changed
- Examples: "Fix broken test suite, add integration tests, and set up GitHub Actions CI"
- Mix of atomic commits and aggregated changes

**PR Structure:**
- Clear summary with bullet points
- Test plan with checkboxes
- Links to related issues/documentation

**Communication:**
- Responsive to feedback
- Documents changes in TASKS.md and CLAUDE.md
- Updates documentation alongside code changes

### Recommended Patterns Going Forward

1. **PR frequency:** 2-3 PRs per week during active development
2. **PR size:** Small-to-medium (100-400 lines changed preferred, 500 max)
3. **Commit granularity:** Continue atomic commit style where practical
4. **Commit format:** Adopt conventional commits for Phase 4 work (tooling compatibility)
5. **Communication:** GitHub comments for technical discussion

### Reference Examples

**Well-structured PR:**
- PR #10 "Fix test suite, add integration tests, and set up CI"
  - Clear summary of multiple related changes
  - Comprehensive test plan
  - Self-contained (no external dependencies)
  - Good size (374 additions, 91 deletions)

**Good commit messages from IliasTriant:**
- "Fix columns_of_interest test to use string names (matches loader API, documents bug 1.8)"
- "Document Chow-Liu tree outlier scoring in CLAUDE.md and TASKS.md"
- Both are descriptive, explain context, and reference documentation

---

## Common Scenarios

### Scenario: Conflicts with Main

```bash
# Update main
git checkout main
git pull origin main

# Rebase feature branch
git checkout feature/my-feature
git rebase main

# Resolve conflicts
# Edit conflicting files, resolve markers
git add resolved-files
git rebase --continue

# Force push (rebase rewrites history)
git push --force-with-lease  # Safer than --force
```

### Scenario: Need to Split Large PR

```bash
# Create separate branch for Part 1
git checkout feature/original-branch
git checkout -b feature/part-1

# Remove Part 2 commits (interactive rebase)
git rebase -i main  # Mark Part 2 commits as "drop"

# Push Part 1
git push -u origin feature/part-1

# Create PR for Part 1

# Create branch for Part 2
git checkout feature/original-branch
git checkout -b feature/part-2

# Remove Part 1 commits
git rebase -i main  # Mark Part 1 commits as "drop"

# Push Part 2
git push -u origin feature/part-2

# Create PR for Part 2 (mark as depends on Part 1)
```

### Scenario: Accidentally Committed to Main

```bash
# Create feature branch from main
git branch feature/accidental-commit

# Reset main to remote
git checkout main
git reset --hard origin/main

# Push feature branch
git push -u origin feature/accidental-commit

# Create PR from feature branch
```

### Scenario: Need to Update PR Title/Description

```bash
# Using gh CLI
gh pr edit {PR_NUMBER} --title "new-title"
gh pr edit {PR_NUMBER} --body "new description"

# Or via GitHub UI: Edit button on PR page
```

### Scenario: CI Fails After Push

```bash
# Check CI logs
gh run view  # or visit GitHub Actions tab

# Fix the issue locally
git add fixed-files
git commit -m "fix(ci): address linting errors"
git push  # CI runs again automatically

# If tests pass now, request re-review
```

---

## Best Practices

### Do

✅ Write descriptive commit messages (reference IliasTriant examples above)
✅ Keep PRs small and focused (< 500 lines)
✅ Test locally before pushing
✅ Respond to review feedback promptly
✅ Ask questions if unsure
✅ Update PR description if scope changes
✅ Update CLAUDE.md and documentation when adding features
✅ Add integration tests for user-facing features

### Don't

❌ Force push without `--force-with-lease` (risk data loss)
❌ Commit directly to main (always use PRs)
❌ Leave unresolved review comments
❌ Merge your own PR without approval (except trivial docs)
❌ Include unrelated changes in PR
❌ Commit secrets or API keys
❌ Skip the test plan in PR description

---

## Tools

**Required:**
- Git 2.30+
- GitHub CLI (`gh`) recommended: `brew install gh`

**Optional but helpful:**
- Pre-commit hooks: `npm install husky -D` (lint, format, test)
- Commitlint: Enforce commit message format
- GitHub Desktop: GUI alternative to CLI

---

## Phase 4 Specific Guidelines

### Commit Format for Phase 4 Plans

All Phase 4 work uses conventional commits with plan-specific scopes:

```bash
# Format: type(plan-id): description
feat(04-01): add CSV export endpoint
test(04-02): add job cancellation tests
fix(04-03): handle GCS lifecycle edge cases
docs(04-07): add GitHub collaboration workflow
```

### Testing Requirements

Every Phase 4 plan includes automated tests:
- Unit tests for business logic
- Integration tests for API endpoints
- Frontend tests for React components

Test stubs created in wave 0 (plan 04-00), implemented per-plan.

### Security Focus

Phase 4 follows Phase 1 security patterns:
- Input validation at multiple layers
- Rate limiting on sensitive endpoints
- No hardcoded secrets (use env vars)
- Structured logging (Winston, not console.log)

---

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [Repository commit history](https://github.com/ipeirotis/autoencoders_census/commits/main) - Examples of project style
- [IliasTriant PR #10](https://github.com/ipeirotis/autoencoders_census/pull/10) - Well-structured PR example

---

**Questions?** Open an issue or discuss in pull request comments.

**Last updated:** 2026-04-06
**Maintainer:** Aaron (with Claude assistance)
**Active Collaborator:** IliasTriant
**Repository:** [ipeirotis/autoencoders_census](https://github.com/ipeirotis/autoencoders_census)
