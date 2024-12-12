# MyoSuite Project

This repository is part of a setup involving two repositories: `myosuite_project` and `myosuite_fork`. Below is a description of the project structure, including how the branches are organized and their roles.

## Repository Structure

```text
myosuite_project/        # Main working repository
├── main                 # Primary development branch for your work
└── fork/main            # Synchronized copy of `myosuite_fork`'s main branch

myosuite_fork/           # Forked repository from the origin
└── main                 # Mirrors the original repository and is kept in sync
```
# Merge Rules

This document outlines the basic rules for merging branches in the project.

## Branches

- **main**: The primary development branch where all new work is done.
- **fork/main**: A synchronized copy of the `myosuite_fork/main` branch.

## Merge Guidelines

1. **Do not modify `fork/main` directly**:
   - `fork/main` is only updated by syncing with the `myosuite_fork` repository.
   - Any manual changes to this branch will be overwritten by the next sync.

2. **Merging `fork/main` into `main`**:
   - Periodically, merge the `fork/main` branch into `main` to stay up to date with changes from the upstream repository.
   - Use the following command to merge:
     ```bash
     git checkout main
     git merge fork/main
     ```

3. **Development on `main`**:
   - All custom development should occur on the `main` branch.
   - Ensure `main` is regularly updated by merging changes from `fork/main`.

By following these rules, the project will remain synchronized with the upstream repository while allowing for independent development.
