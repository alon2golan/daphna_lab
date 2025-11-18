# Daphna Lab - Active Repository

This is the active repository of the **Daphna Weinshell Lab**. It contains implementations and experiments related to active learning methods, including TypiClust and other state-of-the-art active learning algorithms.

## Quick Start

For detailed setup and onboarding instructions, see [Onboarding Guide](onboarding/ONBOARDING.md).

## Running TypiClust

TypiClust is a novel active learning method implemented in this repository. For detailed instructions on how to run TypiClust experiments, see:

- **[USAGE.md](USAGE.md)** - Complete usage guide with examples for:
  - TypiClust (K-Means variant)
  - TypiClust (SCAN variant)
  - ProbCover
  - DCoM
  - And other active learning methods

- **[ACTIVE_LEARNING.md](ACTIVE_LEARNING.md)** - Theoretical background and detailed explanation of active learning approaches

## Repository Structure

- `deep-al/` - Deep active learning implementations and tools
- `scan/` - Self-supervised clustering code
- `onboarding/` - Lab onboarding materials and scripts
- `stats_parquets/` - Experiment results and statistics

## Key Features

- Multiple active learning strategies (TypiClust, ProbCover, DCoM, etc.)
- Representation learning via SimCLR and SCAN
- GPU support for efficient training
- Comprehensive experiment tracking

## Support

For questions or issues, refer to the lab onboarding guide or contact the lab members.
