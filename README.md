# O3-Mini Web Automation: Reasoning Model-Driven Browser Automation

This repository contains the implementation and experimental results for the bachelor thesis "Application Possibilities of Large Language Models in Internet Browsing and Query Execution Automation" (Latvian University, 2025).

## Project Overview

This project explores the use of OpenAI's o3-mini reasoning model for web automation, comparing its performance against existing multi-component systems like Steward. The key innovation is architectural simplification - replacing multiple specialized language model components with a single reasoning model.

### Key Findings

- **35% success rate** on manual tests (26 queries)
- **25% success rate** on automated tests (102 queries from Mind2Web dataset)
- **71% cost reduction** per step compared to Steward
- **75% increase** in execution time per step
- **Architectural simplification**: Single reasoning model vs. multiple specialized components
