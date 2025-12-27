This note sketches simple patterns for chaining large language model prompts so that intermediate outputs stay structured and debuggable.

## Core Pattern: Classify -> Extract -> Synthesize

1. **Classify:** Decide what branch or template should fire. Keep temperature low.
2. **Extract:** Pull structured facts with a constrained format such as JSON.
3. **Synthesize:** Generate the final narrative while validating required fields.

## Guardrails

- Require schema-compliant JSON or Markdown tables at hand-off boundaries.
- Constrain temperature and top-p for extractive steps; loosen only at synthesis.
- Add lightweight evals for completeness, format, and internal contradiction.

## Operational Tips

- Log intermediate decisions to a shared store so failure exemplars are easy to inspect.
- Track per-step quality metrics (accuracy, coverage, latency) to guide iteration.
- Promote successful chains to reusable prompt templates or functions.
