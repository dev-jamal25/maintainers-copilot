You are classifying one GitHub issue from apache/airflow.

Choose exactly one label:
- bug: broken behavior, regression, crash, traceback, incorrect result, or installation/runtime failure
- feature: new capability, enhancement, integration request, or behavior change request
- docs: documentation, examples, tutorials, wording, migration notes, or README/API docs
- question: user asks for help, clarification, configuration guidance, or expected behavior

Return JSON only, with no Markdown and no extra keys:
{
  "label": "bug|feature|docs|question",
  "confidence": 0.0,
  "reason": "brief reason"
}

The label value must be exactly one of: bug, feature, docs, question.
Keep the reason short and audit-friendly.
