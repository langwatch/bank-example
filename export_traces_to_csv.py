"""Convert traces_raw.json to traces_export.csv — one row per span."""

import csv
import json
import re
import sys

# Per-token pricing by model regex
PRICING = [
    (re.compile(r"claude-opus-4-6", re.I), 0.000005, 0.000025),
    (re.compile(r"glm-4\.7", re.I), 0.0000004, 0.0000015),
    (re.compile(r"deepseek-v3\.2", re.I), 0.00000025, 0.00000038),
    (re.compile(r"gpt-oss-120b", re.I), 0.000000039, 0.00000019),
    (re.compile(r"minimax-m2\.1", re.I), 0.00000027, 0.00000095),
    (re.compile(r"gpt-4o", re.I), 0.0000025, 0.00001),
]

COLUMNS = [
    "trace_id",
    "scenario_name",
    "scenario_turn",
    "batch_id",
    "thread_id",
    "trace_has_error",
    "span_id",
    "parent_span_id",
    "span_name",
    "span_type",
    "model",
    "input_type",
    "input_value",
    "output_type",
    "output_value",
    "prompt_tokens",
    "completion_tokens",
    "tokens_estimated",
    "started_at",
    "finished_at",
    "duration_ms",
    "span_has_error",
    "span_error_message",
    "trace_total_time_ms",
    "trace_prompt_tokens",
    "trace_completion_tokens",
    "labels",
    "cost_input",
    "cost_output",
    "cost_total",
]


def get_nested(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def get_cost(model, prompt_tokens, completion_tokens):
    if not model or prompt_tokens is None:
        return None, None, None
    for pattern, inp_price, out_price in PRICING:
        if pattern.search(model):
            ci = (prompt_tokens or 0) * inp_price
            co = (completion_tokens or 0) * out_price
            return ci, co, ci + co
    return None, None, None


def compute_duration(started, finished):
    if started is not None and finished is not None:
        try:
            return finished - started
        except Exception:
            return None
    return None


def process(traces):
    rows = []
    for trace in traces:
        metadata = trace.get("metadata") or {}
        trace_id = trace.get("trace_id") or trace.get("traceId") or trace.get("id", "")
        scenario_name = metadata.get("scenario.name", "")
        scenario_turn = metadata.get("scenario.turn", "")
        batch_id = metadata.get("scenario.batch_id", "")
        thread_id = metadata.get("thread_id", "")
        trace_has_error = get_nested(trace, "error", "has_error", default=False)
        trace_metrics = trace.get("metrics") or {}
        trace_total_time = trace_metrics.get("total_time_ms")
        trace_prompt = trace_metrics.get("prompt_tokens")
        trace_completion = trace_metrics.get("completion_tokens")
        labels = metadata.get("labels", [])

        spans = trace.get("spans") or []
        for span in spans:
            metrics = span.get("metrics") or {}
            prompt_tokens = metrics.get("prompt_tokens")
            completion_tokens = metrics.get("completion_tokens")
            model = span.get("model", "")
            span_type = span.get("type", "")

            ci, co, ct = (None, None, None)
            if span_type == "llm":
                ci, co, ct = get_cost(model, prompt_tokens, completion_tokens)

            timestamps = span.get("timestamps") or {}
            started = timestamps.get("started_at")
            finished = timestamps.get("finished_at")

            inp = span.get("input") or {}
            out = span.get("output") or {}
            input_val = str(inp.get("value", ""))[:500]
            output_val = str(out.get("value", ""))[:500]

            span_error = span.get("error") or {}

            rows.append({
                "trace_id": trace_id,
                "scenario_name": scenario_name,
                "scenario_turn": scenario_turn,
                "batch_id": batch_id,
                "thread_id": thread_id,
                "trace_has_error": trace_has_error,
                "span_id": span.get("span_id", ""),
                "parent_span_id": span.get("parent_id", ""),
                "span_name": span.get("name", ""),
                "span_type": span_type,
                "model": model,
                "input_type": inp.get("type", ""),
                "input_value": input_val,
                "output_type": out.get("type", ""),
                "output_value": output_val,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_estimated": metrics.get("tokens_estimated"),
                "started_at": started,
                "finished_at": finished,
                "duration_ms": compute_duration(started, finished),
                "span_has_error": span_error.get("has_error", False),
                "span_error_message": span_error.get("message", ""),
                "trace_total_time_ms": trace_total_time,
                "trace_prompt_tokens": trace_prompt,
                "trace_completion_tokens": trace_completion,
                "labels": json.dumps(labels) if isinstance(labels, list) else labels,
                "cost_input": ci,
                "cost_output": co,
                "cost_total": ct,
            })
    return rows


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "traces_raw.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "traces_export.csv"

    with open(input_file) as f:
        traces = json.load(f)

    rows = process(traces)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
