"""Streamlit dashboard for the reproducible P7 benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="P7 benchmark dashboard",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> dict[str, Any]:
    """Load one benchmark result, keeping file IO out of widget reruns."""
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


def discover_results() -> list[Path]:
    benchmark_dir = ROOT / "evaluation" / "benchmarks"
    return sorted(benchmark_dir.glob("benchmark_*.json")) if benchmark_dir.exists() else []


def ratio(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def number(value: Any, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def available_strategies(result: dict[str, Any]) -> list[str]:
    return [name for name, data in result.get("strategies", {}).items() if data.get("status") == "completed"]


def strategy_label(name: str) -> str:
    return {"ocr_kie": "OCR + KIE", "vlm": "VLM", "hybrid": "Hybrid"}.get(name, name)


def values_match(expected: Any, predicted: Any) -> bool:
    if expected in (None, "") or predicted in (None, ""):
        return expected in (None, "") and predicted in (None, "")
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return abs(float(expected) - float(predicted)) <= max(0.011, abs(float(expected)) * 0.001)
    return " ".join(str(expected).casefold().split()) == " ".join(str(predicted).casefold().split())


def metric_frame(result: dict[str, Any], selected: list[str]) -> pd.DataFrame:
    rows = []
    for name in selected:
        data = result["strategies"][name]
        metrics = data.get("metrics", {})
        rows.append(
            {
                "strategy": strategy_label(name),
                "field_accuracy": metrics.get("field_accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "exact_match": metrics.get("exact_match"),
                "validation_failure_rate": metrics.get("validation_failure_rate"),
            }
        )
    return pd.DataFrame(rows)


def category_frame(result: dict[str, Any], selected: list[str]) -> pd.DataFrame:
    rows = []
    for name in selected:
        for category, metrics in result["strategies"][name].get("category_metrics", {}).items():
            rows.append(
                {
                    "category": category,
                    "strategy": strategy_label(name),
                    "field_accuracy": metrics.get("field_accuracy"),
                    "f1": metrics.get("f1"),
                    "samples": metrics.get("samples", 0),
                }
            )
    return pd.DataFrame(rows)


def field_frame(result: dict[str, Any], strategy: str) -> pd.DataFrame:
    metrics = result["strategies"][strategy].get("metrics", {})
    rows = []
    for field, values in metrics.get("per_field", {}).items():
        rows.append(
            {
                "field": field,
                "accuracy": values.get("accuracy"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1"),
                "correct": values.get("correct", 0),
                "total": values.get("total", 0),
            }
        )
    return pd.DataFrame(rows)


def record_frame(result: dict[str, Any], strategy: str, category: str) -> pd.DataFrame:
    records = result["strategies"][strategy].get("records", [])
    rows = []
    for record in records:
        if category != "Tất cả" and record.get("category") != category:
            continue
        metrics = record.get("timings", {})
        rows.append(
            {
                "fixture": record.get("fixture"),
                "category": record.get("category"),
                "validation": record.get("validation_status"),
                "field_accuracy": sum(
                    values_match(record.get("truth", {}).get(field), record.get("prediction", {}).get(field))
                    for field in result["strategies"][strategy].get("metrics", {}).get("per_field", {})
                ),
                "cold_ms": metrics.get("cold_ms"),
                "warm_ms": metrics.get("warm_ms_median"),
                "issues": len(record.get("validation_issues", [])),
            }
        )
    return pd.DataFrame(rows)


results = discover_results()
with st.sidebar:
    st.header("Bộ lọc benchmark")
    if not results:
        st.warning("Chưa có kết quả benchmark.", icon=":material/info:")
        st.stop()
    chosen_path = st.selectbox(
        "Kết quả",
        results,
        index=max(0, len(results) - 1),
        format_func=lambda path: path.name,
        key="benchmark_result",
    )
    result = load_json(str(chosen_path))
    completed = available_strategies(result)
    selected = st.multiselect(
        "Chiến lược đã hoàn tất",
        completed,
        default=completed,
        format_func=strategy_label,
        key="selected_strategies",
    )
    st.divider()
    st.caption(f"Dataset: {result.get('dataset', '—')}")
    st.caption(f"Split: {result.get('split', '—')}")
    st.caption("Số liệu VLM/Hybrid chỉ hiển thị khi benchmark live có ảnh gốc.")

st.title("P7 · Benchmark dashboard")
st.caption("So sánh OCR + KIE, VLM và Hybrid với provenance rõ ràng; không trình bày replay token như một lần OCR mới.")

if not selected:
    st.info("Chọn ít nhất một chiến lược đã hoàn tất ở thanh bên.", icon=":material/tune:")
    st.stop()

metrics_df = metric_frame(result, selected)
first_strategy = selected[0]
first_data = result["strategies"][first_strategy]
first_metrics = first_data.get("metrics", {})

with st.container(horizontal=True):
    st.metric("Mẫu benchmark", number(first_metrics.get("samples"), 0), border=True)
    st.metric("Field accuracy", ratio(first_metrics.get("field_accuracy")), border=True)
    st.metric("Entity F1", ratio(first_metrics.get("f1")), border=True)
    st.metric("Exact match", ratio(first_metrics.get("exact_match")), border=True)
    st.metric("Validation lỗi", ratio(first_metrics.get("validation_failure_rate")), border=True)

st.subheader("So sánh chiến lược")
with st.container(border=True):
    st.dataframe(
        metrics_df.set_index("strategy").style.format(
            {column: "{:.1%}" for column in metrics_df.columns if column != "strategy"}
        ),
        hide_index=False,
        alt="Bảng so sánh các chỉ số benchmark theo chiến lược",
    )

chart_col, category_col = st.columns(2)
with chart_col:
    st.markdown("**Field accuracy và Entity F1**")
    chart_df = metrics_df.set_index("strategy")[["field_accuracy", "f1"]]
    st.bar_chart(chart_df, alt="Field accuracy và Entity F1 theo chiến lược")
with category_col:
    st.markdown("**Theo nhóm hóa đơn**")
    categories = category_frame(result, selected)
    if categories.empty:
        st.info("Kết quả không có category metrics.")
    else:
        st.bar_chart(
            categories.pivot(index="category", columns="strategy", values="f1"),
            alt="Entity F1 theo nhóm hóa đơn và chiến lược",
        )

st.subheader("Chi tiết theo trường")
field_strategy = st.selectbox(
    "Chiến lược cần phân tích",
    selected,
    format_func=strategy_label,
    key="field_strategy",
)
fields = field_frame(result, field_strategy)
field_col, timing_col = st.columns([1.35, 1])
with field_col:
    st.dataframe(
        fields.set_index("field").style.format(
            {column: "{:.1%}" for column in ("accuracy", "precision", "recall", "f1")}
        ),
        hide_index=False,
        column_config={"correct": st.column_config.NumberColumn("Đúng"), "total": st.column_config.NumberColumn("Tổng")},
        alt="Độ chính xác theo từng trường hóa đơn",
    )
with timing_col:
    st.markdown("**Độ trễ và trạng thái tài nguyên**")
    latency = result["strategies"][field_strategy].get("latency")
    if latency:
        st.table(
            {
                "Cold median": f"{number(latency.get('cold_median_ms'))} ms",
                "Warm median": f"{number(latency.get('warm_median_ms'))} ms",
                "Resource": latency.get("resource_status", "—"),
            },
            alt="Độ trễ và trạng thái tài nguyên của strategy",
        )
    else:
        st.info("Chiến lược này chưa có số đo latency.")

st.subheader("Theo fixture")
record_category = st.selectbox(
    "Nhóm fixture",
    ["Tất cả"] + sorted({record.get("category", "unknown") for record in first_data.get("records", [])}),
    key="record_category",
)
records = record_frame(result, field_strategy, record_category)
if records.empty:
    st.info("Không có fixture phù hợp với bộ lọc.")
else:
    st.dataframe(
        records.set_index("fixture"),
        column_config={
            "field_accuracy": st.column_config.NumberColumn("Số trường khớp"),
            "cold_ms": st.column_config.NumberColumn("Cold (ms)", format="%.1f"),
            "warm_ms": st.column_config.NumberColumn("Warm median (ms)", format="%.1f"),
        },
        hide_index=False,
        alt="Kết quả benchmark theo từng fixture",
    )

with st.expander("Provenance và giới hạn benchmark", expanded=False):
    st.markdown(f"**Leakage policy:** {result.get('leakage_policy', '—')}")
    baseline = result.get("historical_baseline", {})
    st.markdown(
        f"**Historical baseline:** {baseline.get('name', '—')} · "
        f"{baseline.get('correct', '—')}/{baseline.get('total', '—')} "
        f"({ratio(baseline.get('accuracy'))})"
    )
    for name, data in result.get("strategies", {}).items():
        if data.get("status") == "unavailable":
            st.warning(f"{strategy_label(name)}: {data.get('reason', 'Không có lý do.')}", icon=":material/block:")

st.download_button(
    "Tải JSON benchmark",
    data=chosen_path.read_bytes(),
    file_name=chosen_path.name,
    mime="application/json",
    icon=":material/download:",
    width="content",
)
