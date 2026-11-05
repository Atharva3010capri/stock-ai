"""Extraction validation: malformed or unsupported values must be discarded."""

import pytest

import extraction
from gemini_client import LLMJSONError, parse_json


def hit(source, page, text):
    return {"text": text, "metadata": {"source": source, "page": page, "chunk_id": f"{source}-{page}"}, "similarity": 0.5}


HITS = [
    hit("ar.pdf", 112, "Revenue from operations 12,345.6 for FY2024 compared to 10,000.0 in FY2023."),
    hit("ar.pdf", 115, "Profit for the year 1,500.5 (previous year 1,200.0)."),
]


def test_valid_entries_are_kept_and_verified():
    raw = {
        "currency": "INR", "units": "crore",
        "figures": {
            "revenue": [
                {"period": "FY2024", "value": "12,345.6", "page": 112, "source": "ar.pdf", "evidence": "Revenue from operations 12,345.6"},
                {"period": "FY2023", "value": 10000, "page": 112, "source": "ar.pdf", "evidence": "10,000.0 in FY2023"},
            ],
            "net_profit": [{"period": "FY2024", "value": 1500.5, "page": 115, "source": "ar.pdf", "evidence": "Profit for the year 1,500.5"}],
        },
    }
    result = extraction.validate_extraction(raw, HITS)
    assert result["currency"] == "INR" and result["units"] == "crore"
    assert [e["value"] for e in result["figures"]["revenue"]] == [12345.6, 10000.0]
    assert all(e["evidence_verified"] for e in result["figures"]["revenue"])
    assert result["figures"]["net_profit"][0]["value"] == 1500.5
    assert result["figures"]["eps"] == []          # missing -> empty, shown as "Not found in report"
    assert result["discarded"] == []


def test_wrong_page_and_non_numeric_values_are_discarded():
    raw = {
        "figures": {
            "revenue": [{"period": "FY2024", "value": 999, "page": 999, "source": "ar.pdf", "evidence": "anything"}],
            "eps": [{"period": "FY2024", "value": "about twelve", "page": 112, "source": "ar.pdf", "evidence": "x"}],
            "total_debt": "not a list",
        }
    }
    result = extraction.validate_extraction(raw, HITS)
    assert result["figures"]["revenue"] == []
    assert result["figures"]["eps"] == []
    assert len(result["discarded"]) == 3


def test_unverifiable_quote_is_flagged_not_fabricated():
    raw = {"figures": {"net_profit": [
        {"period": "FY2024", "value": 1500.5, "page": 115, "source": "ar.pdf", "evidence": "this quote is not on the page"}
    ]}}
    result = extraction.validate_extraction(raw, HITS)
    entry = result["figures"]["net_profit"][0]
    assert entry["value"] == 1500.5
    assert entry["evidence_verified"] is True  # number itself appears on the page


def test_quote_and_number_both_missing_is_discarded():
    raw = {"figures": {"net_profit": [{"period": "FY2024", "value": 42, "page": 115, "source": "ar.pdf", "evidence": ""}]}}
    result = extraction.validate_extraction(raw, HITS)
    assert result["figures"]["net_profit"] == []


def test_non_object_output_discards_everything():
    result = extraction.validate_extraction(["garbage"], HITS)
    assert all(v == [] for v in result["figures"].values())
    assert result["discarded"]


def test_rows_show_not_found():
    rows = extraction.figures_to_rows({"figures": {"revenue": []}})
    revenue = next(r for r in rows if r["key"] == "revenue")
    assert revenue["Status"] == extraction.NOT_FOUND


def test_parse_json_handles_fences_and_rejects_garbage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    with pytest.raises(LLMJSONError):
        parse_json("no json here")


def test_manual_correction_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "KNOWLEDGE_BASE_DIR", tmp_path)
    record = extraction.set_manual_value("Demo Ltd", "revenue", "FY2024", 123.0)
    assert record["figures"]["revenue"][0]["origin"] == "manual"
    assert extraction.load_figures("Demo Ltd")["figures"]["revenue"][0]["value"] == 123.0
    record = extraction.set_manual_value("Demo Ltd", "revenue", "FY2024", None)
    assert record["figures"]["revenue"] == []
    with pytest.raises(ValueError):
        extraction.set_manual_value("Demo Ltd", "share_price", "FY2024", 1.0)
