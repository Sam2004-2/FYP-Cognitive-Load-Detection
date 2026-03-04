"""Tests for study upload/admin report endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi.testclient import TestClient
import pytest

from src.cle import server


def make_session_record(
    participant_id: str,
    record_id: str,
    started_at_iso: str,
    due_at_iso: str,
    pending_delayed: bool = True,
) -> Dict:
    return {
        "recordVersion": 1,
        "recordId": record_id,
        "participantId": participant_id,
        "sessionNumber": 1,
        "assignment": {
            "participantId": participant_id,
            "participantIdNormalized": participant_id.lower(),
            "hashValue": 1,
            "hashParity": "odd",
            "conditionOrder": ["adaptive", "baseline"],
            "formOrder": ["A", "B"],
            "sessionNumber": 1,
            "condition": "adaptive",
            "form": "A",
            "delayedDueAtIso": due_at_iso,
        },
        "plan": {
            "baselineSeconds": 60,
            "easyItemCount": 12,
            "easyExposureSeconds": 5.5,
            "hardItemCount": 18,
            "hardExposureSeconds": 3.5,
            "hardInterferenceEnabled": True,
            "recognitionChoices": 4,
            "microBreakSeconds": 60,
            "maxMicroBreaksPerSession": 2,
            "adaptationCooldownSeconds": 120,
            "decisionWindowSeconds": 5,
            "smoothingWindows": 3,
            "overloadThreshold": 0.7,
        },
        "startedAtIso": started_at_iso,
        "completedAtIso": started_at_iso,
        "session2StartedEarlyOverride": False,
        "condition": "adaptive",
        "form": "A",
        "totalSessionSeconds": 100,
        "activeTaskSeconds": 80,
        "breakSeconds": 20,
        "cliSamples": [],
        "interventions": [],
        "trials": [
            {
                "trialId": "learn-easy-1",
                "timestampMs": 1000,
                "sessionTimeS": 1,
                "phase": "learning_easy",
                "kind": "learning",
                "difficulty": "easy",
                "blockIndex": 1,
                "itemId": "easy-item-1",
                "cue": "Apple",
                "target": "Red",
                "correct": True,
                "reactionTimeMs": 1000,
                "condition": "adaptive",
                "form": "A",
            },
            {
                "trialId": "learn-hard-1",
                "timestampMs": 2000,
                "sessionTimeS": 2,
                "phase": "learning_hard",
                "kind": "learning",
                "difficulty": "hard",
                "blockIndex": 2,
                "itemId": "hard-item-1",
                "cue": "Gamma",
                "target": "Delta",
                "correct": True,
                "reactionTimeMs": 1200,
                "condition": "adaptive",
                "form": "A",
            },
        ],
        "blockSummaries": [],
        "pendingDelayedTest": pending_delayed,
        "delayedDueAtIso": due_at_iso,
    }


def make_delayed_record(participant_id: str, record_id: str, linked_session_record_id: str) -> Dict:
    return {
        "recordVersion": 1,
        "recordId": record_id,
        "linkedSessionRecordId": linked_session_record_id,
        "participantId": participant_id,
        "sessionNumber": 1,
        "condition": "adaptive",
        "form": "A",
        "dueAtIso": "2026-02-22T12:00:00Z",
        "completedAtIso": "2026-02-22T12:15:00Z",
        "trials": [],
        "recognitionAccuracy": 0.7,
        "recognitionMeanRtMs": 1234,
        "cuedRecallAccuracy": 0.6,
        "cuedRecallMeanRtMs": 1400,
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLE_REPORTS_DIR", str(tmp_path))
    monkeypatch.setenv("CLE_ADMIN_TOKEN", "admin-token")
    monkeypatch.setenv("CLE_ALLOWED_ORIGINS", "*")

    original_startup = list(server.app.router.on_startup)
    server.app.router.on_startup.clear()

    with TestClient(server.app) as test_client:
        yield test_client

    server.app.router.on_startup[:] = original_startup


def test_create_participant_id_unique(client: TestClient):
    first = client.post("/study/participants")
    second = client.post("/study/participants")

    assert first.status_code == 200
    assert second.status_code == 200

    first_id = first.json()["participant_id"]
    second_id = second.json()["participant_id"]

    assert first_id.startswith("P-")
    assert second_id.startswith("P-")
    assert first_id != second_id


def test_upload_validation(client: TestClient):
    session_missing_fields = client.post("/study/session-records", json={"record": {"participantId": "P-X"}})
    delayed_missing_fields = client.post("/study/delayed-records", json={"record": {"participantId": "P-X"}})

    assert session_missing_fields.status_code == 400
    assert delayed_missing_fields.status_code == 400


def test_admin_requires_bearer_token(client: TestClient):
    unauthorized = client.get("/admin/reports/index")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/admin/reports/index",
        headers={"Authorization": "Bearer admin-token"},
    )
    assert authorized.status_code == 200

    monitoring_unauthorized = client.get("/admin/monitoring/summary")
    assert monitoring_unauthorized.status_code == 401

    monitoring_authorized = client.get(
        "/admin/monitoring/summary",
        headers={"Authorization": "Bearer admin-token"},
    )
    assert monitoring_authorized.status_code == 200


def test_export_filtering(client: TestClient):
    session_a = make_session_record(
        participant_id="P-A",
        record_id="record-a",
        started_at_iso="2026-02-01T10:00:00Z",
        due_at_iso="2026-02-02T10:00:00Z",
    )
    session_b = make_session_record(
        participant_id="P-B",
        record_id="record-b",
        started_at_iso="2026-02-20T10:00:00Z",
        due_at_iso="2026-02-21T10:00:00Z",
    )

    assert client.post("/study/session-records", json={"record": session_a}).status_code == 200
    assert client.post("/study/session-records", json={"record": session_b}).status_code == 200

    participant_filtered = client.get(
        "/admin/reports/export?format=json&participant_id=P-A",
        headers={"Authorization": "Bearer admin-token"},
    )
    assert participant_filtered.status_code == 200
    participant_payload = participant_filtered.json()
    assert participant_payload["count"] == 1
    assert participant_payload["records"][0]["participant_id"] == "P-A"

    date_filtered = client.get(
        "/admin/reports/export?format=json&from_iso=2026-02-15T00:00:00Z&to_iso=2026-02-28T00:00:00Z",
        headers={"Authorization": "Bearer admin-token"},
    )
    assert date_filtered.status_code == 200
    date_payload = date_filtered.json()
    assert date_payload["count"] == 1
    assert date_payload["records"][0]["participant_id"] == "P-B"


def test_pending_delayed_resolves_after_delayed_upload(client: TestClient):
    participant_id = "P-PENDING"
    session_record = make_session_record(
        participant_id=participant_id,
        record_id="session-1",
        started_at_iso="2026-02-22T10:00:00Z",
        due_at_iso="2026-02-23T10:00:00Z",
        pending_delayed=True,
    )

    upload_session = client.post("/study/session-records", json={"record": session_record})
    assert upload_session.status_code == 200

    pending_before = client.get(f"/study/pending-delayed/{participant_id}")
    assert pending_before.status_code == 200
    pending_payload = pending_before.json()
    assert len(pending_payload["pending"]) == 1
    assert len(pending_payload["pending"][0]["easy_items"]) == 1
    assert len(pending_payload["pending"][0]["hard_items"]) == 1

    delayed_record = make_delayed_record(
        participant_id=participant_id,
        record_id="delayed-1",
        linked_session_record_id="session-1",
    )
    upload_delayed = client.post("/study/delayed-records", json={"record": delayed_record})
    assert upload_delayed.status_code == 200

    pending_after = client.get(f"/study/pending-delayed/{participant_id}")
    assert pending_after.status_code == 200
    assert pending_after.json()["pending"] == []


def test_monitoring_summary_includes_activity_metrics(client: TestClient):
    participant_id = "P-MONITOR"
    session_record = make_session_record(
        participant_id=participant_id,
        record_id="session-monitor",
        started_at_iso="2026-02-22T10:00:00Z",
        due_at_iso="2026-02-23T10:00:00Z",
        pending_delayed=True,
    )
    delayed_record = make_delayed_record(
        participant_id=participant_id,
        record_id="delayed-monitor",
        linked_session_record_id="session-monitor",
    )

    assert client.post("/study/session-records", json={"record": session_record}).status_code == 200
    assert client.post("/study/delayed-records", json={"record": delayed_record}).status_code == 200

    activity_response = client.post(
        "/study/activity",
        json={
            "event_type": "page_view",
            "page": "study_setup",
            "participant_id": participant_id,
            "visitor_id": "VISITOR-123",
            "session_number": 1,
            "condition": "adaptive",
            "metadata": {"source": "test"},
        },
    )
    assert activity_response.status_code == 200
    assert activity_response.json()["success"] is True

    summary_response = client.get(
        "/admin/monitoring/summary",
        headers={"Authorization": "Bearer admin-token"},
    )
    assert summary_response.status_code == 200

    payload = summary_response.json()
    assert payload["totals"]["session_records"] == 1
    assert payload["totals"]["delayed_records"] == 1
    assert payload["totals"]["unique_participants"] == 1
    assert payload["totals"]["participants_with_delayed"] == 1
    assert payload["totals"]["pending_delayed_records"] == 0
    assert payload["activity"]["page_views_last_24h"] >= 1
    assert payload["activity"]["active_last_60m"] >= 1
    assert len(payload["activity"]["recent_events"]) >= 1
