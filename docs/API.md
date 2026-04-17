# API

The backend is implemented in `machine_learning/src/cle/server.py`.

## Public Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | basic API root |
| `GET` | `/health` | health check |
| `GET` | `/model-info` | current model metadata and feature list |
| `POST` | `/predict` | predict CLI from a feature map |
| `POST` | `/save-training-data` | save collected labelled samples |
| `POST` | `/study/participants` | generate a participant ID |
| `POST` | `/study/session-records` | upload one study session record |
| `POST` | `/study/delayed-records` | upload one delayed-test record |
| `POST` | `/study/activity` | store a client activity event |
| `GET` | `/study/pending-delayed/{participant_id}` | list pending delayed tasks |

## Admin Routes

All `/admin/*` routes require:

```http
Authorization: Bearer <CLE_ADMIN_TOKEN>
```

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/admin/reports/index` | list stored report metadata |
| `GET` | `/admin/monitoring/summary` | dashboard summary for uploads and activity |
| `GET` | `/admin/reports/export` | export reports as JSON or ZIP |

## Minimal Request Shapes

`POST /predict`

```json
{
  "features": {
    "blink_rate": 0.0,
    "blink_count": 0.0
  }
}
```

`POST /study/session-records`

```json
{
  "record": {
    "recordVersion": 3,
    "recordId": "example-record",
    "participantId": "P-EXAMPLE"
  }
}
```

`POST /study/delayed-records`

```json
{
  "record": {
    "recordVersion": 3,
    "recordId": "example-delayed",
    "linkedSessionRecordId": "example-record",
    "participantId": "P-EXAMPLE"
  }
}
```

`POST /study/activity`

```json
{
  "event_type": "page_view",
  "page": "study_setup",
  "participant_id": "P-EXAMPLE",
  "metadata": {}
}
```

For the full frontend-facing shapes, see `UI/src/types/study.ts` and `UI/src/types/features.ts`.
