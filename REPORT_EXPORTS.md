# Study Report Exports

Reports are persisted on the backend and can be listed/exported with admin-protected endpoints.

## Security

All `/admin/*` endpoints require:

```http
Authorization: Bearer <CLE_ADMIN_TOKEN>
```

Set `CLE_ADMIN_TOKEN` in `/opt/cle-app/.env.production` and keep it secret.

## Endpoint: List Report Metadata

```bash
curl -sS "https://study.example.com/admin/reports/index" \
  -H "Authorization: Bearer <CLE_ADMIN_TOKEN>" | jq
```

Optional filters:
- `participant_id`
- `from_iso`
- `to_iso`

Example:

```bash
curl -sS "https://study.example.com/admin/reports/index?participant_id=P-260222-ABC123" \
  -H "Authorization: Bearer <CLE_ADMIN_TOKEN>" | jq
```

## Endpoint: Export JSON Bundle

```bash
curl -sS "https://study.example.com/admin/reports/export?format=json" \
  -H "Authorization: Bearer <CLE_ADMIN_TOKEN>" | jq
```

## Endpoint: Export ZIP Archive

```bash
curl -sS "https://study.example.com/admin/reports/export?format=zip" \
  -H "Authorization: Bearer <CLE_ADMIN_TOKEN>" \
  -o study_reports.zip
```

Filtered ZIP export:

```bash
curl -sS "https://study.example.com/admin/reports/export?participant_id=P-260222-ABC123&from_iso=2026-02-01T00:00:00Z&to_iso=2026-02-28T23:59:59Z&format=zip" \
  -H "Authorization: Bearer <CLE_ADMIN_TOKEN>" \
  -o participant_reports.zip
```

## Data Layout on VM

Backend stores reports in:
- `/opt/cle-data/reports/sessions/<participant_id>/<record_id>.json`
- `/opt/cle-data/reports/delayed/<participant_id>/<record_id>.json`

## Recommended Backup Routine

Daily backup command on VM:

```bash
tar -czf "/opt/cle-data/backups/reports_$(date +%Y%m%d_%H%M%S).tgz" -C /opt/cle-data reports
```
