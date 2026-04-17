# Data Schema

## Stored Report Layout

The backend stores reports under:

- `reports/sessions/<participant_id>/<record_id>.json`
- `reports/delayed/<participant_id>/<record_id>.json`
- `reports/activity/<YYYY-MM-DD>.ndjson`

At runtime this is rooted at `CLE_REPORTS_DIR`.

## Session Record

Session records are represented in the frontend by `StudySessionRecord` in `UI/src/types/study.ts`.

Key top-level fields:

- `recordVersion`
- `recordId`
- `participantId`
- `sessionNumber`
- `assignment`
- `plan`
- `startedAtIso`
- `completedAtIso`
- `condition`
- `form`
- `cliSamples`
- `featureWindows`
- `interventions`
- `trials`
- `arithmeticChallenge`
- `blockSummaries`
- `runtimeDiagnostics`
- `nasaTlx`
- `pendingDelayedTest`
- `delayedDueAtIso`

## Delayed Record

Delayed-test records are represented by `StudyDelayedTestRecord`.

Key top-level fields:

- `recordVersion`
- `recordId`
- `linkedSessionRecordId`
- `participantId`
- `sessionNumber`
- `condition`
- `form`
- `dueAtIso`
- `completedAtIso`
- `trials`
- `recognitionAccuracy`
- `recognitionMeanRtMs`
- `cuedRecallAccuracy`
- `cuedRecallMeanRtMs`

## Activity Event

Activity events are sent through `/study/activity` and stored as NDJSON. They include:

- `event_type`
- `page`
- `participant_id`
- `visitor_id`
- `session_number`
- `condition`
- `metadata`

## Admin Export Shapes

Admin export and index responses are represented by:

- `AdminReportIndexResponse`
- `AdminMonitoringSummary`

Both are defined in `UI/src/types/study.ts`.
