# Study Protocol

## Frontend Routes

- `/` - session setup entry point
- `/study/setup` - participant setup
- `/study/session` - main study session
- `/study/summary` - session summary
- `/study/delayed` - delayed memory test
- `/admin` - admin monitoring and export page

Route wiring lives in `UI/src/App.tsx`.

## Current Session Settings

The current study constants live in `UI/src/config/studyConfig.ts`.

- baseline calibration: `45s`
- easy block: `8` items at `4.5s`
- hard block: `10` items at `3.0s`
- recognition choices: `4`
- micro-break length: `45s`
- max micro-breaks per session: `1`
- controller cooldown: `120s`
- decision window: `5s`
- smoothing windows: `3`
- adaptive mode: `relative`
- relative threshold: `1.0`
- absolute fallback threshold: `0.45`

## Quality Gates

- minimum valid-frame ratio: `0.95`
- maximum illumination standard deviation: `28`

## Participant-Facing Instructions

The participant copy used by the app lives in `UI/public/study-instructions.md`.
