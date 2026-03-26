import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  downloadAdminReports,
  getAdminMonitoringSummary,
  getAdminReportIndex,
  StudyAPIError,
} from '../services/studyApiClient';
import { AdminExportQuery, AdminMonitoringSummary, AdminReportIndexResponse } from '../types/study';

const ADMIN_TOKEN_STORAGE_KEY = 'cle_admin_token';

function formatPercentFromFraction(value: number): string {
  if (!Number.isFinite(value)) return '0.0%';
  return `${(value * 100).toFixed(1)}%`;
}

const AdminDashboard: React.FC = () => {
  const [tokenInput, setTokenInput] = useState('');
  const [adminToken, setAdminToken] = useState('');
  const [summary, setSummary] = useState<AdminMonitoringSummary | null>(null);
  const [reportIndex, setReportIndex] = useState<AdminReportIndexResponse | null>(null);

  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingIndex, setLoadingIndex] = useState(false);
  const [exporting, setExporting] = useState(false);

  const [error, setError] = useState('');
  const [indexError, setIndexError] = useState('');
  const [exportError, setExportError] = useState('');

  const [participantFilter, setParticipantFilter] = useState('');
  const [fromIsoFilter, setFromIsoFilter] = useState('');
  const [toIsoFilter, setToIsoFilter] = useState('');
  const [exportFormat, setExportFormat] = useState<'zip' | 'json'>('zip');

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) ?? '';
      setTokenInput(saved);
      setAdminToken(saved.trim());
    } catch {
      // Ignore localStorage failures.
    }
  }, []);

  const loadSummary = useCallback(async () => {
    if (!adminToken.trim()) return;
    setLoadingSummary(true);
    setError('');
    try {
      const loaded = await getAdminMonitoringSummary(adminToken);
      setSummary(loaded);
    } catch (err) {
      if (err instanceof StudyAPIError) {
        setError(`Failed to load summary (${err.status ?? 'unknown'}). Check token and backend status.`);
      } else {
        setError('Failed to load summary.');
      }
      setSummary(null);
    } finally {
      setLoadingSummary(false);
    }
  }, [adminToken]);

  useEffect(() => {
    if (!adminToken.trim()) return;
    void loadSummary();
  }, [adminToken, loadSummary]);

  useEffect(() => {
    if (!adminToken.trim()) return;
    const timer = window.setInterval(() => {
      void loadSummary();
    }, 30000);
    return () => window.clearInterval(timer);
  }, [adminToken, loadSummary]);

  const exportQuery: AdminExportQuery = useMemo(
    () => ({
      participantId: participantFilter.trim() || undefined,
      fromIso: fromIsoFilter.trim() || undefined,
      toIso: toIsoFilter.trim() || undefined,
      format: exportFormat,
    }),
    [exportFormat, fromIsoFilter, participantFilter, toIsoFilter]
  );

  const saveToken = () => {
    const trimmed = tokenInput.trim();
    setAdminToken(trimmed);
    try {
      if (trimmed) {
        window.localStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, trimmed);
      } else {
        window.localStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
      }
    } catch {
      // Ignore localStorage failures.
    }
  };

  const loadIndex = async () => {
    if (!adminToken.trim()) {
      setIndexError('Enter an admin token first.');
      return;
    }
    setLoadingIndex(true);
    setIndexError('');
    try {
      const loaded = await getAdminReportIndex(adminToken, exportQuery);
      setReportIndex(loaded);
    } catch (err) {
      if (err instanceof StudyAPIError) {
        setIndexError(`Failed to load report index (${err.status ?? 'unknown'}).`);
      } else {
        setIndexError('Failed to load report index.');
      }
      setReportIndex(null);
    } finally {
      setLoadingIndex(false);
    }
  };

  const runExport = async () => {
    if (!adminToken.trim()) {
      setExportError('Enter an admin token first.');
      return;
    }
    setExporting(true);
    setExportError('');
    try {
      const blob = await downloadAdminReports(adminToken, exportQuery);
      const ext = exportFormat === 'zip' ? 'zip' : 'json';
      const filename = `study_reports_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.${ext}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      if (err instanceof StudyAPIError) {
        setExportError(`Failed to export reports (${err.status ?? 'unknown'}).`);
      } else {
        setExportError('Failed to export reports.');
      }
    } finally {
      setExporting(false);
    }
  };

  const totals = summary?.totals ?? {};
  const conditionCounts = summary?.conditionCounts ?? {};
  const interventionCounts = summary?.interventionCounts ?? {};
  const pageViewCounts = summary?.activity.pageViewCounts ?? {};

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900">Admin Monitoring Dashboard</h1>
            <p className="text-sm text-gray-600 mt-1">
              Report uploads, participant progression, and recent site activity.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
            <div className="md:col-span-3">
              <label className="block text-sm font-medium text-gray-700 mb-1">Admin Bearer Token</label>
              <input
                type="password"
                value={tokenInput}
                onChange={(event) => setTokenInput(event.target.value)}
                placeholder="Paste CLE_ADMIN_TOKEN"
                className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-gray-400"
              />
            </div>
            <button
              onClick={saveToken}
              className="w-full bg-gray-900 hover:bg-gray-800 text-white rounded-lg px-4 py-2"
            >
              Save Token
            </button>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => void loadSummary()}
              disabled={loadingSummary || !adminToken.trim()}
              className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white rounded-lg px-4 py-2"
            >
              {loadingSummary ? 'Refreshing...' : 'Refresh Summary'}
            </button>
            {summary && (
              <div className="text-sm text-gray-600 self-center">
                Updated: {new Date(summary.generatedAtIso).toLocaleString()}
              </div>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-3 py-2 text-sm">{error}</div>
          )}
        </div>

        {summary && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
              <MetricCard label="Unique participants" value={totals.unique_participants ?? 0} />
              <MetricCard label="Session records" value={totals.session_records ?? 0} />
              <MetricCard label="Delayed records" value={totals.delayed_records ?? 0} />
              <MetricCard label="Session 2 participants" value={totals.participants_with_session2 ?? 0} />
              <MetricCard label="Pending delayed records" value={totals.pending_delayed_records ?? 0} />
              <MetricCard label="Uploads last 24h" value={totals.uploads_last_24h ?? 0} />
              <MetricCard label="Active users (15m)" value={summary.activity.activeLast15m} />
              <MetricCard label="Active users (60m)" value={summary.activity.activeLast60m} />
              <MetricCard label="Visitors (24h)" value={summary.activity.visitorsLast24h} />
              <MetricCard label="Page views (24h)" value={summary.activity.pageViewsLast24h} />
              <MetricCard label="Phase integrity issues" value={totals.phase_integrity_issue_records ?? 0} />
              <MetricCard label="Total records" value={totals.total_records ?? 0} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <SimpleCountPanel title="Condition Split" values={conditionCounts} />
              <SimpleCountPanel title="Intervention Counts" values={interventionCounts} />
              <SimpleCountPanel title="Page Views by Route (24h)" values={pageViewCounts} />
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-gray-800 mb-3">Daily Uploads (Last 14 Days)</h2>
              <div className="overflow-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-gray-200 text-gray-600">
                      <th className="py-2 pr-3">Date</th>
                      <th className="py-2 pr-3">Sessions</th>
                      <th className="py-2 pr-3">Delayed</th>
                      <th className="py-2 pr-3">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.dailyUploads.map((point) => (
                      <tr key={point.date} className="border-b border-gray-100">
                        <td className="py-2 pr-3">{point.date}</td>
                        <td className="py-2 pr-3">{point.sessionRecords}</td>
                        <td className="py-2 pr-3">{point.delayedRecords}</td>
                        <td className="py-2 pr-3">{point.totalRecords}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-gray-800 mb-3">Recent Records</h2>
                <div className="overflow-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-gray-200 text-gray-600">
                        <th className="py-2 pr-3">Stored</th>
                        <th className="py-2 pr-3">Participant</th>
                        <th className="py-2 pr-3">Kind</th>
                        <th className="py-2 pr-3">Condition</th>
                        <th className="py-2 pr-3">Session</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.recentRecords.map((record) => (
                        <tr key={`${record.kind}-${record.recordId}`} className="border-b border-gray-100">
                          <td className="py-2 pr-3">{new Date(record.storedAtIso).toLocaleString()}</td>
                          <td className="py-2 pr-3 font-mono">{record.participantId}</td>
                          <td className="py-2 pr-3">{record.kind}</td>
                          <td className="py-2 pr-3">{record.condition ?? '-'}</td>
                          <td className="py-2 pr-3">{record.sessionNumber ?? '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h2 className="text-lg font-semibold text-gray-800 mb-3">Recent Site Activity</h2>
                <div className="overflow-auto max-h-96">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left border-b border-gray-200 text-gray-600">
                        <th className="py-2 pr-3">Time</th>
                        <th className="py-2 pr-3">Event</th>
                        <th className="py-2 pr-3">Page</th>
                        <th className="py-2 pr-3">Participant</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.activity.recentEvents.map((event, idx) => (
                        <tr key={`${event.occurredAtIso}-${idx}`} className="border-b border-gray-100">
                          <td className="py-2 pr-3">{new Date(event.occurredAtIso).toLocaleString()}</td>
                          <td className="py-2 pr-3">{event.eventType}</td>
                          <td className="py-2 pr-3">{event.page}</td>
                          <td className="py-2 pr-3 font-mono">{event.participantId ?? event.visitorId ?? '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}

        <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
          <h2 className="text-xl font-semibold text-gray-800">Report Index and Export</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Participant ID</label>
              <input
                value={participantFilter}
                onChange={(event) => setParticipantFilter(event.target.value)}
                placeholder="Optional"
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">From (ISO)</label>
              <input
                value={fromIsoFilter}
                onChange={(event) => setFromIsoFilter(event.target.value)}
                placeholder="2026-03-01T00:00:00Z"
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">To (ISO)</label>
              <input
                value={toIsoFilter}
                onChange={(event) => setToIsoFilter(event.target.value)}
                placeholder="2026-03-31T23:59:59Z"
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Format</label>
              <select
                value={exportFormat}
                onChange={(event) => setExportFormat(event.target.value as 'zip' | 'json')}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="zip">zip</option>
                <option value="json">json</option>
              </select>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => void loadIndex()}
              disabled={loadingIndex}
              className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white rounded-lg px-4 py-2"
            >
              {loadingIndex ? 'Loading Index...' : 'Load Report Index'}
            </button>
            <button
              onClick={() => void runExport()}
              disabled={exporting}
              className="bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 text-white rounded-lg px-4 py-2"
            >
              {exporting ? 'Exporting...' : 'Download Export'}
            </button>
          </div>

          {indexError && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-3 py-2 text-sm">{indexError}</div>
          )}
          {exportError && (
            <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-3 py-2 text-sm">{exportError}</div>
          )}

          {reportIndex && (
            <div className="space-y-3">
              <div className="text-sm text-gray-700">
                Index count: <span className="font-semibold">{reportIndex.count}</span> records
              </div>
              <div className="overflow-auto max-h-80 border border-gray-100 rounded-lg">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left border-b border-gray-200 text-gray-600">
                      <th className="py-2 px-3">Stored</th>
                      <th className="py-2 px-3">Participant</th>
                      <th className="py-2 px-3">Kind</th>
                      <th className="py-2 px-3">Record ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reportIndex.records.map((record) => (
                      <tr key={`${record.kind}-${record.recordId}`} className="border-b border-gray-100">
                        <td className="py-2 px-3">{new Date(record.storedAtIso).toLocaleString()}</td>
                        <td className="py-2 px-3 font-mono">{record.participantId}</td>
                        <td className="py-2 px-3">{record.kind}</td>
                        <td className="py-2 px-3 font-mono">{record.recordId}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const MetricCard: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div className="bg-white border border-gray-200 rounded-lg p-4">
    <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
    <div className="text-2xl font-semibold text-gray-800 mt-1">{Number(value).toLocaleString()}</div>
  </div>
);

const SimpleCountPanel: React.FC<{ title: string; values: Record<string, number> }> = ({ title, values }) => {
  const entries = Object.entries(values).sort((a, b) => b[1] - a[1]);
  const total = entries.reduce((sum, entry) => sum + entry[1], 0);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">{title}</h2>
      {entries.length === 0 ? (
        <div className="text-sm text-gray-500">No data yet.</div>
      ) : (
        <div className="space-y-2">
          {entries.map(([key, value]) => (
            <div key={key} className="flex items-center justify-between text-sm">
              <span className="text-gray-700">{key}</span>
              <span className="text-gray-900 font-medium">
                {value} {total > 0 ? `(${formatPercentFromFraction(value / total)})` : ''}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
