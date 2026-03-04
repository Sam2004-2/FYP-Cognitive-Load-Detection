import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SessionSetup from './pages/SessionSetup';
import StudySetup from './pages/StudySetup';
import StudySession from './pages/StudySession';
import StudySummary from './pages/StudySummary';
import StudyDelayedTest from './pages/StudyDelayedTest';
import AdminDashboard from './pages/AdminDashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SessionSetup />} />
        <Route path="/study/setup" element={<StudySetup />} />
        <Route path="/study/session" element={<StudySession />} />
        <Route path="/study/summary" element={<StudySummary />} />
        <Route path="/study/delayed" element={<StudyDelayedTest />} />
        <Route path="/admin" element={<AdminDashboard />} />
      </Routes>
    </Router>
  );
}

export default App;
