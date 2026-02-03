import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SessionSetup from './pages/SessionSetup';
import ActiveSession from './pages/ActiveSession';
import Summary from './pages/Summary';
import Settings from './pages/Settings';
import DataCollection from './pages/DataCollection';
import PilotStudy from './pages/PilotStudy';
import DelayedTest from './pages/DelayedTest';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SessionSetup />} />
        <Route path="/session" element={<ActiveSession />} />
        <Route path="/summary" element={<Summary />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/collect" element={<DataCollection />} />
        <Route path="/pilot" element={<PilotStudy />} />
        <Route path="/study/delayed/:sessionId" element={<DelayedTest />} />
      </Routes>
    </Router>
  );
}

export default App;
