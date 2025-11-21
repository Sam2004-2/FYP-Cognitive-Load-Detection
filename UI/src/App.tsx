import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SessionSetup from './pages/SessionSetup';
import ActiveSession from './pages/ActiveSession';
import Summary from './pages/Summary';
import Settings from './pages/Settings';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SessionSetup />} />
        <Route path="/session" element={<ActiveSession />} />
        <Route path="/summary" element={<Summary />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/history" element={<Summary />} />
      </Routes>
    </Router>
  );
}

export default App;
