import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './i18n'
import './index.css'
import { Landing } from './Landing'
import { PrivacyPolicy } from './PrivacyPolicy'
import { TermsOfService } from './TermsOfService'
import { NotFound } from './NotFound'
import { ErrorBoundary } from './components/ErrorBoundary.jsx'
import { Admin } from './Admin'
import App from './App.jsx'

const authUrl = window.location.hash + window.location.search
if (/type=recovery/.test(authUrl)) sessionStorage.setItem('pw_recovery', '1')
if (/error_code=otp_expired|error=access_denied/.test(authUrl)) sessionStorage.setItem('auth_link_error', '1')

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/analyse" element={<App />} />
          <Route path="/privacy" element={<PrivacyPolicy />} />
          <Route path="/terms" element={<TermsOfService />} />
          <Route path="/admin" element={<Admin />} /> 
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>
)