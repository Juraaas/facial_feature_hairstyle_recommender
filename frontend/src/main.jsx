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
import * as Sentry from '@sentry/react'

const authUrl = window.location.hash + window.location.search
if (/type=recovery/.test(authUrl)) sessionStorage.setItem('pw_recovery', '1')
if (/error_code=otp_expired|error=access_denied/.test(authUrl)) sessionStorage.setItem('auth_link_error', '1')

function cleanUrl(u) {
  if (!u || typeof u !== 'string') return u
  return u.split('#')[0].replace(/([?&])(access_token|refresh_token|token|token_hash|code)=[^&]*/g, '$1$2=[removed]')
}

if (import.meta.env.VITE_SENTRY_DSN) {
  Sentry.init({
    dsn: import.meta.env.VITE_SENTRY_DSN,
    environment: import.meta.env.MODE,
    sendDefaultPii: false,
    tracesSampleRate: 0,
    ignoreErrors: [
      'ResizeObserver loop limit exceeded',
      'ResizeObserver loop completed with undelivered notifications',
      'Non-Error promise rejection captured',
    ],
    beforeBreadcrumb(b) {
      if (b?.data) for (const k of ['url', 'from', 'to']) if (b.data[k]) b.data[k] = cleanUrl(b.data[k])
      return b
    },
    beforeSend(event) {
      if (event.request) {
        event.request.url = cleanUrl(event.request.url)
        delete event.request.headers
        delete event.request.cookies
        delete event.request.query_string
      }
      delete event.user
      return event
    },
  })
}

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