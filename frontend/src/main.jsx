import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Landing } from './Landing'
import './index.css'
import App from './App.jsx'
import './i18n'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/analyse" element={<App />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
)
