import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useDarkMode } from './hooks/useDarkMode'
import { ArrowLeft } from 'lucide-react'
import { btnOutline } from './styles/shared'

const LAST_UPDATED = '13 września 2026'
const LAST_UPDATED_EN = 'September 13, 2026'
const CONTACT_EMAIL = 'kontakt@stylizzer.com'

export function PrivacyPolicy() {
  const { i18n } = useTranslation()
  const navigate  = useNavigate()
  const pl = i18n.language === 'pl'
  const [dark] = useDarkMode()

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}
      data-theme={dark ? 'dark' : 'light'}>

      {/* nav */}
      <nav style={{
        padding: '16px 32px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', gap: 12,
        position: 'sticky', top: 0, background: 'var(--bg)', zIndex: 100,
      }}>
        <button onClick={() => navigate('/')} style={{
          ...btnOutline, display: 'inline-flex', alignItems: 'center', gap: 6,
        }}>
          <ArrowLeft size={13} strokeWidth={1.5} />
          {pl ? 'Wróć' : 'Back'}
        </button>
        <span style={{
          fontFamily: 'var(--font-display)', fontSize: 16,
          fontWeight: 500, color: 'var(--text)',
        }}>
          {pl ? 'Polityka Prywatności' : 'Privacy Policy'}
        </span>
      </nav>

      {/* content */}
      <div style={{ maxWidth: 720, margin: '0 auto', padding: '48px 24px 80px', textAlign: 'left' }}>
        {pl ? <PolicyPL /> : <PolicyEN />}
      </div>
    </div>
  )
}

const H1 = ({ children }) => (
  <h1 style={{
    fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 500,
    color: 'var(--text)', marginBottom: 8, lineHeight: 1.2,
  }}>{children}</h1>
)
const H2 = ({ children }) => (
  <h2 style={{
    fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 500,
    color: 'var(--text)', marginTop: 40, marginBottom: 12,
  }}>{children}</h2>
)
const P = ({ children }) => (
  <p style={{
    fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.75,
    fontWeight: 300, marginBottom: 12,
  }}>{children}</p>
)
const UL = ({ children }) => (
  <ul style={{
    paddingLeft: 20, marginBottom: 12, textAlign: 'left',
    fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.75, fontWeight: 300,
  }}>{children}</ul>
)
const LI = ({ children }) => <li style={{ marginBottom: 6 }}>{children}</li>
const Table = ({ rows }) => (
  <div style={{ overflowX: 'auto', marginBottom: 20 }}>
    <table style={{
      width: '100%', borderCollapse: 'collapse',
      fontSize: 13, color: 'var(--text-muted)',
    }}>
      <thead>
        <tr>
          {rows[0].map((h, i) => (
            <th key={i} style={{
              padding: '8px 12px', textAlign: 'left',
              borderBottom: '2px solid var(--border)',
              fontWeight: 500, color: 'var(--text)',
              fontFamily: 'var(--font-body)',
            }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.slice(1).map((row, ri) => (
          <tr key={ri}>
            {row.map((cell, ci) => (
              <td key={ci} style={{
                padding: '8px 12px',
                borderBottom: '1px solid var(--border)',
                verticalAlign: 'top',
              }}>{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)
const Divider = () => (
  <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '32px 0' }} />
)

function PolicyPL() {
  return (
    <>
      <H1>Polityka Prywatności</H1>
      <P style={{ fontSize: 12 }}>
        Ostatnia aktualizacja: {LAST_UPDATED} · Kontakt: {CONTACT_EMAIL}
      </P>

      <Divider />

      <H2>1. Administrator danych osobowych</H2>
      <P>
        Administratorem Twoich danych osobowych jest *, prowadzący
        serwis Stylizzer dostępny pod adresem stylizzer.vercel.app
        (dalej: „Serwis"). Kontakt w sprawach ochrony danych: {CONTACT_EMAIL}.
      </P>

      <H2>2. Jakie dane zbieramy i w jakim celu</H2>
      <Table rows={[
        ['Kategoria danych', 'Cel', 'Podstawa prawna', 'Okres przechowywania'],
        ['Adres e-mail', 'Rejestracja i logowanie', 'Art. 6 ust. 1 lit. b RODO (wykonanie umowy)', 'Do usunięcia konta'],
        ['Zdjęcia twarzy', 'Jednorazowa analiza geometrii twarzy', 'Art. 6 ust. 1 lit. a, Art. 9 ust. 2 lit. a RODO (wyraźna zgoda)', 'NIE przechowujemy — przetwarzamy wyłącznie w czasie analizy'],
        ['Wyniki analizy (cechy geometryczne, bez zdjęcia)', 'Historia analiz, poprawa usługi', 'Art. 6 ust. 1 lit. a RODO (zgoda)', 'Do usunięcia konta lub na żądanie'],
        ['Dane techniczne (IP, typ przeglądarki)', 'Bezpieczeństwo i stabilność', 'Art. 6 ust. 1 lit. f RODO (prawnie uzasadniony interes)', 'Do 90 dni (logi serwera)'],
      ]} />

      <P>
        <strong>Ważne:</strong> Twoje zdjęcia twarzy są danymi biometrycznymi w rozumieniu
        art. 4 pkt 14 RODO. Przetwarzamy je wyłącznie na podstawie Twojej wyraźnej zgody,
        jednorazowo w czasie analizy, i nie przechowujemy ich po jej zakończeniu.
      </P>

      <H2>3. Komu udostępniamy dane</H2>
      <P>
        W celu świadczenia usługi korzystamy z następujących serwisów zewnętrznych:
      </P>
      <Table rows={[
        ['Podprocesor', 'Rola', 'Kraj serwerów', 'Polityka prywatności'],
        ['Supabase Inc.', 'Baza danych i uwierzytelnianie', 'EU (Frankfurt)', 'supabase.com/privacy'],
        ['Groq Inc.', 'Analiza tekstu przez AI (LLM)', 'USA (SCCs)', 'groq.com/privacy'],
        ['fal.ai Inc.', 'Generowanie podglądu fryzury (FLUX AI)', 'USA (SCCs)', 'fal.ai/privacy'],
        ['Hugging Face Inc.', 'Modele ML', 'USA/EU (SCCs)', 'huggingface.co/privacy'],
        ['Vercel Inc.', 'Hosting frontendu', 'USA/EU (SCCs)', 'vercel.com/legal/privacy-policy'],
      ]} />
      <P>
        Transfery danych do USA odbywają się na podstawie Standardowych Klauzul Umownych (SCCs)
        zatwierdzonych przez Komisję Europejską.
      </P>

      <H2>4. Pliki cookies</H2>
      <P>
        Używamy wyłącznie cookies niezbędnych do działania serwisu:
      </P>
      <UL>
        <LI><strong>Cookies sesji Supabase</strong> — przechowują token logowania. Niezbędne do
          funkcjonowania konta. Czas życia: do wylogowania lub 7 dni (remember me).</LI>
        <LI><strong>localStorage</strong> — przechowujemy preferencje językowe i ustawienia
          motywu kolorystycznego. Nie służą śledzeniu.</LI>
      </UL>
      <P>
        Nie używamy cookies reklamowych, analitycznych ani śledzących stron trzecich.
      </P>

      <H2>5. Twoje prawa (RODO)</H2>
      <P>Przysługują Ci następujące prawa:</P>
      <UL>
        <LI><strong>Dostęp</strong> - możesz poprosić o kopię swoich danych.</LI>
        <LI><strong>Sprostowanie</strong> - możesz poprosić o korektę błędnych danych.</LI>
        <LI><strong>Usunięcie</strong> - możesz usunąć konto i wszystkie dane (historia analiz
          usuwana jest automatycznie wraz z kontem).</LI>
        <LI><strong>Ograniczenie przetwarzania</strong> - możesz poprosić o wstrzymanie
          przetwarzania.</LI>
        <LI><strong>Przenoszenie</strong> - możesz otrzymać swoje dane w formacie JSON.</LI>
        <LI><strong>Cofnięcie zgody</strong> - w każdej chwili możesz cofnąć zgodę na
          przetwarzanie zdjęć. Nie wpływa to na przetwarzanie dokonane przed jej cofnięciem.</LI>
        <LI><strong>Skarga</strong> - możesz wnieść skargę do Prezesa UODO (uodo.gov.pl).</LI>
      </UL>
      <P>W celu realizacji praw napisz na: {CONTACT_EMAIL}</P>

      <H2>6. Bezpieczeństwo danych</H2>
      <P>
        Stosujemy następujące środki techniczne i organizacyjne:
      </P>
      <UL>
        <LI>Szyfrowanie transmisji danych (TLS/HTTPS)</LI>
        <LI>Szyfrowanie danych w spoczynku (Supabase — AES-256)</LI>
        <LI>Uwierzytelnianie z tokenami JWT z ograniczonym czasem ważności</LI>
        <LI>Kontrola dostępu na poziomie wiersza (Row Level Security) w bazie danych</LI>
        <LI>Brak przechowywania zdjęć twarzy po zakończeniu analizy</LI>
        <LI>Ograniczenia liczby zapytań (rate limiting) na endpointach API</LI>
      </UL>

      <H2>7. Wiek użytkowników</H2>
      <P>
        Serwis jest przeznaczony dla osób, które ukończyły 16 lat. Nie zbieramy
        świadomie danych osób poniżej tego wieku. Jeśli dowiesz się, że osoba poniżej
        16 lat przekazała nam dane, skontaktuj się z nami - usuniemy je niezwłocznie.
      </P>

      <H2>8. Zmiany polityki prywatności</H2>
      <P>
        O istotnych zmianach powiadomimy e-mailem (jeśli posiadasz konto) lub komunikatem
        w Serwisie z 14-dniowym wyprzedzeniem. Data ostatniej aktualizacji widnieje
        na górze niniejszego dokumentu.
      </P>

      <H2>9. Kontakt</H2>
      <P>
        W sprawach dotyczących ochrony danych osobowych skontaktuj się z nami:<br />
        E-mail: {CONTACT_EMAIL}
      </P>
    </>
  )
}

function PolicyEN() {
  return (
    <>
      <H1>Privacy Policy</H1>
      <P>Last updated: {LAST_UPDATED_EN} · Contact: {CONTACT_EMAIL}</P>

      <Divider />

      <H2>1. Data controller</H2>
      <P>
        The data controller for your personal data is *, operating
        the Stylizzer service at stylizzer.vercel.app (the "Service").
        For data protection enquiries: {CONTACT_EMAIL}.
      </P>

      <H2>2. What data we collect and why</H2>
      <Table rows={[
        ['Category', 'Purpose', 'Legal basis', 'Retention'],
        ['Email address', 'Account registration and login', 'Art. 6(1)(b) GDPR — performance of a contract', 'Until account deletion'],
        ['Facial photographs', 'One-time face geometry analysis', 'Art. 6(1)(a) and Art. 9(2)(a) GDPR — explicit consent', 'NOT stored — processed only during analysis'],
        ['Analysis results (geometry, no photo)', 'Analysis history, service improvement', 'Art. 6(1)(a) GDPR — consent', 'Until account deletion or on request'],
        ['Technical data (IP, browser)', 'Security and stability', 'Art. 6(1)(f) GDPR — legitimate interest', 'Up to 90 days (server logs)'],
      ]} />

      <P>
        <strong>Important:</strong> Your facial photographs constitute biometric data
        under Art. 4(14) GDPR. We process them solely on the basis of your explicit consent,
        one time during analysis, and do not store them afterwards.
      </P>

      <H2>3. Sub-processors — who we share data with</H2>
      <Table rows={[
        ['Sub-processor', 'Role', 'Server location', 'Privacy policy'],
        ['Supabase Inc.', 'Database and authentication', 'EU (Frankfurt)', 'supabase.com/privacy'],
        ['Groq Inc.', 'AI text analysis (LLM)', 'USA (SCCs)', 'groq.com/privacy'],
        ['fal.ai Inc.', 'Hairstyle preview generation (FLUX AI)', 'USA (SCCs)', 'fal.ai/privacy'],
        ['Hugging Face Inc.', 'ML models (hair segmentation)', 'USA/EU (SCCs)', 'huggingface.co/privacy'],
        ['Vercel Inc.', 'Frontend hosting', 'USA/EU (SCCs)', 'vercel.com/legal/privacy-policy'],
      ]} />
      <P>
        Transfers to the USA are made under Standard Contractual Clauses (SCCs)
        approved by the European Commission.
      </P>

      <H2>4. Cookies</H2>
      <P>We use only cookies strictly necessary for the Service to function:</P>
      <UL>
        <LI><strong>Supabase session cookies</strong> — store your login token. Required for account
          functionality. Lifetime: until logout or 7 days (remember me).</LI>
        <LI><strong>localStorage</strong> — stores your language preference and colour theme.
          Not used for tracking.</LI>
      </UL>
      <P>We do not use advertising, analytics, or third-party tracking cookies.</P>

      <H2>5. Your rights (GDPR)</H2>
      <UL>
        <LI><strong>Access</strong> — you may request a copy of your data.</LI>
        <LI><strong>Rectification</strong> — you may ask us to correct inaccurate data.</LI>
        <LI><strong>Erasure</strong> — you may delete your account; analysis history is
          deleted automatically with the account.</LI>
        <LI><strong>Restriction</strong> — you may ask us to pause processing.</LI>
        <LI><strong>Portability</strong> — you may receive your data in JSON format.</LI>
        <LI><strong>Withdrawal of consent</strong> — you may withdraw consent for photo
          processing at any time. This does not affect processing already carried out.</LI>
        <LI><strong>Complaint</strong> — you may lodge a complaint with your national
          supervisory authority (e.g. UODO in Poland).</LI>
      </UL>
      <P>To exercise your rights, write to: {CONTACT_EMAIL}</P>

      <H2>6. Data security</H2>
      <UL>
        <LI>TLS/HTTPS encryption for all data in transit</LI>
        <LI>Encryption at rest (Supabase — AES-256)</LI>
        <LI>JWT authentication with limited token lifetime</LI>
        <LI>Row Level Security (RLS) in the database</LI>
        <LI>Facial photographs are not stored after analysis</LI>
        <LI>Rate limiting on all API endpoints</LI>
      </UL>

      <H2>7. Age restriction</H2>
      <P>
        The Service is intended for users aged 16 and over. We do not knowingly collect
        data from anyone under 16. If you believe a person under 16 has provided us with
        data, contact us and we will delete it promptly.
      </P>

      <H2>8. Changes to this policy</H2>
      <P>
        We will notify you of material changes by email (if you have an account) or
        via an in-Service notice at least 14 days in advance. The date of the
        most recent update is shown at the top of this document.
      </P>

      <H2>9. Contact</H2>
      <P>
        For any data protection enquiries:<br />
        Email: {CONTACT_EMAIL}
      </P>
    </>
  )
}