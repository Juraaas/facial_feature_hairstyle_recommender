import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useDarkMode } from './hooks/useDarkMode'
import { ArrowLeft } from 'lucide-react'
import { btnOutline } from './styles/shared'

const LAST_UPDATED = '14 września 2026'
const LAST_UPDATED_EN = 'September 14, 2026'
const CONTACT_EMAIL = ''
const SERVICE_URL = 'stylizzer.vercel.app'

export function TermsOfService() {
  const { i18n } = useTranslation()
  const navigate = useNavigate()
  const pl = i18n.language === 'pl'
  const [dark] = useDarkMode()

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}
      data-theme={dark ? 'dark' : 'light'}>

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
          {pl ? 'Regulamin' : 'Terms of Service'}
        </span>
      </nav>

      <div style={{ maxWidth: 720, margin: '0 auto', padding: '48px 24px 80px' }}>
        {pl ? <TermsPL /> : <TermsEN />}
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
    fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.75, fontWeight: 300, marginBottom: 12,
  }}>{children}</p>
)
const UL = ({ children }) => (
  <ul style={{
    paddingLeft: 20, marginBottom: 12, textAlign: 'left',
    fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.75, fontWeight: 300, 
  }}>{children}</ul>
)
const LI = ({ children }) => <li style={{ marginBottom: 6 }}>{children}</li>
const Divider = () => (
  <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '32px 0' }} />
)

function TermsPL() {
  return (
    <>
      <H1>Regulamin Serwisu</H1>
      <P>Ostatnia aktualizacja: {LAST_UPDATED} · Kontakt: {CONTACT_EMAIL}</P>

      <Divider />

      <H2>1. Postanowienia ogólne</H2>
      <P>
        Niniejszy regulamin określa zasady korzystania z serwisu Stylizzer
        dostępnego pod adresem {SERVICE_URL} (dalej: „Serwis"), prowadzonego
        przez X (dalej: „Usługodawca").
      </P>
      <P>
        Korzystanie z Serwisu oznacza akceptację niniejszego Regulaminu.
      </P>

      <H2>2. Opis usługi</H2>
      <P>
        Stylizzer to aplikacja internetowa umożliwiająca:
      </P>
      <UL>
        <LI>Analizę geometrii twarzy na podstawie wgranego zdjęcia</LI>
        <LI>Otrzymanie spersonalizowanych rekomendacji fryzur</LI>
        <LI>Generowanie podglądu fryzury na zdjęciu użytkownika (funkcja Premium)</LI>
        <LI>Przechowywanie historii analiz (po założeniu konta)</LI>
      </UL>
      <P>
        Serwis jest dostępny w wersji darmowej (ograniczone funkcje) oraz Premium (pełny dostęp).
        Plan Premium jest dostępny w modelu czasowego dostępu z limitami użycia.
        Aktualny plan obejmuje X analiz twarzy i Y podglądów fryzur miesięcznie.
        Szczegóły aktualnego planu dostępne są na stronie głównej.
      </P>

      <H2>3. Wymagania i ograniczenia wiekowe</H2>
      <P>
        Z Serwisu mogą korzystać wyłącznie osoby, które ukończyły 16 lat.
        Korzystając z Serwisu, potwierdzasz, że spełniasz ten warunek.
      </P>

      <H2>4. Zasady korzystania z Serwisu</H2>
      <P>Użytkownik zobowiązuje się do:</P>
      <UL>
        <LI>Wgrywania wyłącznie własnych zdjęć lub zdjęć, do których posiada prawa</LI>
        <LI>Niewgrywania zdjęć osób trzecich bez ich zgody</LI>
        <LI>Niewgrywania zdjęć osób niepełnoletnich</LI>
        <LI>Nieużywania Serwisu do celów niezgodnych z prawem</LI>
        <LI>Niepodejmowania prób ominięcia zabezpieczeń technicznych</LI>
        <LI>Nieprzeciążania infrastruktury Serwisu (np. przez automatyczne wysyłanie zapytań)</LI>
      </UL>
      <P>
        Usługodawca zastrzega sobie prawo do zawieszenia lub usunięcia konta użytkownika
        naruszającego powyższe zasady, bez uprzedzenia.
      </P>

      <H2>5. Konta użytkowników</H2>
      <P>
        Rejestracja konta jest dobrowolna i bezpłatna. Użytkownik zobowiązuje się do:
      </P>
      <UL>
        <LI>Podania prawdziwego adresu e-mail</LI>
        <LI>Zachowania poufności hasła</LI>
        <LI>Niezwłocznego poinformowania nas o nieautoryzowanym dostępie do konta</LI>
      </UL>
      <P>
        Każdy użytkownik może posiadać tylko jedno konto.
        Usługodawca nie ponosi odpowiedzialności za szkody wynikłe z nieautoryzowanego
        użycia konta spowodowanego niedbalstwem użytkownika.
      </P>

      <H2>6. Płatności i plan Premium</H2>
      <P>
        Plan Premium jest dostępny za jednorazową opłatą. Płatności obsługiwane są
        przez Stripe Inc., bezpieczny procesor płatności. Usługodawca nie przechowuje
        danych kart płatniczych.
      </P>
      <P>
        Po pomyślnym dokonaniu płatności dostęp Premium jest aktywowany niezwłocznie.
        W przypadku problemów technicznych prosimy o kontakt: {CONTACT_EMAIL}.
      </P>
      <P>
        <strong>Zwroty:</strong> Ze względu na cyfrowy charakter usługi i jej natychmiastowe
        dostarczenie, zwroty nie są standardowo realizowane. W wyjątkowych przypadkach
        prosimy o kontakt, każda sytuacja jest rozpatrywana indywidualnie.
      </P>

      <H2>7. Własność intelektualna</H2>
      <P>
        Wszelkie prawa do Serwisu, jego wyglądu, kodu oraz treści należą do Usługodawcy
        lub są używane na podstawie odpowiednich licencji.
      </P>
      <P>
        Użytkownik zachowuje pełne prawa do wgrywanych zdjęć. Udziela jedynie
        niewyłącznej, nieodpłatnej licencji na jednorazowe przetworzenie zdjęcia
        w celu wykonania analizy.
      </P>
      <P>
        Wyniki generowane przez Serwis (rekomendacje, podglądy fryzur) są przeznaczone
        do użytku osobistego. Ich komercyjne wykorzystanie bez zgody Usługodawcy
        jest zabronione.
      </P>

      <H2>8. Ograniczenie odpowiedzialności</H2>
      <P>
        Serwis i jego wyniki mają charakter informacyjny i rozrywkowy. Nie stanowią
        profesjonalnej porady stylistycznej. Usługodawca nie gwarantuje:
      </P>
      <UL>
        <LI>Że rekomendowane fryzury będą odpowiednie dla każdego użytkownika</LI>
        <LI>Nieprzerwanego działania Serwisu</LI>
        <LI>Że generowane podglądy fryzur będą perfekcyjne</LI>
      </UL>
      <P>
        Odpowiedzialność Usługodawcy jest ograniczona do maksymalnej kwoty zapłaconej
        przez użytkownika za Plan Premium w ciągu ostatnich 12 miesięcy.
      </P>

      <H2>9. Dostępność i zmiany Serwisu</H2>
      <P>
        Usługodawca dołoży wszelkich starań, aby Serwis był dostępny nieprzerwanie,
        jednak nie gwarantuje 100% dostępności. Zastrzega sobie prawo do:
      </P>
      <UL>
        <LI>Czasowego zawieszenia Serwisu w celach konserwacyjnych</LI>
        <LI>Zmiany funkcjonalności Serwisu</LI>
        <LI>Zakończenia świadczenia usługi z 30-dniowym wyprzedzeniem</LI>
      </UL>

      <H2>10. Zmiany Regulaminu</H2>
      <P>
        O istotnych zmianach Regulaminu użytkownicy posiadający konta zostaną
        powiadomieni e-mailem z 14-dniowym wyprzedzeniem. Dalsze korzystanie
        z Serwisu po tym terminie oznacza akceptację nowego Regulaminu.
      </P>

      <H2>11. Prawo właściwe</H2>
      <P>
        Regulamin podlega prawu polskiemu. Wszelkie spory rozpatrywane będą przez
        sąd właściwy dla miejsca zamieszkania Usługodawcy, chyba że bezwzględnie
        obowiązujące przepisy konsumenckie stanowią inaczej.
      </P>
      <P>
        Konsumenci mają prawo do pozasądowego rozwiązywania sporów - więcej informacji
        na stronie ec.europa.eu/consumers/odr.
      </P>

      <H2>12. Kontakt</H2>
      <P>
        W sprawach dotyczących Regulaminu: {CONTACT_EMAIL}
      </P>
    </>
  )
}

function TermsEN() {
  return (
    <>
      <H1>Terms of Service</H1>
      <P>Last updated: {LAST_UPDATED_EN} · Contact: {CONTACT_EMAIL}</P>

      <Divider />

      <H2>1. General provisions</H2>
      <P>
        These Terms of Service govern your use of the Stylizzer service
        available at {SERVICE_URL} (the "Service"), operated by Jakub Jurewicz
        (the "Provider").
      </P>
      <P>
        By using the Service you accept these Terms. If you do not agree,
        please do not use the Service.
      </P>

      <H2>2. Description of the Service</H2>
      <P>Stylizzer is a web application that allows you to:</P>
      <UL>
        <LI>Analyse your facial geometry from an uploaded photo</LI>
        <LI>Receive personalised hairstyle recommendations</LI>
        <LI>Generate a hairstyle preview on your photo (Premium feature)</LI>
        <LI>Store your analysis history (registered users)</LI>
      </UL>
      <P>
        The Service is available in a free tier (limited features) and a Premium
        tier (full access). The Premium Plan is available on a time-limited basis with usage limits.
        Your current plan includes X facial analyses and Y hairstyle previews per month.
        Details of your current plan are available on the homepage.
      </P>

      <H2>3. Age requirement</H2>
      <P>
        You must be at least 16 years old to use the Service.
        By using the Service you confirm that you meet this requirement.
      </P>

      <H2>4. Acceptable use</H2>
      <P>You agree to:</P>
      <UL>
        <LI>Upload only photos of yourself or photos for which you hold the rights</LI>
        <LI>Not upload photos of third parties without their explicit consent</LI>
        <LI>Not upload photos of minors</LI>
        <LI>Not use the Service for any unlawful purpose</LI>
        <LI>Not attempt to circumvent any security measures</LI>
        <LI>Not overload the Service infrastructure (e.g. automated bulk requests)</LI>
      </UL>
      <P>
        The Provider reserves the right to suspend or delete accounts that violate
        these rules without prior notice.
      </P>

      <H2>5. User accounts</H2>
      <P>
        Registration is voluntary and free. You agree to provide a valid email
        address, keep your password confidential, and notify us immediately of any
        unauthorised access to your account.
      </P>
      <P>
        Each user may hold only one account. The Provider is not liable for losses
        resulting from unauthorised use of your account due to your own negligence.
      </P>

      <H2>6. Payments and Premium plan</H2>
      <P>
        The Premium plan is available for a one-time fee, processed securely by
        Stripe Inc. The Provider does not store payment card details.
      </P>
      <P>
        Premium access is activated immediately upon successful payment.
        For payment issues please contact: {CONTACT_EMAIL}.
      </P>
      <P>
        <strong>Refunds:</strong> Due to the digital and immediately delivered
        nature of the service, refunds are not provided as standard.
        Exceptional cases are reviewed individually — please contact us.
      </P>

      <H2>7. Intellectual property</H2>
      <P>
        All rights to the Service, its design, code and content belong to the
        Provider or are used under appropriate licences.
      </P>
      <P>
        You retain full ownership of your uploaded photos. You grant the Provider
        a non-exclusive, royalty-free licence solely to process your photo
        for the purpose of performing the analysis.
      </P>
      <P>
        Results generated by the Service (recommendations, hairstyle previews)
        are for personal use only. Commercial use without the Provider's
        written consent is prohibited.
      </P>

      <H2>8. Limitation of liability</H2>
      <P>
        The Service and its results are informational and for entertainment purposes.
        They do not constitute professional styling advice. The Provider does not
        guarantee that recommendations will suit every user, uninterrupted service
        availability, or perfect quality of generated previews.
      </P>
      <P>
        The Provider's liability is limited to the amount paid by you for the
        Premium plan in the preceding 12 months.
      </P>

      <H2>9. Availability and changes</H2>
      <P>
        The Provider will make reasonable efforts to keep the Service available
        but does not guarantee 100% uptime. The Provider reserves the right to
        temporarily suspend the Service for maintenance, change its features,
        or discontinue it with 30 days' notice.
      </P>

      <H2>10. Changes to these Terms</H2>
      <P>
        Registered users will be notified of material changes by email with at
        least 14 days' notice. Continued use of the Service after that date
        constitutes acceptance of the updated Terms.
      </P>

      <H2>11. Governing law</H2>
      <P>
        These Terms are governed by Polish law. Any disputes shall be resolved
        by the court having jurisdiction over the Provider's place of residence,
        unless mandatory consumer protection laws provide otherwise.
      </P>
      <P>
        Consumers have the right to use out-of-court dispute resolution —
        see ec.europa.eu/consumers/odr for more information.
      </P>

      <H2>12. Contact</H2>
      <P>For questions about these Terms: {CONTACT_EMAIL}</P>
    </>
  )
}