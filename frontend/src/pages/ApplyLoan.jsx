import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'

export default function ApplyLoan() {
  const navigate = useNavigate()
  const [step, setStep] = useState(1) // 1: Personal, 2: Financial, 3: Documents
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [appId, setAppId] = useState(null)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const [form, setForm] = useState({
    loan_amount: '',
    loan_purpose: '',
    loan_tenure_months: 12,
    applicant_name: '',
    applicant_age: '',
    applicant_gender: '',
    applicant_email: '',
    applicant_phone: '',
    applicant_address: '',
    applicant_pan: '',
    applicant_aadhaar: '',
    monthly_income: '',
    employment_type: '',
    employer_name: '',
  })

  const handleChange = (e) => {
    let val = e.target.type === 'number' ? (e.target.value === '' ? '' : Number(e.target.value)) : e.target.value
    if (e.target.name === 'applicant_pan') val = val.toUpperCase()
    setForm({ ...form, [e.target.name]: val })
  }

  const handleNextToFinancial = (e) => {
    e.preventDefault()
    setStep(2)
  }

  const handleBack = () => {
    setStep(step - 1)
  }

  const submitApplication = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const payload = { ...form }
      Object.keys(payload).forEach(k => {
        if (payload[k] === '') payload[k] = undefined
      })
      const result = await api.submitApplication(payload)
      setAppId(result.id)
      setStep(3) // Move to documents
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError('')
    try {
      await api.uploadDocument(appId, file)
      setUploadSuccess(true)
      setTimeout(() => navigate(`/applications/${appId}`), 2000)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.currentTarget.classList.add('drag-over')
  }
  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('drag-over')
  }
  const handleDrop = (e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('drag-over')
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0])
  }

  return (
    <div className="page-content animate-in flex flex-col items-center">
      <div className="w-full max-w-2xl mb-8 sticky top-20 z-50 bg-[var(--bg-primary)] pt-4 pb-4 border-b border-[var(--border-glass)]">
        <h1 className="text-3xl font-serif text-primary text-center mb-6">Loan Application</h1>
        <div className="flex justify-between items-center relative">
          <div className="absolute left-0 top-1/2 w-full h-1 bg-[var(--border-glass)] -z-10 transform -translate-y-1/2" />
          <div className="absolute left-0 top-1/2 h-1 bg-accent transition-all duration-500 -z-10 transform -translate-y-1/2" style={{ width: step === 1 ? '0%' : step === 2 ? '50%' : '100%' }} />
          
          {[1, 2, 3].map(s => (
            <div key={s} className={`flex items-center justify-center rounded-full w-10 h-10 transition-all duration-300 ${step >= s ? 'bg-[var(--bg-secondary)] border border-accent text-accent shadow-[0_0_10px_rgba(0,212,255,0.3)]' : 'bg-[var(--bg-primary)] border border-[var(--border-glass)] text-muted'}`}>
              {s === 1 ? '👤' : s === 2 ? '💰' : '📄'}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 font-mono text-xs uppercase tracking-widest text-secondary">
          <span className={step >= 1 ? 'text-accent' : ''}>Personal</span>
          <span className={step >= 2 ? 'text-accent' : ''}>Financial</span>
          <span className={step >= 3 ? 'text-accent' : ''}>Documents</span>
        </div>
      </div>

      {error && <div className="alert alert-error w-full max-w-2xl mb-4">⚠️ {error}</div>}

      <div className="w-full max-w-2xl">
        {/* ── Step 1: Personal Details ── */}
        {step === 1 && (
          <div className="card animate-fade-up">
            <form onSubmit={handleNextToFinancial}>
              <h2 className="text-xl mb-6 text-primary">Personal Details</h2>
              <div className="form-row">
                <div className="form-group">
                  <label>Full Name *</label>
                  <input type="text" name="applicant_name" className="form-input" placeholder="Rahul Sharma"
                    value={form.applicant_name} onChange={handleChange} required />
                </div>
                <div className="form-group">
                  <label>Age</label>
                  <input type="number" name="applicant_age" className="form-input" placeholder="28"
                    value={form.applicant_age} onChange={handleChange} min="18" max="100" />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Gender</label>
                  <select name="applicant_gender" className="form-select" value={form.applicant_gender} onChange={handleChange}>
                    <option value="">Select</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Phone</label>
                  <input type="text" name="applicant_phone" className="form-input" placeholder="9876543210"
                    value={form.applicant_phone} onChange={handleChange} />
                </div>
              </div>
              <div className="form-group">
                <label>Address</label>
                <textarea name="applicant_address" className="form-textarea" placeholder="Full postal address..."
                  value={form.applicant_address} onChange={handleChange} />
              </div>

              <div className="flex justify-end mt-6">
                <button type="submit" className="btn btn-primary">
                  Next Step →
                </button>
              </div>
            </form>
          </div>
        )}

        {/* ── Step 2: Financial Details ── */}
        {step === 2 && (
          <div className="card animate-fade-up">
            <form onSubmit={submitApplication}>
              <h2 className="text-xl mb-6 text-primary">Loan Request & Financials</h2>
              
              <div className="form-row">
                <div className="form-group">
                  <label>Loan Amount (₹) *</label>
                  <input type="number" name="loan_amount" className="form-input" placeholder="500000"
                    value={form.loan_amount} onChange={handleChange} required min="10000" max="50000000" />
                </div>
                <div className="form-group">
                  <label>Purpose *</label>
                  <input type="text" name="loan_purpose" className="form-input" placeholder="Home purchase"
                    value={form.loan_purpose} onChange={handleChange} required />
                </div>
              </div>
              <div className="form-group">
                <label>Tenure (Months)</label>
                <select name="loan_tenure_months" className="form-select" value={form.loan_tenure_months} onChange={handleChange}>
                  {[6, 12, 24, 36, 48, 60, 84, 120, 180, 240, 360].map(m => (
                    <option key={m} value={m}>{m} months ({(m/12).toFixed(0)} years)</option>
                  ))}
                </select>
              </div>

              <h3 className="text-lg mt-6 mb-4 text-primary font-serif">Identity & Income</h3>
              <div className="form-row">
                <div className="form-group">
                  <label>PAN Number</label>
                  <input type="text" name="applicant_pan" className="form-input uppercase" placeholder="ABCDE1234F"
                    value={form.applicant_pan} onChange={handleChange} />
                </div>
                <div className="form-group">
                  <label>Aadhaar Number</label>
                  <input type="text" name="applicant_aadhaar" className="form-input" placeholder="123456789012"
                    value={form.applicant_aadhaar} onChange={handleChange} />
                </div>
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Monthly Income (₹)</label>
                  <input type="number" name="monthly_income" className="form-input" placeholder="85000"
                    value={form.monthly_income} onChange={handleChange} />
                </div>
                <div className="form-group">
                  <label>Employment Type</label>
                  <select name="employment_type" className="form-select" value={form.employment_type} onChange={handleChange}>
                    <option value="">Select</option>
                    <option value="salaried">Salaried</option>
                    <option value="self_employed">Self Employed</option>
                    <option value="business">Business Owner</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label>Employer / Business Name</label>
                <input type="text" name="employer_name" className="form-input" placeholder="TCS, Infosys, etc."
                  value={form.employer_name} onChange={handleChange} />
              </div>

              <div className="flex justify-between mt-6">
                <button type="button" className="btn btn-outline" onClick={handleBack}>
                  ← Back
                </button>
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? <><div className="spinner"></div> Submitting...</> : 'Submit Application →'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* ── Step 3: Upload PDF ── */}
        {step === 3 && (
          <div className="card animate-fade-up">
            {uploadSuccess ? (
              <div className="empty-state">
                <div className="empty-icon text-5xl mb-4">✅</div>
                <h3 className="text-xl text-primary font-serif">Application Submitted!</h3>
                <p className="text-secondary font-mono tracking-widest text-sm uppercase">Redirecting to telemetry dashboard...</p>
              </div>
            ) : (
              <>
                <h2 className="text-xl mb-2 text-primary font-serif">📎 Document Ingestion</h2>
                <p className="text-secondary mb-6 text-sm">
                  Upload your last 6 months bank statement. Our Data Agent will ingest and analyze it automatically.
                </p>

                <div className="upload-zone"
                  onDragOver={handleDrag}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('fileInput').click()}
                >
                  <div className="upload-icon text-4xl mb-4">📤</div>
                  {file ? (
                    <p className="text-primary font-mono tracking-widest text-sm uppercase">📄 {file.name} ({(file.size / 1024).toFixed(0)} KB)</p>
                  ) : (
                    <>
                      <p className="mb-2">Drag and drop your PDF here, or click to browse</p>
                      <p className="upload-hint font-mono text-xs uppercase tracking-widest">Only PDF files, max 10MB</p>
                    </>
                  )}
                  <input type="file" id="fileInput" accept=".pdf" style={{ display: 'none' }}
                    onChange={(e) => setFile(e.target.files[0])} />
                </div>

                <div className="flex gap-4 mt-6">
                  <button className="btn btn-outline" onClick={() => navigate(`/applications/${appId}`)}>
                    Skip Upload
                  </button>
                  <button className="btn btn-primary flex-1" onClick={handleUpload} disabled={!file || uploading}>
                    {uploading ? <><div className="spinner"></div> Uploading...</> : 'Upload & Proceed'}
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
