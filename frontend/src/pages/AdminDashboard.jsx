import { useState, useEffect } from 'react'
import { api } from '../api'

const statusBadge = (status) => {
  const cls = `badge badge-${status}`
  const icons = { submitted: '📩', processing: '⏳', approved: '✅', rejected: '❌', escalated: '⚠️' }
  return <span className={cls}>{icons[status] || '●'} {status}</span>
}

export default function AdminDashboard() {
  const [apps, setApps] = useState([])
  const [escalations, setEscalations] = useState([])
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [tab, setTab] = useState('all') // 'all', 'escalated', 'users'

  // Resolve modal
  const [resolving, setResolving] = useState(null)
  const [notes, setNotes] = useState('')
  const [resolveLoading, setResolveLoading] = useState(false)
  const [resolveSuccess, setResolveSuccess] = useState('')

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [allApps, escData, allUsers] = await Promise.all([
        api.getAllApplications(),
        api.getEscalations(),
        api.getAllUsers(),
      ])
      setApps(allApps.applications || [])
      setEscalations(escData.applications || [])
      setUsers(allUsers.users || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleResolve = async (decision) => {
    setResolveLoading(true)
    try {
      await api.resolveEscalation(resolving.id, decision, notes)
      setResolveSuccess(`Application ${decision === 'approved' ? 'approved' : 'rejected'} successfully`)
      setResolving(null)
      setNotes('')
      loadData()
      setTimeout(() => setResolveSuccess(''), 3000)
    } catch (err) {
      setError(err.message)
    } finally {
      setResolveLoading(false)
    }
  }

  const stats = {
    total: apps.length,
    approved: apps.filter(a => a.status === 'approved').length,
    rejected: apps.filter(a => a.status === 'rejected').length,
    escalated: escalations.length,
    totalUsers: users.length,
  }

  if (loading) return <div className="loading-page"><div className="spinner"></div><p>Loading admin panel...</p></div>

  const displayApps = tab === 'escalated' ? escalations : apps

  return (
    <div className="page-content animate-in">
      <div className="page-header">
        <h1>🛡️ Admin Dashboard</h1>
        <p>System-wide application management and review queue</p>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}
      {resolveSuccess && <div className="alert alert-success">✅ {resolveSuccess}</div>}

      {/* Stats */}
      <div className="stats-grid animate-fade-up">
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Volume</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon text-emerald">✅</div>
          <div className="stat-value text-emerald">{stats.approved}</div>
          <div className="stat-label">System Approvals</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon text-crimson">❌</div>
          <div className="stat-value text-crimson">{stats.rejected}</div>
          <div className="stat-label">System Rejections</div>
        </div>
        <div className="stat-card" style={{ borderColor: stats.escalated > 0 ? 'var(--warning)' : 'var(--border-glass)' }}>
          <div className="stat-icon text-amber">⚠️</div>
          <div className="stat-value text-amber">{stats.escalated}</div>
          <div className="stat-label">Pending Review</div>
        </div>
      </div>

      {/* Kanban Board */}
      <div className="grid grid-cols-3 gap-6 animate-fade-up delay-1">
        {/* Column 1: Pending Escalations */}
        <div className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-glass)] p-4 flex flex-col h-[600px]">
          <h2 className="text-lg text-primary font-serif mb-4 flex justify-between items-center border-b border-[var(--border-glass)] pb-2">
            Pending Escalations
            <span className="badge badge-escalated">{escalations.filter(a => resolving?.id !== a.id).length}</span>
          </h2>
          <div className="flex-1 overflow-y-auto pr-2 space-y-4">
            {escalations.filter(a => resolving?.id !== a.id).map(app => (
              <div key={app.id} className="card card-static p-4 border-l-4" style={{ borderLeftColor: 'var(--warning)' }}>
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-primary font-serif text-lg">{app.applicant_name}</h3>
                  <div className="font-mono text-sm text-primary bg-[rgba(245,158,11,0.15)] px-2 py-1 rounded text-amber">
                    {app.shared_state?.risk_score?.toFixed(1) || 'N/A'}
                  </div>
                </div>
                <div className="text-xs text-muted font-mono uppercase tracking-widest mb-4">
                  {app.loan_purpose} • ₹{Number(app.loan_amount).toLocaleString('en-IN')}
                </div>
                <button 
                  className="btn btn-outline btn-sm w-full text-xs"
                  onClick={() => { setResolving(app); setNotes(''); }}
                >
                  START REVIEW →
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Column 2: Under Review */}
        <div className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-glass)] p-4 flex flex-col h-[600px] shadow-[0_0_20px_rgba(0,212,255,0.05)]">
          <h2 className="text-lg text-primary font-serif mb-4 flex justify-between items-center border-b border-[var(--border-glass)] pb-2">
            Under Review
            <span className="badge badge-processing">{resolving ? '1' : '0'}</span>
          </h2>
          <div className="flex-1 overflow-y-auto pr-2">
            {!resolving ? (
              <div className="flex flex-col items-center justify-center h-full text-muted border-2 border-dashed border-[var(--border-glass)] rounded-md">
                <span className="text-3xl mb-2">🔍</span>
                <span className="font-mono text-xs uppercase tracking-widest">Select application</span>
              </div>
            ) : (
              <div className="card card-static p-4 border-l-4 pulse-active" style={{ borderLeftColor: 'var(--accent)' }}>
                <h3 className="text-primary font-serif text-lg mb-2">{resolving.applicant_name}</h3>
                <div className="text-xs text-secondary font-mono mb-4 border-b border-[var(--border-glass)] pb-4">
                  <div>Amount: ₹{Number(resolving.loan_amount).toLocaleString('en-IN')}</div>
                  <div>Purpose: {resolving.loan_purpose}</div>
                  <div>Risk Score: {resolving.shared_state?.risk_score} ({resolving.shared_state?.risk_category})</div>
                </div>
                
                <div className="form-group mb-4">
                  <label className="text-xs text-accent">Compliance Officer Notes</label>
                  <textarea 
                    className="form-textarea w-full text-sm font-mono mt-1 border-[var(--accent-glow)] focus:border-[var(--accent)]" 
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)} 
                    placeholder="Enter manual review justification..." 
                    rows={4}
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex gap-2">
                    <button className="btn btn-danger btn-sm flex-1" disabled={resolveLoading} onClick={() => handleResolve('rejected')}>
                      ❌ REJECT
                    </button>
                    <button className="btn btn-success btn-sm flex-1" disabled={resolveLoading} onClick={() => handleResolve('approved')}>
                      ✅ APPROVE
                    </button>
                  </div>
                  <button className="btn btn-outline btn-sm w-full" onClick={() => setResolving(null)}>
                    CANCEL
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Column 3: Resolved */}
        <div className="bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-glass)] p-4 flex flex-col h-[600px] opacity-80">
          <h2 className="text-lg text-primary font-serif mb-4 flex justify-between items-center border-b border-[var(--border-glass)] pb-2">
            Resolved
            <span className="badge badge-submitted">{apps.filter(a => a.status === 'approved' || a.status === 'rejected').length}</span>
          </h2>
          <div className="flex-1 overflow-y-auto pr-2 space-y-4">
            {apps.filter(a => a.status === 'approved' || a.status === 'rejected').map(app => (
              <div key={app.id} className="card card-static p-3 opacity-75 hover:opacity-100 transition-opacity border-l-4" style={{ borderLeftColor: app.status === 'approved' ? 'var(--success)' : 'var(--danger)' }}>
                <div className="flex justify-between items-start mb-1">
                  <h3 className="text-primary font-serif text-base">{app.applicant_name}</h3>
                  {statusBadge(app.status)}
                </div>
                <div className="text-[10px] text-muted font-mono uppercase tracking-widest">
                  ₹{Number(app.loan_amount).toLocaleString('en-IN')}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
