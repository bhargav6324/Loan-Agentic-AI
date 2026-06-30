import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import AgentPipeline from '../components/AgentPipeline'

const statusBadge = (status) => {
  const cls = `badge badge-${status}`
  const icons = { submitted: '📩', processing: '⏳', approved: '✅', rejected: '❌', escalated: '⚠️' }
  return <span className={cls}>{icons[status] || '●'} {status}</span>
}

export default function Dashboard() {
  const [apps, setApps] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [processing, setProcessing] = useState(null)
  const user = api.getUser()

  useEffect(() => {
    loadApps()
  }, [])

  const loadApps = async () => {
    try {
      const data = await api.getApplications()
      setApps(data.applications || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const processApp = async (id) => {
    setProcessing(id)
    setError('')
    try {
      await api.processApplication(id)
      await loadApps()
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(null)
    }
  }

  const stats = {
    total: apps.length,
    approved: apps.filter(a => a.status === 'approved').length,
    rejected: apps.filter(a => a.status === 'rejected').length,
    pending: apps.filter(a => ['submitted', 'processing'].includes(a.status)).length,
  }

  if (loading) {
    return <div className="loading-page"><div className="spinner"></div><p>Loading dashboard...</p></div>
  }

  return (
    <div className="page-content animate-in">
      <div className="page-header">
        <h1>👋 Welcome, {user?.full_name || 'User'}</h1>
        <p>Manage your loan applications and track their status</p>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📋</div>
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Applications</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">✅</div>
          <div className="stat-value">{stats.approved}</div>
          <div className="stat-label">Approved</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">❌</div>
          <div className="stat-value">{stats.rejected}</div>
          <div className="stat-label">Rejected</div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">⏳</div>
          <div className="stat-value">{stats.pending}</div>
          <div className="stat-label">Pending</div>
        </div>
      </div>

      {/* Applications List */}
      <div className="section-header mt-8">
        <h2 className="text-2xl text-primary font-serif">Active Applications</h2>
        <Link to="/apply" className="btn btn-primary btn-sm">+ New Application</Link>
      </div>

      {apps.length === 0 ? (
        <div className="card empty-state">
          <div className="empty-icon">📝</div>
          <h3 className="text-primary text-lg">No Applications Yet</h3>
          <p className="text-secondary">Submit your first loan application to get started</p>
          <Link to="/apply" className="btn btn-primary mt-4">Apply for a Loan</Link>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {apps.map((app, index) => {
            const ss = app.shared_state || {};
            // Determine how many agents have executed based on available summaries
            let agentsExecuted = 0;
            if (ss.data_agent_summary) agentsExecuted = 1;
            if (ss.risk_agent_summary) agentsExecuted = 2;
            if (ss.policy_agent_summary) agentsExecuted = 3;
            if (ss.fairness_agent_summary) agentsExecuted = 4;

            return (
              <div key={app.id} className={`card card-static animate-fade-up delay-${(index % 4) + 1}`}>
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h3 className="text-xl text-primary font-serif">{app.loan_purpose} Loan</h3>
                    <div className="text-sm text-secondary font-mono tracking-widest uppercase mt-1">
                      Applicant: {app.applicant_name}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <div className="data-number text-xl">₹{Number(app.loan_amount).toLocaleString('en-IN')}</div>
                    {statusBadge(app.status)}
                  </div>
                </div>

                <div className="mb-6">
                  <AgentPipeline agentsExecuted={agentsExecuted} status={app.status} sharedState={ss} />
                </div>

                <div className="flex justify-between items-center border-t border-[var(--border-glass)] pt-4 mt-4">
                  <div className="text-xs text-muted font-mono tracking-widest uppercase">
                    Applied: {new Date(app.created_at).toLocaleDateString('en-IN')}
                  </div>
                  <div className="flex gap-4">
                    {app.status === 'submitted' && (
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => processApp(app.id)}
                        disabled={processing === app.id}
                      >
                        {processing === app.id ? '⏳ PROCESSING...' : '🚀 PROCESS WITH AI'}
                      </button>
                    )}
                    <Link to={`/applications/${app.id}`} className="btn btn-outline btn-sm">VIEW DETAILS →</Link>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  )
}
