import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api } from '../api'
import RiskGauge from '../components/RiskGauge'
import Accordion from '../components/Accordion'

const statusBadge = (status) => {
  const cls = `badge badge-${status}`
  const icons = { submitted: '📩', processing: '⏳', approved: '✅', rejected: '❌', escalated: '⚠️' }
  return <span className={cls}>{icons[status] || '●'} {status}</span>
}



export default function ApplicationDetail() {
  const { id } = useParams()
  const [app, setApp] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [processing, setProcessing] = useState(false)
  const [streamLogs, setStreamLogs] = useState([])

  useEffect(() => {
    loadApp()
  }, [id])

  const loadApp = async () => {
    try {
      const data = await api.getApplication(id)
      setApp(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const processApp = async () => {
    setProcessing(true)
    setError('')
    setStreamLogs([])
    
    // Simulate updating the status immediately in UI
    setApp(prev => ({ ...prev, status: 'processing' }))

    try {
      await api.processApplicationStream(
        id,
        (msg) => {
          setStreamLogs(prev => [...prev, msg])
        },
        (errMsg) => {
          setError(errMsg)
          setProcessing(false)
          setApp(prev => ({ ...prev, status: 'submitted' }))
        },
        async (completeData) => {
          await loadApp()
          setProcessing(false)
        }
      )
    } catch (err) {
      setError(err.message)
      setProcessing(false)
      setApp(prev => ({ ...prev, status: 'submitted' }))
    }
  }

  if (loading) return <div className="loading-page"><div className="spinner"></div><p>Loading application...</p></div>
  if (error && !app) return <div className="page-content"><div className="alert alert-error">⚠️ {error}</div><Link to="/dashboard" className="btn btn-outline">← Back</Link></div>
  if (!app) return null

  const ss = app.shared_state || {}
  const hasResults = !!ss.risk_score



  return (
    <div className="page-content animate-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
        <div>
          <Link to="/dashboard" style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>← Back to Dashboard</Link>
          <h1 style={{ marginTop: '0.5rem' }}>Application Details</h1>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          {app.status === 'submitted' && (
            <button className="btn btn-primary btn-sm" onClick={processApp} disabled={processing}>
              {processing ? '🧠 AI is thinking...' : '🚀 Check My Application'}
            </button>
          )}
          {statusBadge(app.status)}
        </div>
      </div>

      {error && <div className="alert alert-error">⚠️ {error}</div>}



      {/* ── Top Decision Card ── */}
      <div className={`decision-card mb-8 animate-fade-up ${app.status === 'submitted' || app.status === 'processing' ? '' : app.status}`} style={{ marginTop: '2rem' }}>
        {app.status === 'submitted' ? (
          <>
            <div className="text-5xl mb-4">⏳</div>
            <h1 className="text-text-primary">Awaiting Processing</h1>
            <p className="text-secondary font-mono tracking-widest text-sm uppercase">Application submitted and ready for AI review</p>
          </>
        ) : app.status === 'processing' ? (
          <>
            <div className="text-5xl mb-4 pulse-active">🧠</div>
            <h1 className="text-text-primary pulse-active" style={{ border: 'none', background: 'transparent' }}>AI IS THINKING</h1>
            <p className="text-secondary font-mono tracking-widest text-sm uppercase mb-6">Multi-agent pipeline reasoning in progress...</p>
            
            {/* Live Streaming Terminal */}
            {streamLogs.length > 0 && (
              <div className="text-left bg-[#0f172a] border border-[var(--border-glass)] rounded-md p-4 w-full max-w-2xl mx-auto shadow-inner overflow-hidden flex flex-col font-mono text-sm h-48">
                <div className="flex justify-between items-center mb-2 pb-2 border-b border-[var(--border-glass)]">
                  <span className="text-xs text-muted uppercase tracking-widest">Live Execution Trace</span>
                  <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald animate-pulse"></div><span className="text-xs text-emerald">STREAMING</span></span>
                </div>
                <div className="flex-1 overflow-y-auto flex flex-col gap-2 p-1" style={{ scrollBehavior: 'smooth' }}>
                  {streamLogs.map((log, i) => (
                    <div key={i} className="animate-fade-up flex gap-3 text-secondary">
                      <span className="text-muted">[{new Date().toLocaleTimeString('en-US', {hour12:false})}]</span>
                      <span className={log.event === 'started' ? 'text-accent' : 'text-primary'}>{log.message}</span>
                    </div>
                  ))}
                  {/* Scroll to bottom anchor */}
                  <div style={{ float:"left", clear: "both" }}
                       ref={(el) => { el?.scrollIntoView({ behavior: 'smooth' }) }}>
                  </div>
                </div>
              </div>
            )}
          </>
        ) : app.status === 'approved' ? (
          <>
            <h1 className="text-emerald drop-shadow-lg">APPROVED</h1>
            <p className="text-secondary font-mono tracking-widest text-sm uppercase">Agentic pipeline authorized this loan</p>
          </>
        ) : app.status === 'rejected' ? (
          <>
            <h1 className="text-crimson drop-shadow-lg">REJECTED</h1>
            <p className="text-secondary font-mono tracking-widest text-sm uppercase">Application failed risk or policy thresholds</p>
          </>
        ) : (
          <>
            <h1 className="text-amber drop-shadow-lg">ESCALATED</h1>
            <p className="text-secondary font-mono tracking-widest text-sm uppercase">Flagged for manual human review</p>
          </>
        )}
      </div>

      {/* ── Decision Explanation Card ── */}
      {hasResults && ss.explanation && (
        <div className="card mb-8 animate-fade-up border-l-4 flex flex-col gap-6" style={{ 
          borderLeftColor: app.status === 'approved' ? 'var(--success)' : app.status === 'rejected' ? 'var(--danger)' : 'var(--warning)',
          background: 'var(--bg-secondary)',
          padding: '2rem'
        }}>
          <div>
            <h2 className="text-xl mb-4 text-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: 0 }}>
              <span>🧠</span> Applicant Explanation
            </h2>
            <div style={{ 
              whiteSpace: 'pre-wrap', 
              lineHeight: '1.7', 
              fontSize: '0.975rem',
              color: 'var(--text-secondary)',
              fontFamily: 'system-ui, -apple-system, sans-serif'
            }}>
              {ss.explanation.applicant_message}
            </div>
          </div>
          
          {ss.explanation.process_trace && (
            <div className="pt-4 border-t border-[var(--border-glass)]">
              <h2 className="text-lg mb-4 text-warning" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span>🔍</span> Auditor Process Trace (LLM Log Analysis)
              </h2>
              <div className="bg-[rgba(0,0,0,0.2)] p-4 rounded font-mono text-sm border-l-2 border-warning" style={{ 
                whiteSpace: 'pre-wrap', 
                lineHeight: '1.6', 
                color: 'var(--text-muted)'
              }}>
                {ss.explanation.process_trace}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="grid grid-cols-3 gap-6 mb-8 animate-fade-up delay-1">
        {/* Loan Info */}
        <div className="card card-static col-span-2 flex flex-col justify-between">
          <div>
            <h2 className="text-xl mb-4 text-primary">Application Data</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-muted uppercase tracking-widest block mb-1">Applicant Name</label>
                <div className="text-lg text-primary">{app.applicant_name}</div>
              </div>
              <div>
                <label className="text-xs text-muted uppercase tracking-widest block mb-1">PAN Number</label>
                <div className="font-mono text-lg text-primary">{app.applicant_pan || '—'}</div>
              </div>
              <div className="mt-2">
                <label className="text-xs text-muted uppercase tracking-widest block mb-1">Loan Amount</label>
                <div className="data-number">₹{Number(app.loan_amount).toLocaleString('en-IN')}</div>
              </div>
              <div className="mt-2">
                <label className="text-xs text-muted uppercase tracking-widest block mb-1">Tenure</label>
                <div className="data-number">{app.loan_tenure_months} months</div>
              </div>
              <div className="mt-2">
                <label className="text-xs text-muted uppercase tracking-widest block mb-1">Applied On</label>
                <div className="data-number">{new Date(app.created_at).toLocaleDateString('en-IN')}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Risk Gauge */}
        <div className="card card-static flex flex-col items-center justify-center">
          <h2 className="text-xl mb-2 text-primary text-center w-full">Risk Assessment</h2>
          {hasResults ? (
            <RiskGauge score={ss.risk_score} category={ss.risk_category} />
          ) : (
            <div className="flex flex-col items-center justify-center text-muted h-32 w-full border border-dashed border-[var(--border-glass)] rounded-md mt-4">
              <span className="text-2xl mb-2">📊</span>
              <span className="font-mono text-xs uppercase tracking-widest">Pending execution</span>
            </div>
          )}
        </div>
      </div>

      {/* Agent Analysis */}
      {hasResults && (
        <div className="animate-fade-up delay-2">
          <h2 className="text-2xl mb-4 text-primary">Agent Telemetry & Thinking</h2>
          
          <Accordion title={`Planner Agent (${ss.plan?.length || 0} steps)`} icon="🧭" defaultOpen={true} status="success">
            <div className="font-mono text-sm space-y-4">
              <div>
                <p className="text-secondary font-bold mb-2 flex items-center justify-between">
                  <span>Autonomous Execution Plan</span>
                  <span className={`text-xs px-2 py-0.5 rounded ${ss.plan_reasoning?.includes('Fallback') ? 'bg-[rgba(245,158,11,0.15)] text-amber' : 'bg-[rgba(16,185,129,0.15)] text-emerald'}`}>
                    {ss.plan_reasoning?.includes('Fallback') ? '⚠ FALLBACK' : '✦ LLM-GENERATED'}
                  </span>
                </p>
                <div className="flex gap-1 flex-wrap items-center">
                  {ss.plan?.map((step, i) => {
                    const count = ss.plan.filter(s => s === step).length
                    return (
                      <span key={i} className="flex items-center gap-1">
                        {i > 0 && <span className="text-muted text-xs">→</span>}
                        <span className="px-2 py-1 bg-[var(--bg-primary)] border border-[var(--border-glass)] rounded text-xs text-primary flex items-center gap-1">
                          <span className="text-muted">{i + 1}.</span> {step}
                          {count > 1 && <span className="ml-1 px-1 bg-[rgba(139,92,246,0.2)] text-[#a78bfa] rounded text-[10px] font-bold">×{count}</span>}
                        </span>
                      </span>
                    )
                  })}
                </div>
              </div>
              {ss.plan_reasoning && (
                <div className="bg-[rgba(0,0,0,0.2)] p-3 rounded border-l-2 border-[var(--accent)]">
                  <p className="text-xs text-muted uppercase tracking-widest mb-1 flex justify-between">
                    <span>LLM Planning Rationale</span>
                    <span className="text-accent">Llama-3.3-70b</span>
                  </p>
                  <p className="text-primary italic mt-2" style={{ whiteSpace: 'pre-wrap' }}>"{ss.plan_reasoning}"</p>
                </div>
              )}
            </div>
          </Accordion>

          <Accordion title="Data Agent" icon="📄" defaultOpen={true} status="success">
            <div className="font-mono text-sm space-y-4">
              <div>
                <p><strong>Status:</strong> Completed Data Extraction</p>
                <p><strong>Summary:</strong> {ss.data_agent_summary}</p>
                {ss.credit_report && <p><strong>CIBIL Extracted:</strong> {ss.credit_report.credit_score}</p>}
              </div>
              
              {ss.data_agent_reasoning_trace && ss.data_agent_reasoning_trace.length > 0 && (
                <div className="mt-4 border border-[var(--border-glass)] rounded overflow-hidden">
                  <div className="bg-[var(--bg-primary)] px-3 py-2 border-b border-[var(--border-glass)] flex justify-between items-center">
                    <span className="text-xs text-muted uppercase tracking-widest">ReAct Loop Trace</span>
                    <span className="text-xs text-accent font-bold">Llama-3.1-8b</span>
                  </div>
                  <div className="p-3 space-y-4 bg-[rgba(0,0,0,0.1)]">
                    {ss.data_agent_reasoning_trace.map((trace, i) => (
                      <div key={i} className="pl-4 border-l-2 border-[var(--border-glass)] relative">
                        <div className="absolute w-2.5 h-2.5 rounded-full bg-[var(--accent)] -left-[6px] top-1" style={{ boxShadow: '0 0 8px var(--accent-glow)' }}></div>
                        <p className="text-secondary text-xs mb-1 font-bold uppercase tracking-widest">ReAct Step {trace.iteration}</p>
                        <p className="text-primary mb-3"><em>"{trace.thought}"</em></p>
                        <div className="flex items-center gap-2 text-xs">
                          <span className="px-2 py-1 bg-[var(--bg-primary)] border border-[var(--border-glass)] text-[var(--accent)] rounded font-bold">ACTION: {trace.action}</span>
                          <span className="text-muted">→</span>
                          <span className="text-emerald">{trace.observation}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Accordion>

          
          <Accordion title="Fraud Agent" icon="🛡️" defaultOpen={true} status={ss.fraud_detected ? 'danger' : ss.requires_manual_kyc ? 'warning' : 'success'}>
            <div className="font-mono text-sm space-y-4">
              <div>
                <p><strong>Status:</strong> {ss.fraud_detected ? 'Fraud Detected' : ss.requires_manual_kyc ? 'Manual KYC Required' : 'Clean'}</p>
                <p><strong>Summary:</strong> {ss.fraud_agent_summary || 'No summary available'}</p>
                {ss.fraud_reason && <p className="text-crimson mt-2"><strong>Reason:</strong> {ss.fraud_reason}</p>}
              </div>
              
              <div className="grid grid-cols-2 gap-4 border-t border-[var(--border-glass)] pt-4 mt-2">
                <div className="bg-[rgba(0,0,0,0.1)] p-3 rounded border border-[var(--border-glass)]">
                  <p className="text-xs text-muted uppercase tracking-widest mb-2">PAN Verification</p>
                  {ss.pan_verification ? (
                    <ul className="text-xs space-y-1 text-secondary">
                      <li>Verified: {ss.pan_verification.verified ? '✅ Yes' : '❌ No'}</li>
                      <li>Name Match: {ss.pan_verification.name_match ? '✅ Yes' : '❌ No'}</li>
                      <li>Name on PAN: {ss.pan_verification.registered_name}</li>
                    </ul>
                  ) : <p className="text-xs text-muted">No data</p>}
                </div>
                <div className="bg-[rgba(0,0,0,0.1)] p-3 rounded border border-[var(--border-glass)]">
                  <p className="text-xs text-muted uppercase tracking-widest mb-2">Aadhaar Verification</p>
                  {ss.aadhaar_verification ? (
                    <ul className="text-xs space-y-1 text-secondary">
                      <li>Verified: {ss.aadhaar_verification.verified ? '✅ Yes' : '❌ No'}</li>
                      <li>Name Match: {ss.aadhaar_verification.name_match ? '✅ Yes' : '❌ No'}</li>
                      <li>Name on Aadhaar: {ss.aadhaar_verification.registered_name}</li>
                    </ul>
                  ) : <p className="text-xs text-muted">No data</p>}
                </div>
              </div>
            </div>
          </Accordion>

          <Accordion title="Risk Agent" icon="📊" defaultOpen={true} status={ss.risk_category === 'LOW' ? 'success' : ss.risk_category === 'HIGH' ? 'danger' : 'warning'}>
            <div className="font-mono text-sm space-y-2">
              <p><strong>Composite Score:</strong> {ss.risk_score} ({ss.risk_category})</p>
              <p><strong>Summary:</strong> {ss.risk_agent_summary}</p>
              {ss.risk_factors && (
                <div className="mt-2">
                  <strong>Key Factors:</strong>
                  <ul className="list-disc pl-5 mt-1 text-secondary">
                    {ss.risk_factors.map((f, i) => <li key={i}>{f}</li>)}
                  </ul>
                </div>
              )}
              
              {ss.risk_agent_reflection && ss.risk_agent_reflection.reflection_summary && (
                 <div className="mt-4 bg-[rgba(139,92,246,0.1)] border-l-2 border-[#8b5cf6] p-3 rounded-r text-xs">
                    <p className="text-[#8b5cf6] font-bold uppercase tracking-widest mb-1 flex justify-between">
                      <span>LLM Self-Reflection</span>
                      <span>Adjusted: {ss.risk_agent_reflection.score_adjustment > 0 ? '+' : ''}{ss.risk_agent_reflection.score_adjustment} pts</span>
                    </p>
                    <p className="text-primary italic">"{ss.risk_agent_reflection.reflection_summary}"</p>
                 </div>
              )}
            </div>
          </Accordion>

          <Accordion title="Policy Agent" icon="📜" defaultOpen={true} status={ss.policy_violations?.length ? 'danger' : 'success'}>
            <div className="font-mono text-sm space-y-2">
              <p><strong>Status:</strong> {ss.policy_violations?.length ? 'Violations Found' : 'All Checks Passed'}</p>
              <p><strong>Summary:</strong> {ss.policy_agent_summary}</p>
              
              {ss.policy_rag_reasoning && (
                <div className="mt-4 bg-[rgba(0,0,0,0.2)] p-3 rounded border-l-2 border-[var(--accent)] text-xs">
                  <p className="text-muted uppercase tracking-widest mb-1 flex justify-between">
                    <span>RAG Policy Enforcement</span>
                    <span className="text-accent">Llama-3.1-8b</span>
                  </p>
                  <p className="text-primary italic mt-2" style={{ whiteSpace: 'pre-wrap' }}>"{ss.policy_rag_reasoning}"</p>
                </div>
              )}
            </div>
          </Accordion>

          <Accordion title="Fairness Agent" icon="⚖️" defaultOpen={true} status={ss.fairness_report?.bias_detected ? 'danger' : 'success'}>
            <div className="font-mono text-sm space-y-4">
              <div>
                <p><strong>Status:</strong> {ss.fairness_report?.bias_detected ? 'Bias Detected (Escalating)' : 'No Demographic Bias Found'}</p>
                <p><strong>Summary:</strong> {ss.fairness_agent_summary}</p>
              </div>
              
              {ss.fairness_report?.counterfactual_results?.filter(r => r.bias_flag).length > 0 && (
                <div className="p-3 bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)] rounded">
                  <strong className="text-crimson">Failed Simulations:</strong>
                  <ul className="list-disc pl-5 mt-1 text-secondary">
                    {ss.fairness_report.counterfactual_results.filter(r => r.bias_flag).map((r, i) => (
                      <li key={i}>Flipping {r.attribute} to {r.counterfactual_value} changed score by {r.score_difference} pts</li>
                    ))}
                  </ul>
                </div>
              )}

              {ss.fairness_debate_log && ss.fairness_debate_log.length > 0 && (
                <div className="mt-4 border border-[var(--border-glass)] rounded overflow-hidden">
                  <div className="bg-[var(--bg-primary)] px-3 py-2 border-b border-[var(--border-glass)] flex justify-between items-center">
                    <span className="text-xs text-muted uppercase tracking-widest">Multi-Turn Debate Log</span>
                    <span className="text-xs text-warning font-bold">Risk vs Fairness</span>
                  </div>
                  <div className="p-3 space-y-5 bg-[rgba(0,0,0,0.1)]">
                    {ss.fairness_debate_log.map((turn, i) => (
                      <div key={i} className="space-y-3">
                        {/* Fairness Challenge */}
                        <div className="bg-[rgba(245,158,11,0.05)] border border-[rgba(245,158,11,0.2)] p-3 rounded-lg rounded-tl-none w-5/6 shadow-sm">
                          <p className="text-xs text-amber font-bold mb-2 flex items-center gap-2"><span>⚖️</span> Fairness Agent (Challenge)</p>
                          <p className="text-secondary text-sm">Flipping <span className="text-primary font-bold">{turn.attribute_tested}</span> to <span className="text-primary font-bold">{turn.counterfactual_value}</span> caused a score change of <span className="text-crimson font-bold">{turn.score_delta}</span> pts.</p>
                        </div>
                        {/* Risk Response */}
                        <div className="bg-[rgba(16,185,129,0.05)] border border-[rgba(16,185,129,0.2)] p-3 rounded-lg rounded-tr-none w-5/6 ml-auto shadow-sm">
                          <p className="text-xs text-emerald font-bold mb-2 flex items-center justify-end gap-2">Risk Agent (Response) <span>📊</span></p>
                          <p className="text-primary text-sm italic text-right">"{turn.risk_agent_justification}"</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Accordion>
        
          <Accordion title="Audit Agent" icon="🔐" defaultOpen={true} status={ss.audit_anomalies?.length ? 'danger' : 'success'}>
            <div className="font-mono text-sm space-y-4">
              <div>
                <p><strong>Status:</strong> {ss.audit_anomalies?.length ? 'Anomalies Detected' : 'Cryptographic Audit Passed'}</p>
                <p><strong>Summary:</strong> {ss.audit_agent_summary || 'No summary available'}</p>
              </div>

              {ss.audit_anomalies && ss.audit_anomalies.length > 0 && (
                <div className="p-3 bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)] rounded mt-2">
                  <strong className="text-crimson">Security Anomalies:</strong>
                  <ul className="list-disc pl-5 mt-1 text-secondary text-xs">
                    {ss.audit_anomalies.map((anom, i) => <li key={i}>{anom}</li>)}
                  </ul>
                </div>
              )}

              {ss.audit_trail && ss.audit_trail.length > 0 && (
                <div className="mt-4 border border-[var(--border-glass)] rounded overflow-hidden">
                   <div className="bg-[var(--bg-primary)] px-3 py-2 border-b border-[var(--border-glass)] text-xs text-muted uppercase tracking-widest">Cryptographic SHA-256 Trail</div>
                   <div className="p-0 bg-[#0f172a]">
                     <table className="w-full text-left text-xs">
                       <thead className="bg-[rgba(0,0,0,0.3)] text-muted">
                         <tr>
                           <th className="px-3 py-2 font-normal">Agent</th>
                           <th className="px-3 py-2 font-normal">Action</th>
                           <th className="px-3 py-2 font-normal">Hash</th>
                         </tr>
                       </thead>
                       <tbody className="text-secondary divide-y divide-[var(--border-glass)]">
                         {ss.audit_trail.map((log, i) => (
                           <tr key={i} className="hover:bg-[rgba(255,255,255,0.02)]">
                             <td className="px-3 py-2">{log.agent}</td>
                             <td className="px-3 py-2">{log.action_type}</td>
                             <td className="px-3 py-2 font-mono text-emerald truncate max-w-[150px]">{log.hash.substring(0, 16)}...</td>
                           </tr>
                         ))}
                       </tbody>
                     </table>
                   </div>
                </div>
              )}
            </div>
          </Accordion>

          <Accordion title="Execution Timeline (Logs)" icon="⏱️" defaultOpen={false} status="neutral">
            <div className="font-mono text-sm space-y-4">
              <p><strong>Full Agent Execution Trace</strong></p>
              
              {ss.agent_logs && ss.agent_logs.length > 0 ? (
                <div className="border border-[var(--border-glass)] rounded overflow-hidden bg-[#0f172a] p-3">
                   <div className="space-y-3 relative before:absolute before:inset-0 before:ml-2 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-700 before:to-transparent">
                     {ss.agent_logs.map((log, i) => (
                        <div key={i} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                          <div className="flex items-center justify-center w-5 h-5 rounded-full border border-slate-700 bg-slate-900 text-slate-500 shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 shadow">
                            <div className="w-2 h-2 rounded-full bg-[var(--accent)]"></div>
                          </div>
                          <div className="w-[calc(100%-2.5rem)] md:w-[calc(50%-1.25rem)] p-3 rounded border border-slate-800 bg-slate-800/50">
                            <div className="flex justify-between mb-1">
                               <span className="font-bold text-primary text-xs">{log.agent}</span>
                               <span className="text-xs text-muted">{log.duration_ms}ms</span>
                            </div>
                            <p className="text-secondary text-xs">{log.action}</p>
                            {log.details && <p className="text-muted text-[10px] mt-1 border-t border-slate-700/50 pt-1">"{log.details.substring(0, 100)}{log.details.length > 100 ? '...' : ''}"</p>}
                          </div>
                        </div>
                     ))}
                   </div>
                </div>
              ) : <p className="text-muted">No logs recorded.</p>}
            </div>
          </Accordion>

</div>
      )}
    </div>
  )
}
