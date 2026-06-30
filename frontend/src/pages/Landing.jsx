import { Link } from 'react-router-dom'

const features = [
  { icon: '🤖', title: 'Multi-Agent AI', desc: '6 specialised AI agents collaborate to analyse every application from every angle — credit, risk, policy, fairness.' },
  { icon: '⚖️', title: 'Built-in Fairness', desc: 'Active counterfactual bias detection ensures no demographic discrimination in loan decisions.' },
  { icon: '🛡️', title: 'Regulatory Compliance', desc: 'Policy-as-Code engine enforces RBI rules automatically. Immutable audit trail for every decision.' },
  { icon: '📊', title: 'Explainable Decisions', desc: '3-audience explanations: simple for applicants, technical for reviewers, comprehensive for regulators.' },
  { icon: '⚡', title: 'Instant Processing', desc: 'High-confidence applications approved in seconds. Borderline cases intelligently escalated to humans.' },
  { icon: '🔒', title: 'Zero-Trust Security', desc: 'JWT authentication, input sanitisation, rate limiting, and secure file scanning on every request.' },
]

export default function Landing() {
  return (
    <div>
      <section className="hero animate-in">
        <h1>Autonomous Loan<br />Approval System</h1>
        <p>
          A multi-agent AI architecture that processes loan applications with
          fairness, transparency, and regulatory compliance — from submission to decision.
        </p>
        <div className="hero-actions">
          <Link to="/register" className="btn btn-primary">Get Started</Link>
          <Link to="/login" className="btn btn-outline">Login →</Link>
        </div>
      </section>

      <section className="features-grid">
        {features.map((f, i) => (
          <div key={i} className={`feature-card animate-in delay-${(i % 4) + 1}`}>
            <div className="feature-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p>{f.desc}</p>
          </div>
        ))}
      </section>
    </div>
  )
}
