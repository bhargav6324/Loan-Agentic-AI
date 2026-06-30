import { Link } from 'react-router-dom'

export default function Navbar({ user, onLogout }) {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        <div className="logo-icon">🏦</div>
        Loan AI
      </Link>

      <div className="navbar-links">
        {user ? (
          <>
            {user.role === 'user' && (
              <>
                <Link to="/dashboard" className="nav-link">Dashboard</Link>
                <Link to="/apply" className="nav-link">Apply</Link>
              </>
            )}
            {user.role === 'admin' && (
              <Link to="/admin" className="nav-link">Admin</Link>
            )}
            <span className="nav-link" style={{ color: 'var(--text-muted)', cursor: 'default' }}>
              {user.full_name}
            </span>
            <button onClick={onLogout} className="nav-link logout">Logout</button>
          </>
        ) : (
          <>
            <Link to="/login" className="nav-link">Login</Link>
            <Link to="/register" className="btn btn-primary btn-sm">Get Started</Link>
          </>
        )}
      </div>
    </nav>
  )
}
