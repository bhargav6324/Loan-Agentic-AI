import { Routes, Route, Navigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { api } from './api'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Login from './pages/Login'
import Register from './pages/Register'
import Dashboard from './pages/Dashboard'
import ApplyLoan from './pages/ApplyLoan'
import ApplicationDetail from './pages/ApplicationDetail'
import AdminDashboard from './pages/AdminDashboard'

function ProtectedRoute({ children, adminOnly = false, userOnly = false }) {
  const user = api.getUser()
  if (!api.isLoggedIn() || !user) return <Navigate to="/login" />
  if (adminOnly && user.role !== 'admin') return <Navigate to="/dashboard" />
  if (userOnly && user.role === 'admin') return <Navigate to="/admin" />
  return children
}

export default function App() {
  const [user, setUser] = useState(api.getUser())

  const handleLogin = (userData) => {
    setUser(userData)
  }

  const handleLogout = () => {
    api.logout()
    setUser(null)
  }

  const getHomeRoute = () => user?.role === 'admin' ? '/admin' : '/dashboard'

  return (
    <div className="app-container">
      <Navbar user={user} onLogout={handleLogout} />
      <Routes>
        <Route path="/" element={user ? <Navigate to={getHomeRoute()} /> : <Landing />} />
        <Route path="/login" element={
          user ? <Navigate to={getHomeRoute()} /> : <Login onLogin={handleLogin} />
        } />
        <Route path="/register" element={
          user ? <Navigate to={getHomeRoute()} /> : <Register />
        } />
        <Route path="/dashboard" element={
          <ProtectedRoute userOnly><Dashboard /></ProtectedRoute>
        } />
        <Route path="/apply" element={
          <ProtectedRoute userOnly><ApplyLoan /></ProtectedRoute>
        } />
        <Route path="/applications/:id" element={
          <ProtectedRoute userOnly><ApplicationDetail /></ProtectedRoute>
        } />
        <Route path="/admin" element={
          <ProtectedRoute adminOnly><AdminDashboard /></ProtectedRoute>
        } />
      </Routes>
    </div>
  )
}
