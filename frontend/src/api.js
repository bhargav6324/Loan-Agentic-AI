/**
 * API Helper — Centralized API calls with JWT auth
 */

const API_BASE = '';  // Uses Vite proxy in dev

class ApiClient {
  constructor() {
    this.token = localStorage.getItem('access_token');
  }

  setToken(token) {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  setRefreshToken(token) {
    localStorage.setItem('refresh_token', token);
  }

  clearTokens() {
    this.token = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  }

  getUser() {
    const u = localStorage.getItem('user');
    return u ? JSON.parse(u) : null;
  }

  setUser(user) {
    localStorage.setItem('user', JSON.stringify(user));
  }

  isLoggedIn() {
    return !!this.token;
  }

  async request(path, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...options.headers };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

      if (res.status === 401) {
        this.clearTokens();
        window.location.href = '/login';
        throw new Error('Session expired');
      }

      const data = await res.json();

      if (!res.ok) {
        let errMsg = data.detail || 'Something went wrong';
        if (Array.isArray(data.detail)) {
          errMsg = data.detail.map(d => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join(' | ');
        }
        throw new Error(errMsg);
      }
      return data;
    } catch (err) {
      if (err.message === 'Failed to fetch') {
        throw new Error('Server is offline. Please start the backend.');
      }
      throw err;
    }
  }

  // Auth
  async register(email, password, fullName, role = 'user') {
    const data = await this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, full_name: fullName, role }),
    });
    return data;
  }

  async login(email, password) {
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    this.setToken(data.access_token);
    this.setRefreshToken(data.refresh_token);

    const user = await this.getMe();
    this.setUser(user);
    return user;
  }

  async getMe() {
    return this.request('/auth/me');
  }

  // Applications
  async submitApplication(applicationData) {
    return this.request('/applications/', {
      method: 'POST',
      body: JSON.stringify(applicationData),
    });
  }

  async getApplications() {
    return this.request('/applications/');
  }

  async getApplication(id) {
    return this.request(`/applications/${id}`);
  }

  async processApplication(id) {
    return this.request(`/applications/${id}/process`, { method: 'POST' });
  }

  async processApplicationStream(id, onMessage, onError, onComplete) {
    const headers = { 'Content-Type': 'application/json' };
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const res = await fetch(`${API_BASE}/applications/${id}/process_stream`, {
        method: 'POST',
        headers,
      });

      if (!res.ok) {
        throw new Error('Stream failed to start');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          onComplete({ event: 'complete' });
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          
          if (chunk.startsWith('data: ')) {
            try {
              const data = JSON.parse(chunk.substring(6));
              if (data.event === 'error') {
                onError(data.detail);
                return;
              } else if (data.event === 'complete') {
                onComplete(data);
                return;
              } else {
                onMessage(data);
              }
            } catch (e) {
              console.error("Failed to parse SSE JSON", e, chunk);
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
      }
    } catch (err) {
      onError(err.message);
    }
  }

  async uploadDocument(applicationId, file) {
    const formData = new FormData();
    formData.append('file', file);

    const headers = {};
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`;

    const res = await fetch(`${API_BASE}/applications/${applicationId}/upload`, {
      method: 'POST',
      headers,
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    return data;
  }

  // Admin
  async getAllApplications() {
    return this.request('/admin/applications');
  }

  async getAllUsers() {
    return this.request('/admin/users');
  }

  async getEscalations() {
    return this.request('/admin/escalations');
  }

  async resolveEscalation(applicationId, decision, notes) {
    return this.request(`/admin/applications/${applicationId}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ decision, reviewer_notes: notes }),
    });
  }

  async getAuditTrail(applicationId) {
    return this.request(`/admin/applications/${applicationId}/audit`);
  }

  logout() {
    this.clearTokens();
    window.location.href = '/';
  }
}

export const api = new ApiClient();
