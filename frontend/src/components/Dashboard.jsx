import React, { useState, useEffect } from 'react';
import { Plus, Video, Trash2, Loader2, Sparkles } from 'lucide-react';

export default function Dashboard({ onNewProject, onOpenProject, onLogout }) {
  const [projects, setProjects] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const token = localStorage.getItem('vlogforge_token');
      const res = await fetch(`/api/projects/?t=${Date.now()}`, {
        headers: { 'Authorization': `Bearer ${token}` },
        cache: 'no-store'
      });
      if (res.ok) {
        const data = await res.json();
        setProjects(data);
      }
    } catch (err) {
      console.error("Failed to fetch projects", err);
    } finally {
      setIsLoading(false);
    }
  };



  const deleteProject = async (e, id) => {
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this project?')) return;
    
    try {
      const token = localStorage.getItem('vlogforge_token');
      await fetch(`/api/projects/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      fetchProjects();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="dashboard-layout">
      {/* Sidebar (Collapsed 72px) */}
      <aside className="dashboard-sidebar">
        <div style={{ marginBottom: '2rem' }}>
          <Sparkles size={28} style={{ color: 'var(--primary)', filter: 'drop-shadow(0 0 8px var(--primary-glow))' }} />
        </div>
        
        <div className="dashboard-sidebar-item active" title="Projects">
          <Video size={22} />
        </div>
        
        <div style={{ flex: 1 }}></div>

        <div className="dashboard-sidebar-item" onClick={onLogout} title="Logout" style={{ marginTop: 'auto', marginBottom: 0 }}>
          <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
        </div>
      </aside>

      {/* Main Column */}
      <div className="dashboard-main">
        {/* Header */}
        <header className="dashboard-header">
          <div className="app-logo">
            VlogForge Studio
          </div>
          <button className="btn btn-primary" onClick={onNewProject}>
            <Plus size={18} /> New Project
          </button>
        </header>

        {/* Content */}
        <main className="dashboard-content" style={{ padding: '2rem', overflowY: 'auto' }}>
          <div style={{ marginBottom: '2rem' }}>
            <h2 style={{ margin: 0, fontSize: '1.75rem', fontFamily: 'Outfit, sans-serif' }}>Your Projects</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.25rem' }}>Manage and create your vlogs</p>
          </div>

          {isLoading ? (
            <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
              <Loader2 size={32} className="spinner" style={{ color: 'var(--primary)' }} />
            </div>
          ) : projects.length === 0 ? (
            <div style={{ background: 'var(--card-bg)', border: '1px dashed var(--card-border)', borderRadius: 'var(--radius-lg)', padding: '4rem 2rem', textAlign: 'center' }}>
              <Video size={48} style={{ color: 'var(--text-disabled)', margin: '0 auto 1rem' }} />
              <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>No projects yet</h3>
              <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>Create your first AI-edited vlog to get started.</p>
              <button className="btn btn-primary" onClick={onNewProject}>
                Create Project
              </button>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.5rem' }}>
              {projects.map(p => (
                <div 
                  key={p.id} 
                  onClick={() => onOpenProject(p)}
                  style={{ 
                    background: 'var(--card-bg)', 
                    border: '1px solid var(--card-border)', 
                    borderRadius: 'var(--radius-lg)', 
                    padding: '1.5rem',
                    cursor: 'pointer',
                    transition: 'transform 0.2s, box-shadow 0.2s, border-color 0.2s',
                    position: 'relative'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 10px 25px rgba(0,0,0,0.2)';
                    e.currentTarget.style.borderColor = 'var(--card-border-glow)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                    e.currentTarget.style.borderColor = 'var(--card-border)';
                  }}
                >
                  <button 
                    onClick={(e) => deleteProject(e, p.id)}
                    style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'transparent', border: 'none', color: 'var(--text-disabled)', cursor: 'pointer', padding: '0.25rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    onMouseEnter={(e) => e.currentTarget.style.color = '#ef4444'}
                    onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-disabled)'}
                    title="Delete Project"
                  >
                    <Trash2 size={18} />
                  </button>
                  <div style={{ width: '40px', height: '40px', borderRadius: '8px', background: 'rgba(109, 40, 217, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                    <Video size={20} style={{ color: 'var(--primary)' }} />
                  </div>
                  <h3 style={{ margin: '0 0 0.5rem 0', fontSize: '1.1rem', fontWeight: 600 }}>{p.title}</h3>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span>{new Date(p.created_at).toLocaleDateString()}</span>
                    {(() => {
                      let text = 'In Progress';
                      let bg = 'rgba(59, 130, 246, 0.1)';
                      let color = '#3b82f6';
                      
                      if (p.status === 'pending') {
                        text = 'Pending';
                        bg = 'rgba(245, 158, 11, 0.1)';
                        color = 'var(--warning)';
                      } else if (p.status === 'complete') {
                        text = 'Completed';
                        bg = 'rgba(16, 185, 129, 0.1)';
                        color = 'var(--success)';
                      } else if (p.status === 'cancelled') {
                        text = 'Terminated';
                        bg = 'rgba(239, 68, 68, 0.1)';
                        color = 'var(--danger)';
                      } else if (p.status === 'failed') {
                        text = 'Cancelled'; // User definition for failure/error
                        bg = 'rgba(239, 68, 68, 0.1)';
                        color = 'var(--danger)';
                      }

                      return (
                        <span style={{ 
                          padding: '0.25rem 0.6rem', 
                          borderRadius: '4px', 
                          background: bg,
                          color: color,
                          fontWeight: 600,
                          textTransform: 'uppercase',
                          letterSpacing: '0.05em',
                          fontSize: '0.7rem'
                        }}>
                          {text}
                        </span>
                      );
                    })()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
