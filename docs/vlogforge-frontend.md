# VlogForge — New Screen/Component Prompt

> **Copy-paste this prompt when asking the AI to build a new screen or component for VlogForge. It contains every design token, pattern, and convention needed — no `@frontend` context required.**

---

## Prompt (copy everything below this line)

---

I need you to build a new screen/component for the **VlogForge** frontend. Follow these design rules **exactly** to match the existing app aesthetic.

### 1. Stack & Dependencies (DO NOT add new ones)
- **React 18** (functional components, hooks only)
- **Vite** dev server
- **lucide-react** for all icons — import only what you need from `lucide-react`
- **react-dropzone** (only if file upload is needed)
- **No Tailwind**. All styling is done via CSS classes defined in the global `index.css`, or inline `style={{}}` objects on JSX elements (the codebase mixes both).

### 2. Color Palette (CSS custom properties — reference via `var(--token)`)

| Token | Value | Usage |
|---|---|---|
| `--bg-color` | `#1A1A1A` | Page / app background |
| `--card-bg` | `#222222` | Card & panel backgrounds |
| `--card-border` | `#333333` | Default border color |
| `--card-border-glow` | `rgba(212, 255, 0, 0.3)` | Hover-glow border accent |
| `--primary` | `#D4FF00` (neon lime) | Primary action / brand color |
| `--primary-glow` | `rgba(212, 255, 0, 0.4)` | Primary glow (box-shadow, filter) |
| `--secondary` | `#06B6D4` (cyan) | Secondary accent / data highlights |
| `--secondary-glow` | `rgba(6, 182, 212, 0.4)` | Secondary glow |
| `--accent` | `#D4FF00` | Alias for primary (used on playhead, badges) |
| `--text-main` | `#FFFFFF` | Primary text |
| `--text-muted` | `#A1A1AA` | Secondary / descriptive text |
| `--text-disabled` | `#52525B` | Disabled / placeholder text |
| `--success` | `#10B981` | Success states |
| `--warning` | `#F59E0B` | Warning states |
| `--danger` | `#EF4444` | Error / destructive states |
| `--sidebar-bg` | `#1F1F1F` | Sidebar panel background |
| `--header-bg` | `#1A1A1A` | Header bar background |
| `--tab-group-bg` | `#141414` | Tab group / recessed panel backgrounds |

### 3. Typography
- **Body font**: `'Inter', system-ui, -apple-system, sans-serif` (already imported via Google Fonts)
- **Headings**: `'Outfit', sans-serif` — `font-weight: 700` or `800`
- Section titles use `<h2>` at `1.4rem`, paired with a lucide icon at `size={18}` and a `<p>` subtitle at `0.82rem` in `var(--text-muted)`.
- Labels: `0.75rem`, `font-weight: 500`, `color: var(--text-muted)`, often uppercase with `letter-spacing: 0.03em`.

### 4. Spacing & Radius
| Token | Value |
|---|---|
| `--radius-lg` | `16px` |
| `--radius-md` | `8px` |
| `--radius-sm` | `4px` |
| Panel padding | `2rem 2.5rem` |
| Section gap | `1.25rem` — `1.5rem` |
| Element gap | `0.5rem` — `0.75rem` |

### 5. Layout Shell
The app wraps everything in this structure — **your new screen renders inside `<main className="dashboard-content">`**:

```jsx
<div className="dashboard-layout">
  <div className="dashboard-main">
    <header className="dashboard-header"> ... </header>
    <main className="dashboard-content">
      {/* YOUR NEW SCREEN GOES HERE */}
    </main>
  </div>
</div>
```

For a **two-column split** layout (like the current setup screen), use:
```jsx
<div className="setup-split fade-in">
  <div className="setup-panel-left"> ... </div>
  <div className="setup-divider" />
  <div className="setup-panel-right"> ... </div>
</div>
```

### 6. Reusable CSS Classes

| Class | Purpose |
|---|---|
| `.fade-in` | Entry animation: `slideUpFade 0.7s` (blur-resolve slide up) |
| `.glass-card` | Glassmorphism card with border, shadow, slide-up animation |
| `.btn` | Base button reset |
| `.btn-primary` | Neon-lime filled button, dark text `#111111`, glow shadow. Hover: `translateY(-1px)`, brighter. Disabled: `--btn-disabled-bg` / `--btn-disabled-color` |
| `.btn-secondary` | Ghost button: `rgba(255,255,255,0.05)` bg, 1px border |
| `.input-field` | Text input / textarea: dark bg, `--card-border`, `--radius-md`. Focus: `border-color: var(--primary)`, purple glow ring |
| `.dropzone` | Dashed border upload zone with hover glow |
| `.file-item` | List row for files: recessed bg, slide-in anim |
| `.stage-card` | Processing stage card (`.active` / `.completed` variants) |
| `.spinner` | `animation: spin 1.5s linear infinite` for `<Loader2>` icon |

### 7. Animation Keyframes Available
- `fadeInUp` — simple opacity + translateY(20px→0)
- `slideUpFade` — Apple-style slide up with blur resolve (the primary entry anim)
- `pulseGlow` — box-shadow pulse
- `gradientMove` — background-position cycle
- `spin` / `spinCcw` — rotation
- `cascadeIn` — stagger-friendly translateY + scaleX for lists

### 8. Interaction Patterns
- **Hover lift**: `transform: translateY(-1px)` + stronger `box-shadow`
- **Active press**: `transform: translateY(0) scale(0.98)`
- **Transition**: `all 0.3s cubic-bezier(0.4, 0, 0.2, 1)` or `var(--transition)`
- **Expo ease**: `var(--ease-out-expo)` = `cubic-bezier(0.16, 1, 0.3, 1)` for UI entries
- **Border glow on hover**: `border-color: var(--card-border-glow)`
- **Scrollbars**: Styled thin dark, thumb glows `var(--primary)` on hover

### 9. Component Conventions
- Export one default function per `.jsx` file in `src/components/`
- Props are destructured in the function signature
- Inline styles (`style={{}}`) for one-off layout tweaks; CSS classes for shared patterns
- Icons from `lucide-react` are sized `14–20`, colored via `style={{ color: 'var(--primary)' }}` or `var(--text-muted)`
- Section headers follow this pattern:
  ```jsx
  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
    <IconName size={18} style={{ color: 'var(--primary)', filter: 'drop-shadow(0 0 4px var(--primary-glow))' }} />
    <h2 style={{ margin: 0, fontSize: '1.4rem', fontFamily: 'Outfit, sans-serif', fontWeight: 700 }}>
      Section Title
    </h2>
  </div>
  <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', margin: 0, lineHeight: 1.45 }}>
    Subtitle describing the section.
  </p>
  ```

### 10. Dark-Mode-Only
The app is locked to dark mode. The `[data-theme="light"]` selector exists but is intentionally empty. Do not add light mode overrides.

### 11. File Placement
- Component: `frontend/src/components/YourComponent.jsx`
- Styles: Add any new CSS classes to `frontend/src/index.css` (append at the bottom, use a `/* ═══ SECTION NAME ═══ */` banner comment)
- Integration: Import and render in `frontend/src/App.jsx`

---

**Now build: [DESCRIBE YOUR NEW SCREEN HERE]**
