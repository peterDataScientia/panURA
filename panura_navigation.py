"""Responsive inline navigation with an isolated Streamlit Components v2 frontend."""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

PAGES = {"molecular": "Molecular Prediction", "batch": "Batch Screening", "methods": "Model & Methods"}

NAV_HTML = """
<div class="panura-shell">
  <header class="panura-bar">
    <img class="panura-logo" alt="PanURA — urease inhibitor QSAR platform">
    <button class="panura-toggle" type="button" aria-expanded="false" aria-controls="panura-nav-links" aria-label="Open navigation menu">
      <span aria-hidden="true">☰</span><span> Menu</span>
    </button>
  </header>
  <nav id="panura-nav-links" class="panura-menu" aria-label="Main navigation" hidden>
    <a href="?view=molecular" data-page="molecular">Molecular Prediction</a>
    <a href="?view=batch" data-page="batch">Batch Screening</a>
    <a href="?view=methods" data-page="methods">Model &amp; Methods</a>
  </nav>
</div>
"""
NAV_CSS = """
.panura-shell{width:100%;position:relative;font-family:system-ui,-apple-system,sans-serif}
.panura-bar{width:100%;min-height:76px;display:flex;align-items:center;justify-content:space-between;gap:12px;box-sizing:border-box}
.panura-logo{display:block;max-width:min(240px,calc(100% - 112px));max-height:75px;width:auto;height:auto;object-fit:contain}
.panura-toggle{flex:0 0 auto;min-width:96px;min-height:46px;padding:10px 14px;cursor:pointer;border-radius:12px;border:1px solid #0c707c;background:linear-gradient(110deg,#173a70,#117c88);color:white;font-size:15px;font-weight:750}
.panura-toggle:hover,.panura-toggle[aria-expanded="true"]{background:linear-gradient(110deg,#24568e,#119c94)}
button:focus-visible{outline:3px solid #34c8ba;outline-offset:3px}
.panura-menu{margin:4px 0 10px auto;box-sizing:border-box;width:min(320px,100%);padding:9px;display:grid;gap:5px;border:1px solid #d8e5ef;border-radius:14px;background:#fff;box-shadow:0 12px 30px #0c27472c}
.panura-menu[hidden]{display:none}
.panura-menu a{display:block;box-sizing:border-box;text-decoration:none;width:100%;text-align:left;background:#fff;border:0;border-radius:9px;padding:12px 13px;min-height:46px;color:#183459;font-weight:650;cursor:pointer;font-size:14px}
.panura-menu a:hover,.panura-menu a:focus-visible{background:#e8f5f5;color:#066d70}
.panura-menu a[aria-current="page"]{background:#e2f3f3;border-left:3px solid #159d99}
@media(max-width:480px){.panura-bar{min-height:66px}.panura-logo{max-width:min(195px,calc(100% - 102px));max-height:64px}.panura-toggle{min-height:44px;min-width:91px;padding:8px 10px;font-size:13px}}
@media(prefers-reduced-motion:no-preference){.panura-menu{animation:panura-open .16s ease-out}@keyframes panura-open{from{opacity:0;transform:translateY(-5px)}to{opacity:1;transform:translateY(0)}}}
"""
NAV_JS = """
export default function({parentElement, data}) {
    const root = parentElement.querySelector(".panura-shell");
    const logo = root.querySelector(".panura-logo");
    const toggle = root.querySelector(".panura-toggle");
    const menu = root.querySelector(".panura-menu");
    const links = [...menu.querySelectorAll("a[data-page]")];
    logo.src = data.logo;
    const close = (restoreFocus = false) => {
        menu.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
        toggle.setAttribute("aria-label", "Open navigation menu");
        if (restoreFocus) toggle.focus();
    };
    const open = () => {
        menu.hidden = false;
        toggle.setAttribute("aria-expanded", "true");
        toggle.setAttribute("aria-label", "Close navigation menu");
    };
    toggle.onclick = () => menu.hidden ? open() : close();
    for (const link of links) {
        const selected = link.getAttribute("data-page");
        if (selected === data.page) link.setAttribute("aria-current", "page");
        else link.removeAttribute("aria-current");
        link.onclick = (event) => {
            event.preventDefault();
            close();
            // Force a top-level navigation rather than relying on an anchor
            // default action from inside a Streamlit component shadow tree.
            const target = new URL(window.location.href);
            target.searchParams.set("view", selected);
            window.location.assign(target.toString());
        };
    }
    const outside = (event) => {
        if (!menu.hidden && !root.contains(event.target)) close();
    };
    const keyboard = (event) => {
        if (menu.hidden) return;
        if (event.key === "Escape") {
            event.preventDefault();
            close(true);
        } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            const index = links.indexOf(document.activeElement);
            if (index >= 0 || document.activeElement === toggle) {
                event.preventDefault();
                const next = event.key === "ArrowDown" ? index + 1 : index - 1;
                links[(next + links.length) % links.length].focus();
            }
        }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", keyboard);
    return () => {
        document.removeEventListener("pointerdown", outside);
        document.removeEventListener("keydown", keyboard);
    };
}
"""
WIDTH_JS = """
export default function({parentElement, data, setStateValue}) {
    const shadowRoot = parentElement.getRootNode();
    const host = shadowRoot.host ?? parentElement;
    const column = host.closest?.('[data-testid="stColumn"]');
    const read = () => {
        const columnWidth = column?.getBoundingClientRect().width
            || host.getBoundingClientRect().width
            || parentElement.getBoundingClientRect().width;
        const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
        const measured = Math.floor(Math.min(columnWidth - 8, viewportWidth - 26));
        if (Number.isFinite(measured) && measured > 160 &&
            Math.abs(measured - (Number(data.known) || 0)) >= 8) {
            setStateValue("available", measured);
        }
    };
    const observer = new ResizeObserver(read);
    observer.observe(column || host);
    window.addEventListener("resize", read, {passive:true});
    read();
    return () => {
        observer.disconnect();
        window.removeEventListener("resize", read);
    };
}
"""
# Register only once per app process, not once per individual widget rerun.
navigation_component = st.components.v2.component(
    "panura_hamburger", html=NAV_HTML, css=NAV_CSS, js=NAV_JS
)
width_component = st.components.v2.component(
    "panura_width_probe", html="<div aria-hidden='true' style='height:1px'></div>",
    js=WIDTH_JS,
)


def render_navigation(logo_path: Path) -> str:
    """Render accessible top-right menu with bookmarkable page links.

    Real browser navigation avoids transient Streamlit v2 component events
    that can be lost during reruns of the surrounding molecular editor.
    """
    section = str(st.query_params.get("view", "molecular")).lower()
    if section not in PAGES:
        section = "molecular"
    logo_data = base64.b64encode(Path(logo_path).read_bytes()).decode("ascii")
    navigation_component(
        data={"logo": "data:image/svg+xml;base64," + logo_data, "page": section},
        key="panura_header", width="stretch", height="content",
    )
    return PAGES[section]


def available_editor_width() -> int:
    """Measure the editor column using ResizeObserver, not CSS viewport guesses."""
    known = getattr(st.session_state.get("panura_probe"), "available", 270)
    result = width_component(
        data={"known": known}, default={"available": 270},
        key="panura_probe", width="stretch", height=1,
        on_available_change=lambda: None,
    )
    measured = result.available if isinstance(result.available, (int, float)) else 270
    return max(180, min(700, int(measured)))
