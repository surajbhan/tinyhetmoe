"use strict";

// ── Setup marked ─────────────────────────────────────────────────────
function configureMarked() {
  if (typeof marked === "undefined") return;
  marked.setOptions({
    gfm: true,
    breaks: false,
    headerIds: true,
  });
}

// ── Blog (post list + post viewer) ──────────────────────────────────
function initBlog(posts) {
  configureMarked();
  const listEl = document.getElementById("post-list");
  const contentEl = document.getElementById("post-content");

  function showList() {
    listEl.classList.remove("hidden");
    contentEl.classList.add("hidden");
    contentEl.innerHTML = "";
    listEl.innerHTML = "";
    for (const p of posts) {
      const card = document.createElement("a");
      card.className = "post-card";
      card.href = `?post=${encodeURIComponent(p.slug)}`;
      card.innerHTML = `
        <div class="post-card-date">${p.date}</div>
        <h2 class="post-card-title">${escapeHtml(p.title)}</h2>
        <p class="post-card-summary">${escapeHtml(p.summary)}</p>
        <span class="post-card-arrow">read →</span>
      `;
      card.addEventListener("click", (e) => {
        e.preventDefault();
        history.pushState({ slug: p.slug }, "", `?post=${encodeURIComponent(p.slug)}`);
        loadPost(p.slug);
      });
      listEl.appendChild(card);
    }
  }

  async function loadPost(slug) {
    const post = posts.find(p => p.slug === slug);
    if (!post) {
      showList();
      return;
    }
    listEl.classList.add("hidden");
    contentEl.classList.remove("hidden");
    contentEl.innerHTML = `<div class="loading">Loading "${escapeHtml(post.title)}"…</div>`;

    try {
      const url = `https://raw.githubusercontent.com/surajbhan/tinyhetmoe/main/blog/${slug}.md`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Could not load post (HTTP ${res.status})`);
      const md = await res.text();
      contentEl.innerHTML = `
        <div class="post-meta">
          <a href="?" class="post-back">← all posts</a>
          <span class="post-date">${post.date}</span>
        </div>
        ${marked.parse(md)}
      `;
      contentEl.querySelector(".post-back")?.addEventListener("click", (e) => {
        e.preventDefault();
        history.pushState({}, "", "?");
        showList();
      });
      window.scrollTo(0, 0);
    } catch (err) {
      contentEl.innerHTML = `<p class="error">Could not load post: ${escapeHtml(err.message)}</p>`;
    }
  }

  // Route based on URL
  const params = new URLSearchParams(location.search);
  const slug = params.get("post");
  if (slug) loadPost(slug); else showList();

  window.addEventListener("popstate", () => {
    const p = new URLSearchParams(location.search);
    const s = p.get("post");
    if (s) loadPost(s); else showList();
  });
}

// ── Journal (single long markdown, no list view) ────────────────────
async function initJournal() {
  configureMarked();
  const el = document.getElementById("post-content");
  try {
    const url = "https://raw.githubusercontent.com/surajbhan/tinyhetmoe/main/JOURNAL.md";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load journal (HTTP ${res.status})`);
    const md = await res.text();
    el.innerHTML = marked.parse(md);
    // Build a quick floating TOC from the H2s for navigation
    addJournalTOC(el);
  } catch (err) {
    el.innerHTML = `<p class="error">Could not load journal: ${escapeHtml(err.message)}</p>`;
  }
}

function addJournalTOC(contentEl) {
  const h2s = contentEl.querySelectorAll("h2");
  if (h2s.length < 4) return;
  const toc = document.createElement("aside");
  toc.className = "toc";
  toc.innerHTML = `<div class="toc-title">on this page</div>`;
  const list = document.createElement("ul");
  for (const h2 of h2s) {
    if (!h2.id) {
      // Generate a slug for anchor linking
      const slug = h2.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
      h2.id = slug;
    }
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `#${h2.id}`;
    a.textContent = h2.textContent;
    li.appendChild(a);
    list.appendChild(li);
  }
  toc.appendChild(list);
  document.querySelector(".prose-main").appendChild(toc);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

window.initBlog = initBlog;
window.initJournal = initJournal;
