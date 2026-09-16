const noteSources = {
  zh: 'https://raw.githubusercontent.com/SamZebrado/hf-llm-agents-companion/main/notes/zh-CN.md',
  en: 'https://raw.githubusercontent.com/SamZebrado/hf-llm-agents-companion/main/notes/en.md'
};

const notes = document.getElementById('notes');
const toc = document.getElementById('toc');
const langButtons = [...document.querySelectorAll('[data-lang]')];
let currentLanguage = 'zh';

function applyInterfaceLanguage(lang) {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-zh][data-en]').forEach((el) => {
    el.textContent = lang === 'zh' ? el.dataset.zh : el.dataset.en;
  });
  langButtons.forEach((button) => {
    const active = button.dataset.lang === lang;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

function slugify(text, index) {
  const base = text
    .toLowerCase()
    .replace(/<[^>]+>/g, '')
    .replace(/[^\p{L}\p{N}]+/gu, '-')
    .replace(/^-+|-+$/g, '');
  return base || `section-${index}`;
}

function buildToc() {
  toc.innerHTML = '';
  const headings = [...notes.querySelectorAll('h2, h3')];
  const used = new Set();

  headings.forEach((heading, index) => {
    let id = slugify(heading.textContent, index);
    let suffix = 2;
    while (used.has(id)) {
      id = `${id}-${suffix++}`;
    }
    used.add(id);
    heading.id = id;

    const link = document.createElement('a');
    link.href = `#${id}`;
    link.textContent = heading.textContent;
    link.className = heading.tagName === 'H3' ? 'level-3' : 'level-2';
    toc.appendChild(link);
  });
}

async function loadNotes(lang) {
  currentLanguage = lang;
  applyInterfaceLanguage(lang);
  localStorage.setItem('hf-companion-lang', lang);
  notes.innerHTML = `<div class="loading">${lang === 'zh' ? '正在加载笔记…' : 'Loading notes…'}</div>`;
  toc.innerHTML = '';

  try {
    const response = await fetch(noteSources[lang], { cache: 'no-cache' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const markdown = await response.text();
    notes.innerHTML = marked.parse(markdown, { gfm: true, breaks: false });
    buildToc();
  } catch (error) {
    const fallback = lang === 'zh'
      ? 'https://github.com/SamZebrado/hf-llm-agents-companion/blob/main/notes/zh-CN.md'
      : 'https://github.com/SamZebrado/hf-llm-agents-companion/blob/main/notes/en.md';
    notes.innerHTML = lang === 'zh'
      ? `<p>笔记加载失败。可以先<a href="${fallback}">在 GitHub 查看 Markdown 原文</a>。</p>`
      : `<p>Could not load the notes. You can <a href="${fallback}">read the Markdown source on GitHub</a>.</p>`;
    console.error(error);
  }
}

langButtons.forEach((button) => {
  button.addEventListener('click', () => {
    if (button.dataset.lang !== currentLanguage) {
      loadNotes(button.dataset.lang);
    }
  });
});

const saved = localStorage.getItem('hf-companion-lang');
const initial = saved || (navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en');
loadNotes(initial);
