const toggle = document.getElementById('lang-toggle');
const notesLink = document.getElementById('full-notes-link');

function setLanguage(lang) {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-zh][data-en]').forEach((el) => {
    el.textContent = lang === 'zh' ? el.dataset.zh : el.dataset.en;
  });
  toggle.textContent = lang === 'zh' ? 'EN' : '中';
  toggle.setAttribute('aria-label', lang === 'zh' ? 'Switch to English' : '切换到中文');
  notesLink.href = lang === 'zh'
    ? 'https://github.com/SamZebrado/hf-llm-agents-companion/blob/main/notes/zh-CN.md'
    : 'https://github.com/SamZebrado/hf-llm-agents-companion/blob/main/notes/en.md';
  localStorage.setItem('hf-companion-lang', lang);
}

const saved = localStorage.getItem('hf-companion-lang');
const initial = saved || (navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en');
setLanguage(initial);

toggle.addEventListener('click', () => {
  const next = document.documentElement.lang === 'zh-CN' ? 'en' : 'zh';
  setLanguage(next);
});
