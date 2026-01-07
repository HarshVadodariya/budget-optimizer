// Dark Mode Toggle Functionality

(function() {
  'use strict';

  // Get theme from localStorage or default to light
  const getTheme = () => {
    const saved = localStorage.getItem('theme');
    return saved || 'light';
  };

  // Set theme
  const setTheme = (theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    updateThemeIcon(theme);
  };

  // Update theme icon
  const updateThemeIcon = (theme) => {
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
      const icon = toggle.querySelector('.theme-icon');
      if (icon) {
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
      }
    }
  };

  // Initialize theme on page load
  const initTheme = () => {
    const theme = getTheme();
    setTheme(theme);
  };

  // Toggle theme
  const toggleTheme = () => {
    const current = getTheme();
    const newTheme = current === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  };

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }

  // Attach toggle handler
  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('themeToggle');
    if (toggle) {
      toggle.addEventListener('click', toggleTheme);
    }
  });

  // Expose for external use if needed
  window.themeToggle = toggleTheme;
})();

