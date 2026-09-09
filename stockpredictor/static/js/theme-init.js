/* Apply saved theme + accent before paint to avoid a flash of the wrong theme.
   Kept dependency-free and synchronous (loaded from <head>). */
(function () {
    var saved = null;
    try { saved = localStorage.getItem('spp-theme'); } catch (e) { saved = null; }
    if (!saved) {
        try {
            saved = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
        } catch (e) { saved = 'dark'; }
    }
    document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');
    var accent = null;
    try { accent = localStorage.getItem('spp-accent'); } catch (e) { accent = null; }
    var accents = ['blue', 'violet', 'indigo', 'teal', 'gold', 'rose'];
    if (accents.indexOf(accent) !== -1) {
        document.documentElement.setAttribute('data-accent', accent);
    }
})();
