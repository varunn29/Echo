import { defineConfig } from 'wxt';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  modules: ['@wxt-dev/module-react'],
  manifest: {
    permissions: ['storage', 'alarms', 'offscreen'],
    host_permissions: [
      'http://localhost:11434/*',
      'http://127.0.0.1:11435/*',
    ],
  },
  vite: function () {
    return {
      plugins: [tailwindcss()],
    };
  },
});