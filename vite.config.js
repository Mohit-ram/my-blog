import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/my-blog/",
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        project001: resolve(__dirname, 'src/projects/project001/project001.html'),
        project002: resolve(__dirname, 'src/projects/project002/project002.html'),
        project003: resolve(__dirname, 'src/projects/project003/project003.html'),
        project004: resolve(__dirname, 'src/projects/project004/project004.html'),
        project005: resolve(__dirname, 'src/projects/project005/project005.html'),
        project006: resolve(__dirname, 'src/projects/project006/project006.html'),
        project007: resolve(__dirname, 'src/projects/project007/project007.html'),
      },
    },
  },
})
