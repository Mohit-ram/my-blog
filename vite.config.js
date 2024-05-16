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
      },
    },
  },
})
