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
        project008: resolve(__dirname, 'src/projects/project008/project008.html'),
        project009: resolve(__dirname, 'src/projects/project009/project009.html'),
        project010: resolve(__dirname, 'src/projects/project010/project010.html'),
        project011: resolve(__dirname, 'src/projects/project011/project011.html'),
        project012: resolve(__dirname, 'src/projects/project012/project012.html'),
        project013: resolve(__dirname, 'src/projects/project013/project013.html'),
        project014: resolve(__dirname, 'src/projects/project014/project014.html'),
        project015: resolve(__dirname, 'src/projects/project015/project015.html'),
        project016: resolve(__dirname, 'src/projects/project016/project016.html'),
        project017: resolve(__dirname, 'src/projects/project017/project017.html'),
        project018: resolve(__dirname, 'src/projects/project018/project018.html'),
        project019: resolve(__dirname, 'src/projects/project019/project019.html'),
        project020: resolve(__dirname, 'src/projects/project020/project020.html'),
        project021: resolve(__dirname, 'src/projects/project021/project021.html'),
        project022: resolve(__dirname, 'src/projects/project022/project022.html'),
        project023: resolve(__dirname, 'src/projects/project023/project023.html'),
        project024: resolve(__dirname, 'src/projects/project024/project024.html'),
        project025: resolve(__dirname, 'src/projects/project025/project025.html'),
        
      },
    },
  },
})
