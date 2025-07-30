// /home/ps/MCXH/Agent_MCXH/webui/frontend/src/main.js
import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'

// 设置axios默认配置
axios.defaults.baseURL = 'http://localhost:5000'

createApp(App).mount('#app')