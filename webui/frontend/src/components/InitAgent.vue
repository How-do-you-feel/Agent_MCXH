<template>
  <div class="init-agent">
    <h2>初始化Vision Agent</h2>
    <form @submit.prevent="initializeAgent">
      <div class="form-group">
        <label for="modelPath">模型路径</label>
        <input 
          type="text" 
          id="modelPath" 
          v-model="modelPath" 
          required
        />
      </div>
      <div class="form-group">
        <label for="host">Host</label>
        <input 
          type="text" 
          id="host" 
          v-model="host" 
          required
        />
      </div>
      <div class="form-group">
        <label for="port">Port</label>
        <input 
          type="number" 
          id="port" 
          v-model.number="port" 
          required
        />
      </div>
      <button type="submit">初始化Agent</button>
    </form>
    <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'InitAgent',
  data() {
    return {
      modelPath: '/home/ps/Qwen2.5-3B',
      host: '127.0.0.1',
      port: 8001,
      errorMessage: ''
    }
  },
  methods: {
    async initializeAgent() {
      this.errorMessage = ''
      
      try {
        const response = await axios.post('/api/init', {
          model_path: this.modelPath,
          host: this.host,
          port: this.port
        })
        
        if (response.status === 200) {
          // 初始化成功，触发initialized事件
          this.$emit('initialized')
        } else {
          this.errorMessage = '初始化失败'
        }
      } catch (error) {
        console.error('Error initializing agent:', error)
        this.errorMessage = '初始化失败: ' + (error.response?.data?.error || error.message)
      }
    }
  }
}
</script>

<style scoped>
.init-agent {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.form-group {
  margin-bottom: 15px;
}

.form-group label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

.form-group input {
  width: 100%;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

button {
  padding: 10px 20px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

button:hover {
  background-color: #3e9a75;
}

.error {
  margin-top: 10px;
  color: red;
  font-size: 14px;
}
</style>