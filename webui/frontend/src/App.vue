<template>
  <div id="app">
    <header>
      <h1>视觉智能体 WebUI</h1>
    </header>
    <main>
      <InitAgent v-if="!isInitialized" @initialized="onInitialized" />
      <ChatInterface v-else />
    </main>
  </div>
</template>

<script>
import ChatInterface from './components/ChatInterface.vue'
import InitAgent from './components/InitAgent.vue'

export default {
  name: 'App',
  components: {
    ChatInterface,
    InitAgent
  },
  data() {
    return {
      isInitialized: false
    }
  },
  methods: {
    onInitialized() {
      console.log('Agent initialized successfully')
      this.isInitialized = true // 修复拼写错误
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'Arial', sans-serif;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  background-color: #42b983;
  color: white;
  padding: 1rem;
  text-align: center;
}

header h1 {
  font-size: 1.5rem;
  font-weight: normal;
}

main {
  flex: 1;
  overflow: hidden;
}

@media (max-width: 768px) {
  header h1 {
    font-size: 1.2rem;
  }
}
</style>