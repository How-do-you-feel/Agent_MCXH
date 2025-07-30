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
      this.isInitialized = true
    }
  }
}
</script>