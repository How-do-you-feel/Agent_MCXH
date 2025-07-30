<template>
  <div class="tool-selector">
    <label for="tool-select">选择工具:</label>
    <select 
      id="tool-select" 
      v-model="selectedTool" 
      @change="onToolChange"
      class="tool-dropdown"
    >
      <option value="" disabled>请选择一个工具</option>
      <option 
        v-for="tool in tools" 
        :key="tool.name" 
        :value="tool.name"
      >
        {{ tool.name }} - {{ tool.description }}
      </option>
    </select>
  </div>
</template>

<script>
export default {
  name: 'ToolSelector',
  props: {
    tools: {
      type: Array,
      default: () => []
    }
  },
  data() {
    return {
      selectedTool: ''
    }
  },
  methods: {
    onToolChange() {
      this.$emit('tool-selected', this.selectedTool)
    }
  }
}
</script>

<style scoped>
.tool-selector {
  margin-bottom: 20px;
}

label {
  display: block;
  margin-bottom: 8px;
  font-weight: bold;
  color: #333;
}

.tool-dropdown {
  width: 100%;
  padding: 10px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  font-size: 14px;
  background-color: white;
  cursor: pointer;
}

.tool-dropdown:focus {
  outline: none;
  border-color: #409eff;
}
</style>