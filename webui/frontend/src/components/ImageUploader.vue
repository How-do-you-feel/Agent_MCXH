<template>
  <div class="image-uploader">
    <div class="upload-area" @dragover.prevent @drop.prevent="handleDrop">
      <input 
        type="file" 
        ref="fileInput" 
        accept="image/*" 
        @change="handleFileSelect" 
        style="display: none"
      />
      <div 
        class="drop-zone" 
        @click="triggerFileInput"
        :class="{ 'drag-over': isDragOver }"
      >
        <div v-if="!imagePreview" class="upload-placeholder">
          <i class="upload-icon">📁</i>
          <p>点击选择图片或拖拽图片到此处</p>
          <p class="hint">支持 JPG, PNG, BMP 格式</p>
        </div>
        <div v-else class="preview-container">
          <img :src="imagePreview" alt="Preview" class="image-preview" />
          <button class="remove-btn" @click.stop="removeImage">✕</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImageUploader',
  data() {
    return {
      isDragOver: false,
      imagePreview: null,
      selectedFile: null
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },
    handleFileSelect(event) {
      const file = event.target.files[0]
      if (file && file.type.startsWith('image/')) {
        this.processFile(file)
      }
    },
    handleDrop(event) {
      this.isDragOver = false
      const file = event.dataTransfer.files[0]
      if (file && file.type.startsWith('image/')) {
        this.processFile(file)
      }
    },
    processFile(file) {
      this.selectedFile = file
      const reader = new FileReader()
      reader.onload = (e) => {
        this.imagePreview = e.target.result
        this.$emit('image-selected', file)
      }
      reader.readAsDataURL(file)
    },
    removeImage() {
      this.imagePreview = null
      this.selectedFile = null
      this.$refs.fileInput.value = ''
      this.$emit('image-removed')
    }
  }
}
</script>

<style scoped>
.image-uploader {
  width: 100%;
  margin-bottom: 20px;
}

.upload-area {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  transition: all 0.3s ease;
}

.drop-zone {
  cursor: pointer;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.drag-over {
  border-color: #409eff;
  background-color: #f0f9ff;
}

.upload-placeholder {
  color: #909399;
}

.upload-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 10px;
}

.hint {
  font-size: 12px;
  color: #c0c4cc;
  margin-top: 5px;
}

.preview-container {
  position: relative;
  display: inline-block;
}

.image-preview {
  max-width: 100%;
  max-height: 300px;
  border-radius: 4px;
}

.remove-btn {
  position: absolute;
  top: -10px;
  right: -10px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #f56c6c;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>