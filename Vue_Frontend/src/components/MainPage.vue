
<template>
  <img style="height: 200px; width: 200px;" alt="Recycling logo" :src="selectedImage ? selectedImage : defaultImage">

  <h2>Upload your file here</h2>

  <h4>{{ fileName }}</h4>

  <div class="container">
    <label for="file-upload" class="card">

      Select File to Upload
    </label>
    <input id="file-upload" class="card" type="file" @change="uploadFile" ref="file">
    <button class="btn" @click="submitFile">Upload</button>
  </div>

  <h2 style="margin-top: 5%;">Detection Output : </h2>

  <div class="my-table">
    <table>
      <tr>
        <th>Trash Type</th>
        <th>Quantity</th>
      </tr>
      <tr v-for="(obj, index) in result.data" :key="index">
        <td>{{ obj.type }}</td>
        <td>{{ obj.count }}</td>

      </tr>
    </table>
  </div>
</template>

<script>

import defaultImage from '../assets/recycling.png';

export default {

  data() {
    return {
      publicPath: process.env.BASE_URL,
      fileName: "No File Selected",
      selectedImage: "",
      defaultImage: defaultImage,
      result: {}
    }
  },

  methods: {
    uploadFile() {
      this.Images = this.$refs.file.files[0];
      this.fileName = this.Images.name;
      this.selectedImage = URL.createObjectURL(this.Images);
      console.log(this.selectedImage);
    },
    submitFile() {
      let formData = new FormData();
      formData.append("image_file", this.Images)
      fetch('http://127.0.0.1:3000/detect', {
        method: 'POST',
        body: formData
      })
        .then(res => res.json())
        .then(data => {
          console.log(data)
          this.result = data
        })
        .then(error => console.log(error))
    }
  }
}
</script>

<style scoped>
.container,.my-table {
  display: flex;
  justify-content: center;
  align-items: center;


}

.btn {
  background-color: #2d6ac6;
  padding: 14px 40px;
  color: #fff;
  text-transform: uppercase;
  letter-spacing: 2px;
  cursor: pointer;
  border-radius: 10px;
  border: 2px #f4f4f4;
  box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px, rgba(0, 0, 0, 0.3) 0px 1px 3px -1px;
  transition: .4s;
}

.btn span:last-child {
  display: none;
}

.btn:hover {
  transition: .4s;
  border: 2px #000000;
  background-color: #fff;
  color: #2b2e2e;
}

.btn:active {
  background-color: #879ddb;
}


input[type="file"] {
  display: none;
}

.card {
  box-sizing: border-box;
  width: 250px;
  height: 50px;
  background: rgba(217, 217, 217, 0.58);
  border: 1px solid white;
  backdrop-filter: blur(6px);
  border-radius: 17px;
  text-align: center;
  cursor: pointer;
  transition: all 0.5s;
  display: flex;
  align-items: center;
  justify-content: center;
  user-select: none;
  font-weight: bolder;
  color: black;
  margin-right: 50px;
}

.card:hover {
  /* border: 1px solid black; */
  transform: scale(1.05);
}

.card:active {
  /* transform: scale(0.95) rotateZ(1.7deg); */
  transform: scale(0.95);
}

.my-table {
  border-collapse: collapse;
  width: 100%;
}

.my-table th, .my-table td {
  border: 1px solid black;
  padding: 8px;
}

.my-table th {
  background-color: #f2f2f2;
}
</style>