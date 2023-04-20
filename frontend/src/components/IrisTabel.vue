<script lang="ts" setup>
import axios from "axios";
import { ref } from "vue";

const data: any = ref([]);

axios.get("http://localhost:8000/").then(async (res) => {
  let result = await res.data;

  let epochNumber = 0;
  for (let i = 0; i < result.XArr.length; i++) {
    let temp = [
      [result.XArr[i][0].toFixed(2), result.XArr[i][1].toFixed(2)],
      [result.yArr[i]],
    ];
    for (let i = 0; i < result.epochs.length; i++) {
      temp[1].push(result.epochs[i][epochNumber]);
    }
    epochNumber++;

    data.value.push(temp);
  }
  console.log(data.value);
});
</script>

<template>
  <table class="w-full text-center">
    <caption class="bg-slate-600 py-5 text-4xl text-white">
      Perceptron Rule for Iris Data
    </caption>
    <colgroup></colgroup>
    <thead>
      <tr>
        <th>Sepal length</th>
        <th>petal length</th>
        <th>target</th>
        <th>Training Prediction 1</th>
        <th>Training Prediction 2</th>
        <th>Training Prediction 3</th>
        <th>Training Prediction 4</th>
        <th>Training Prediction 5</th>
        <th>Training Prediction 6</th>
        <th>Training Prediction 7</th>
        <th>Training Prediction 8</th>
        <th>Training Prediction 9</th>
        <th>Training Prediction 10</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="el in data" :key="el">
        <td v-for="d in el[0]" :key="d">{{ d }}</td>
        <td v-for="d in el[1]" :key="d">
          <img
            class="w-3/4 mx-auto"
            :src="d === -1 ? '/Iris_setosa.jpg' : '/Iris_versicolor.jpg'"
            :alt="d === -1 ? '/Iris_setosa.jpg' : '/Iris_versicolor.jpg'"
          />
          <!-- {{ d }} -->
        </td>
      </tr>
    </tbody>
  </table>
</template>
