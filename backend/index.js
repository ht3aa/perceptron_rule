const tf = require("@tensorflow/tfjs-node");
const perceptron = require("./perceptron");
const express = require("express");
const app = express();
const cors = require("cors");

var corsOptions = {
  origin: "http://localhost:5173",
};

app.use(express.json());
app.use(cors(corsOptions));

async function main() {
  const dataSet = await tf.data
    .csv(`file://${__dirname}/iris.csv`)
    .take(100)
    .toArray();
  const X = tf.tensor(
    dataSet.map((row) => [row.sepal_length, row.petal_length])
  );
  const y = tf.tensor(
    dataSet.map((row) => (row.class === "Iris-setosa" ? -1 : 1))
  );

  const ppn = new perceptron(0.1, 10);
  await ppn.fit(X, y);
  const yArr = await y.array();
  const XArr = await X.array();
  const epochs = ppn.epochs;
  return { epochs, XArr, yArr };
}

main();

app.listen(8000, () => {
  console.log("helo");
});

app.get("/", async (req, res) => {
  res.json(await main());
});
