const tf = require("@tensorflow/tfjs-node");

class Perceptron {
  constructor(eta = 0.1, nIter = 50, randomState = 1) {
    this.eta = eta;
    this.nIter = nIter;
    this.randomState = randomState;
    this.epochs = [];
  }

  async fit(X, y) {
    //   [-0.38375655 -0.70611756  1.83471828]
    // [ 0.01624345 -0.00611756 -0.00528172]

    // this.bias = tf.scalar(0.01624345);
    // this.w_ = tf.tensor([-0.00611756, -0.00528172]);
    // tf.randomNormal([1, 2], 0.0, 0.01, "float32", 1).print();

    this.w_ = tf.randomNormal([1, 2], 0.0, 0.01, "float32", 1);
    this.bias = tf.randomNormal([1, 1], 0.0, 0.01, "float32", 1);

    let XDataset = tf.data.array(await X.array());
    let yDataset = tf.data.array(await y.array());

    this.error_ = [];

    for (let i = 0; i < 10; i++) {
      let errors = 0;
      let epoch = [];
      await tf.data
        .zip([XDataset, yDataset])
        .forEachAsync(async ([xi, target]) => {
          xi = tf.tensor(xi);
          target = tf.scalar(target);

          let yPredict = await this.predict(xi);
          epoch.push((await yPredict.array())[0][0]);

          let update = tf.scalar(this.eta).mul(target.sub(yPredict));

          this.w_ = this.w_.add(update.mul(xi));
          this.bias = this.bias.add(update);

          errors += (await update.array())[0][0] !== 0 ? 1 : 0;
        });
      this.epochs.push(epoch);
      this.error_.push(errors);
    }
  }

  async netInput(xi) {
    return tf.dot(this.w_, xi).add(this.bias);
  }

  async predict(xi) {
    return tf.where(
      tf.greaterEqual(await this.netInput(xi), 0.0),
      tf.scalar(1),
      tf.scalar(-1)
    );
  }
}

module.exports = Perceptron;
