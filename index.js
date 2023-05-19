const express = require("express");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");

const app = express();
const port = 3000;
let model;

const handler = tfn.io.fileSystem("./my_model.json");
async function loadModel() {
  try {
    model = await tf.loadLayersModel(handler);
    console.log("Model loaded.");
    model.summary();
  } catch (error) {
    console.error("Error loading the model:", error);
  }
}

loadModel();

app.use(bodyParser.json());

app.post("/predict", async (req, res) => {
  try {
    const input_data = req.body.input_data;
    const input_data_as_array = Array.isArray(input_data)
      ? input_data
      : [input_data];
    const input_data_as_tensor = tf.tensor2d(input_data_as_array, [
      1,
      input_data_as_array.length,
    ]);

    const reshaped_input = input_data_as_tensor.reshape([
      1,
      input_data_as_array.length,
    ]);

    const prediction = model.predict(reshaped_input);
    console.log(prediction.dataSync()[0]);
    const prediction_label = prediction.argMax(1).dataSync()[0];
    const response =
      prediction_label === 0 ? "Tümör iyi huylu" : "Tümör kötü huylu";

    res.json({ prediction: response });
  } catch (error) {
    console.log(error);
    res.status(500).json({ error: "Tahmin yapılırken bir hata oluştu." });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}.`);
});
