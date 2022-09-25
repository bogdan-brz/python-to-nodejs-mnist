const tf = require("@tensorflow/tfjs-node");
let model;

const initModel = async () => {
    model = await tf.loadLayersModel(
        // "http://127.0.0.1:8887/model_1/model.json" // this loads from the dummy chrome web server
        "https://raw.githubusercontent.com/bogdan-brz/python-to-browser-mnist-recog/master/model_1/model.json" // this loads from github
    );

    model.compile({
        optimizer: tf.train.adam(),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });
};

const predict = (imgData) => {
    const tensor = tf.tensor([imgData], [1, 28, 28, 1]); // convert the image data received in form of array into a tensor of the right shape the imgData is in [],
    // becuase sent over http an arry of length 1 becomes just the sole element of itself
    const prediction = model.predict(tensor);
    const pIndex = tf.argMax(prediction, 1).arraySync()[0]; // converts the one-hot encoded label into an array (.data() => object; .array => array; sync => synchronously)
    return pIndex;
};

initModel();

module.exports = { predict };
