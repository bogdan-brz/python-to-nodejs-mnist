const express = require("express");
const path = require("path");
const app = express();
const port = 3000;
const { predict } = require("./model");

app.set("view engine", "ejs"); // allows for res.render
app.set("views", path.join(__dirname, "./")); // tells the view engine where the files to render are

app.use(express.json()); // body parser; IMPORTANT!
app.use(express.static("./")); // tells express to serve static files from root directory

app.get("/", (req, res) => {
    res.render("home.ejs");
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});

app.post("/predict", (req, res) => {
    const prediction = predict(req.body.imgData);
    res.send(`${prediction}`);
});
