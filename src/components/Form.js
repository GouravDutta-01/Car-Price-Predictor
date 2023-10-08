import React, { useState } from "react";

const Form = (props) => {
    // let {mode} = props;
  const [isLoading, setIsloading] = useState(false);
  const [formData, setFormData] = useState({
    Year:'', 
    Present_Price:'',
    Kms_Driven:'',
    Fuel_Type:'',
    Seller_Type:'', 
    Transmission:'', 
    Owner:''
});
const [result, setResult] = useState("");


const handleChange = (event) => {
    const value = event.target.value;
    const name = event.target.name;
    let inputData = {...formData};
    inputData[name]=value;
    setFormData(inputData);
    // console.log(inputData)
}

const handlePredictClick = () => {
    const url = "http://localhost:5001/predict";
    setIsloading(true);
    fetch(url,
    {
        headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify(formData)
    })  
    .then(response => response.json())
    .then(response => {
        setResult(response.Prediction);
        setIsloading(false);
    });
}

// const handleCancelClick = (event) => {
//     setResult("");
// }

  return (
    <>
      <div className="container text-center">
        <h1
          className="text-center mb-4"
          style={{
            margin: "35px 0px",
            marginTop: "90px",
            color: props.mode === "dark" ? "white" : "black",
          }}
        >
          Car Price Predictor
        </h1>
        <div className="container" style={{
            margin: "35px 0px",
            marginTop: "90px",
            color: props.mode === "dark" ? "white" : "black",
          }}>
          <form method="post" accept-charset="utf-8" name="Modelform">
            <div className="text-center mb-3">
              <label>
                <b>Enter Year of Purchase:</b>
              </label>
              <br />
              <input
                type="text"
                className="form-control"
                id="Year"
                name="Year"
                value={formData.Year}
                onChange={handleChange}
                placeholder="Enter Year of Purchase "
              />
            </div>
            <div className="form-group">
              <label>
                <b>Enter Present Price(in Lakhs):</b>
              </label>
              <br />
              <input
                type="text"
                className="form-control"
                id="Present_Price"
                name="Present_Price"
                value={formData.Present_Price}
                onChange={handleChange}
                placeholder="Enter Present Price(in Lakhs)"
              />
            </div>
            <div className="form-group">
              <label>
                <b>
                  Enter the Number of Kilometres that the car has travelled:
                </b>
              </label>
              <br />
              <input
                type="text"
                className="form-control"
                id="Kms_Driven"
                name="Kms_Driven"
                value={formData.Kms_Driven}
                onChange={handleChange}
                placeholder="Enter the kilometres driven "
              />
            </div>
            <div className="form-group">
              <label>
                <b>Select the Fuel Type:</b>
              </label>
              <br />
              <select
                className="selectpicker form-control"
                id="Fuel_Type"
                name="Fuel_Type"
                value={formData.Fuel_Type}
                onChange={handleChange}
                required="1"
              >
                <option value="0">Petrol</option>
                <option value="1">Diesel</option>
                <option value="2">CNG</option>
              </select>
            </div>
            <div className="form-group">
              <label>
                <b>Select the Seller Type:</b>
              </label>
              <br />
              <select
                className="selectpicker form-control"
                id="Seller_Type"
                name="Seller_Type"
                value={formData.Seller_Type}
                onChange={handleChange}
                required="1"
              >
                <option value="0">Dealer</option>
                <option value="1">Individual</option>
              </select>
            </div>
            <div className="form-group">
              <label>
                <b>Select the Transmission Type:</b>
              </label>
              <br />
              <select
                className="selectpicker form-control"
                id="Transmission"
                name="Transmission"
                value={formData.Transmission}
                onChange={handleChange}
                required="1"
              >
                <option value="0">Manual</option>
                <option value="1">Automatic</option>
              </select>
            </div>
            <div className="form-group">
              <label>
                <b>Enter the Number of Owners:</b>
              </label>
              <br />
              <input
                type="text"
                className="form-control"
                id="Owner"
                name="Owner"
                value={formData.Owner}
                onChange={handleChange}
                placeholder="Enter the number of Owner "
              />
            </div>
            <div className="form-group mt-3">
              <button
                className="btn btn-primary form-control"
                disabled={isLoading}
                onClick={!isLoading ? handlePredictClick : null}
              >
                Predict Selling Price
              </button>
            </div>
          </form>
          <br />
          <div className="text-center">
            <h4>
              <span id="prediction">{result}</span>
            </h4>
          </div>
        </div>
      </div>
    </>
  );
};
export default Form;
