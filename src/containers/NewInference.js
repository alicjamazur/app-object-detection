import React, { useRef, useState, useEffect } from "react";
import { useHistory } from "react-router-dom";
import { FormGroup, FormControl, ControlLabel } from "react-bootstrap";
import LoaderButton from "../components/LoaderButton";
import config from "../config";
import "./NewInference.css";
import { API } from "aws-amplify";
import { onError } from "../libs/errorLib";
import { Link } from "react-router-dom";
import {Jumbotron, Row, Col} from 'react-bootstrap'
import {Button} from 'react-bootstrap'



export default function NewInference() {
  const file = useRef(null);
  const [outputImage, setOutputImage] = useState("");
  const [inputImage, setInputImage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceMade, setInferenceMade] = useState(false);
  const [inferenceStatus, setInferenceStatus] = useState("");


  // Handle image submission
  function handleFileChange(event) {
    file.current = event.target.files[0];

    // Convert image to base64
    var reader = new FileReader();
    reader.readAsDataURL(file.current);
    reader.onloadend = function() {
    const base64string = reader.result;
    let imageBase64 = base64string.substr(base64string.indexOf(',') + 1);
    setInputImage(imageBase64);
    }
  }

  // Make inference 
  async function handleSubmit(event) {
    event.preventDefault();

    // Validate input file
    if (file.current && (file.current.type != "image/png" && file.current.type != "image/jpeg")) {
      alert(
      ` Pick a JPG or PNG file `)
      return;
      };
    
    setIsLoading(true);
    setInferenceStatus("Processing...")

    try {
      console.log("inference");
      const response = await inference(inputImage);
      setOutputImage(response);
      learnMore();

    } catch (e) {
      const coldStart = onError(e);
      setInferenceMade(false);

      if (coldStart) {
        setInferenceStatus("Just few more seconds, Lambda is warming up...")
        return handleSubmit(event);
      }
    }
    setInferenceStatus("");
    setIsLoading(false);

  }

  function learnMore() {
    setTimeout(() => {
      setInferenceMade(true)
    }, 5000);
  }
  

  // API call with Amplify
  function inference(inputImage) {
    return API.post("inference", "", {
      headers: {
        "Content-Type": "application/json", 
        "Accept": "application/json"
      },
      body: inputImage
    });
  }


  // Render webpage
  return (
    <body>
      <div>
        {inferenceMade ? (
          <>
            <Jumbotron className="PostInference">
              <Row> 
                <Col xs={5} md={5}>
                <h4>Liked it?</h4>
                <h5>If you are curious, learn more about the architecture of this app.</h5>
              </Col>
              <Col xs={5} md={5}>
                <p>
                  <Button size="sm" variant="primary" href="/architecture">Learn more</Button>
                </p>
              </Col>
              </Row>
            </Jumbotron>
          </>
          ) : null }
      </div>
      <div className="NewInference">
        <form>
          <FormGroup controlId="imageFile">
            <ControlLabel>Choose an image to process (jpg or png)</ControlLabel>
            <FormControl onChange={handleFileChange} type="file" />
          </FormGroup>
          <p>{ inferenceStatus }</p>
        </form>
      </div>
      <div className="Button">
        <form onSubmit={handleSubmit}>
        <LoaderButton
            type="submit"
            bsSize="large"
            bsStyle="danger"
            isLoading={isLoading}
           
          >
          Detect objects
          </LoaderButton>
        </form>
      </div>
      <img class="center" src={outputImage == "" ? "" : `data:image/png;base64,${outputImage}`} />
    </body>
  );
  }
