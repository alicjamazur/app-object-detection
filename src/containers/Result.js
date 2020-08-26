import React, { useState, useEffect } from "react";
import { PageHeader, ListGroup, ListGroupItem } from "react-bootstrap";
import { onError } from "../libs/errorLib";
import "./Result.css";
import { Link } from "react-router-dom";


export default function Result() {
  const [isLoading, setIsLoading] = useState(true);
  const [result, setResult] = useState([]);


  return (
    <div className="Home">
     <p>This is my result: { result }</p>
     
    </div>
  );
}