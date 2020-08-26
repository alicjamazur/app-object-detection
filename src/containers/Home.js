import React from "react";
import "./Home.css";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="Home">
      <div className="lander">
      <img 
          src="logo.jpg" 
          alt="DEMO object detection app"
          />
        <p>Detect objects on your images using computer vision</p>
        <Link to="/inference" className="btn btn-danger btn-lg">
          Get started
        </Link>
      </div>

    </div>
  );
}