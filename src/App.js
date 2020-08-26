import React from "react";
import "./App.css";
import Routes from "./Routes";
import { Link} from "react-router-dom";
import { Nav, Navbar, NavItem } from "react-bootstrap";
import { LinkContainer } from "react-router-bootstrap";

function App() {
  return (
    <div className="App container">
      <Navbar fluid collapseOnSelect>
      <Navbar.Header>
        <Navbar.Brand >
          <Link to="/">
          <img 
          src="logo.jpg" 
          alt="DEMO object detection app"
          width="130px"
          height="50px"
          />
          </Link>
        </Navbar.Brand>
        < Navbar.Toggle />
      </Navbar.Header>
      <Navbar.Collapse>
        <Nav pullRight>
          
          <LinkContainer to="/architecture">
            <NavItem>How it works</NavItem>
          </LinkContainer>
        </Nav>
      </Navbar.Collapse>

      </Navbar>
      <Routes />
    </div>
  );
}

export default App;