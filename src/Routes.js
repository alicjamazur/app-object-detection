import React from "react";
import { Route, Switch } from "react-router-dom";
import Home from "./containers/Home";
import NotFound from "./containers/NotFound";
import NewInference from "./containers/NewInference";
import Result from "./containers/Result";
import Architecture from "./containers/Architecture";


export default function Routes() {
  return (
    <Switch>
      <Route exact path="/">
        <Home />
      </Route>
      <Route exact path="/inference">
        <NewInference />
      </Route>
      <Route exact path="/result">
        <Result />
      </Route>
      <Route exact path="/architecture">
        <Architecture />
      </Route>

      <Route>
        <Home />
      </Route>
    </Switch>
  );
}