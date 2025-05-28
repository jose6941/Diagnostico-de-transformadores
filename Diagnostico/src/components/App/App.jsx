import React from "react";
import "./App.css";
import doubleQoute from "../../assets/double-quote.svg";
import "../../fonts/fonts.css";
import Page from "../Page/Page";
import Header from "../Header/Header";
import Hero from "../Hero/Hero";
import Logos from "../Logos/Logos";
import Testimonial from "../Testimonial/Testimonial";
import Features from "../Features/Features";
import Navigation from "../Navigation/Navigation";

const App = () => {
  return (
    <Page>
      <Header>
        <Navigation />
        <Hero />
      </Header>
      <Logos />
      <Features />
    </Page>
  );
};

export default App;
