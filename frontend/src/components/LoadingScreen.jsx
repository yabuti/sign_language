import React from "react";
import "./LoadingScreen.css";

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="loading-spinner"></div>
        <h2 className="loading-title">✨ Sign Language Translator</h2>
        <p className="loading-text">Loading your experience...</p>
      </div>
    </div>
  );
}

export default LoadingScreen;
