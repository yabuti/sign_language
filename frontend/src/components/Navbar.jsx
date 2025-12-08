import React from "react";
import { useNavigate } from "react-router-dom";
import { FiVideo, FiImage, FiMic, FiLogOut } from "react-icons/fi";
import "./Navbar.css";

function Navbar({ onSelect }) {
  const navigate = useNavigate();

  const handleLogout = () => {
    // Add logout logic here (clear tokens, etc.)
    navigate("/login");
  };

  return (
    <nav className="navbar">
      <h1>✨ Sign Language Translator</h1>
      <div className="nav-buttons">
        <button onClick={() => onSelect("webcam")} className="nav-btn">
          <FiVideo size={18} />
          <span>Webcam</span>
        </button>
        <button onClick={() => onSelect("image")} className="nav-btn">
          <FiImage size={18} />
          <span>Image</span>
        </button>
        <button onClick={() => onSelect("voice")} className="nav-btn">
          <FiMic size={18} />
          <span>Voice</span>
        </button>
        <button onClick={handleLogout} className="nav-btn logout-btn">
          <FiLogOut size={18} />
          <span>Logout</span>
        </button>
      </div>
    </nav>
  );
}

export default Navbar;
