import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiUser, FiMail, FiLock, FiCheckCircle } from "react-icons/fi";
import "./auth.css";

function Register() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordMessage, setPasswordMessage] = useState("");

  const validatePassword = (value) => {
    if (!/[A-Z]/.test(value)) return "Password must include at least one uppercase letter";
    if (!/[a-z]/.test(value)) return "Password must include at least one lowercase letter";
    if (!/\d/.test(value)) return "Password must include at least one number";
    if (!/[@$!%*?&]/.test(value)) return "Password must include at least one special character (@$!%*?&)";
    if (value.length < 8) return "Password must be at least 8 characters long";
    return "";
  };

  const handlePasswordChange = (e) => {
    const value = e.target.value;
    setPassword(value);
    const msg = validatePassword(value);
    setPasswordMessage(msg);
  };

  const handleRegister = async (e) => {
    e.preventDefault();

    if (passwordMessage) {
      alert(passwordMessage);
      return;
    }

    if (password !== confirmPassword) {
      alert("Passwords do not match!");
      return;
    }

    try {
      const response = await fetch("http://localhost:5000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        alert("✅ Registration successful!\n\nWelcome to Sign Language Translator!");
        navigate("/dashboard");
      } else {
        // Show clear error messages
        if (data.msg === "Email already exists") {
          alert("❌ This email is already registered.\n\nPlease use a different email or go to Login page.");
        } else if (data.msg === "All fields are required") {
          alert("❌ Please fill in all fields.");
        } else {
          alert("❌ Registration failed: " + (data.msg || "Unknown error"));
        }
      }
    } catch (error) {
      console.error("Registration error:", error);
      alert("Unable to connect to server. Please try again.");
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h2>📝 Register</h2>
          <p className="auth-subtitle">Create a new account to get started</p>
        </div>

        <form onSubmit={handleRegister}>
          <div className="input-group">
            <FiUser className="input-icon" />
            <input
              type="text"
              placeholder="Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <FiMail className="input-icon" />
            <input
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          {/* Password message notification */}
          {passwordMessage && (
            <div className="password-notice">
              <FiCheckCircle size={16} />
              <span>{passwordMessage}</span>
            </div>
          )}

          <div className="input-group">
            <FiLock className="input-icon" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={handlePasswordChange}
              required
            />
          </div>
          <div className="input-group">
            <FiLock className="input-icon" />
            <input
              type="password"
              placeholder="Confirm Password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="auth-button">Create Account</button>
        </form>

        <p className="auth-footer">
          Already have an account?{" "}
          <span className="auth-link" onClick={() => navigate("/login")}>
            Sign In
          </span>
        </p>
      </div>
    </div>
  );
}

export default Register;
