// ====== Import Dependencies ======
const express = require("express");
const mysql = require("mysql2");
const cors = require("cors");
const bodyParser = require("body-parser");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const fs = require("fs");
// const speech = require("@google-cloud/speech"); // Commented out - install if needed for voice

// ====== Initialize App ======
const app = express();
app.use(cors());
app.use(bodyParser.json());

const JWT_SECRET = "supersecretkey"; // ⚠️ Change this in production!

// ====== MySQL Connection ======
let db;
let dbConnected = false;

try {
  db = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "", // your MySQL password
    database: "sign",
  });

  db.connect((err) => {
    if (err) {
      console.error("⚠️  MySQL connection failed:", err.message);
      console.log("⚠️  Server will run without database features");
      dbConnected = false;
    } else {
      console.log("✅ Connected to MySQL database!");
      dbConnected = true;
    }
  });
} catch (error) {
  console.error("⚠️  MySQL setup failed:", error.message);
  dbConnected = false;
}

// ====== Middleware: Verify Admin ======
function verifyAdmin(req, res, next) {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(403).json({ msg: "No token provided" });

  jwt.verify(token, JWT_SECRET, (err, decoded) => {
    if (err) return res.status(401).json({ msg: "Invalid token" });

    db.query("SELECT * FROM users WHERE id = ?", [decoded.id], (err, results) => {
      if (err || results.length === 0) return res.status(404).json({ msg: "User not found" });
      if (!results[0].is_admin) return res.status(403).json({ msg: "Access denied" });
      req.user = results[0];
      next();
    });
  });
}

// ====== Register User ======
app.post("/register", async (req, res) => {
  // Check if database is connected
  if (!dbConnected || !db) {
    return res.status(500).json({ 
      msg: "Database not available. Please make sure XAMPP MySQL is running." 
    });
  }

  const { name, email, password } = req.body;

  if (!name || !email || !password)
    return res.status(400).json({ msg: "All fields are required" });

  const hashedPassword = await bcrypt.hash(password, 10);

  db.query(
    "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
    [name, email, hashedPassword],
    (err) => {
      if (err) {
        console.error("Registration error:", err);
        
        // Check if it's actually a duplicate email error
        if (err.code === 'ER_DUP_ENTRY') {
          return res.status(400).json({ msg: "Email already exists" });
        }
        
        // Other database errors
        return res.status(500).json({ 
          msg: "Registration failed. Please check if database is running.",
          error: err.message 
        });
      }
      res.json({ msg: "User registered successfully" });
    }
  );
});

// ====== Login User ======
app.post("/login", (req, res) => {
  // Check if database is connected
  if (!dbConnected || !db) {
    return res.status(500).json({ 
      msg: "Database not available. Please make sure XAMPP MySQL is running." 
    });
  }

  const { email, password } = req.body;

  db.query("SELECT * FROM users WHERE email = ?", [email], async (err, result) => {
    if (err) return res.status(500).json({ msg: "Server error" });
    if (result.length === 0) return res.status(400).json({ msg: "User not found" });

    const user = result[0];
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ msg: "Invalid password" });

    const token = jwt.sign({ id: user.id }, JWT_SECRET, { expiresIn: "3h" });

    res.json({
      msg: "Login successful",
      token,
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        is_admin: user.is_admin,
      },
    });
  });
});

// ====== Activity Tracking ======
app.post("/track-activity", (req, res) => {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) return res.status(403).json({ msg: "No token provided" });

  jwt.verify(token, JWT_SECRET, (err, decoded) => {
    if (err) return res.status(401).json({ msg: "Invalid token" });

    const { feature } = req.body; // webcam, image, or voice
    
    db.query(
      "INSERT INTO user_activity (user_id, feature_used) VALUES (?, ?)",
      [decoded.id, feature],
      (err) => {
        if (err) {
          console.error("Activity tracking error:", err);
          return res.status(500).json({ msg: "Error tracking activity" });
        }
        res.json({ msg: "Activity tracked" });
      }
    );
  });
});

// ====== Admin Routes ======

// ✅ Get all users with activity count
app.get("/admin/users", verifyAdmin, (req, res) => {
  const query = `
    SELECT 
      u.id, 
      u.name, 
      u.email, 
      u.is_admin, 
      u.created_at,
      COUNT(a.id) as activity_count,
      MAX(a.used_at) as last_activity
    FROM users u
    LEFT JOIN user_activity a ON u.id = a.user_id
    GROUP BY u.id
    ORDER BY u.created_at DESC
  `;
  
  db.query(query, (err, users) => {
    if (err) {
      console.error("Error fetching users:", err);
      return res.status(500).json({ msg: "Error fetching users" });
    }
    res.json(users);
  });
});

// ✅ Get user activity details
app.get("/admin/users/:id/activity", verifyAdmin, (req, res) => {
  db.query(
    "SELECT feature_used, used_at FROM user_activity WHERE user_id = ? ORDER BY used_at DESC LIMIT 50",
    [req.params.id],
    (err, activities) => {
      if (err) return res.status(500).json({ msg: "Error fetching activity" });
      res.json(activities);
    }
  );
});

// ✅ Delete user
app.delete("/admin/users/:id", verifyAdmin, (req, res) => {
  db.query("DELETE FROM users WHERE id = ?", [req.params.id], (err) => {
    if (err) return res.status(500).json({ msg: "Error deleting user" });
    res.json({ msg: "User deleted successfully" });
  });
});

// ✅ Promote user to admin
app.put("/admin/users/:id/make-admin", verifyAdmin, (req, res) => {
  db.query("UPDATE users SET is_admin = TRUE WHERE id = ?", [req.params.id], (err) => {
    if (err) return res.status(500).json({ msg: "Error promoting user" });
    res.json({ msg: "User promoted to admin" });
  });
});

// ✅ Demote admin back to normal user
app.put("/admin/users/:id/remove-admin", verifyAdmin, (req, res) => {
  db.query("UPDATE users SET is_admin = FALSE WHERE id = ?", [req.params.id], (err) => {
    if (err) return res.status(500).json({ msg: "Error demoting user" });
    res.json({ msg: "User demoted to regular user" });
  });
});

// ====== Image Upload & Prediction ======
const upload = multer({ dest: "uploads/" });
const { spawn } = require("child_process");
const path = require("path");

// Image to Text Prediction
app.post("/predict", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  const imagePath = path.resolve(req.file.path);
  // Try Keras model first (better for letters), fallback to PyTorch
  const scriptPath = path.join(__dirname, "predict_keras.py");

  // Call Python prediction script
  const python = spawn("python", [scriptPath, imagePath]);

  let result = "";
  let error = "";

  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    error += data.toString();
  });

  python.on("close", (code) => {
    // Clean up uploaded file
    try {
      fs.unlinkSync(imagePath);
    } catch (e) {
      console.error("Failed to delete temp file:", e);
    }

    if (code !== 0) {
      console.error("Python error:", error);
      return res.status(500).json({ 
        text: "Error processing image",
        error: error 
      });
    }

    try {
      const prediction = JSON.parse(result.trim());
      res.json(prediction);
    } catch (e) {
      console.error("JSON parse error:", e);
      res.json({ 
        text: result.trim() || "Unable to recognize sign",
        confidence: 0,
        success: false
      });
    }
  });
});

// ====== Google Speech-to-Text ======
// Commented out - install @google-cloud/speech if you need voice recognition
/*
const speechClient = new speech.SpeechClient({
  keyFilename: "path_to_your_service_account.json",
});

app.post("/speech-to-text", upload.single("audio"), async (req, res) => {
  try {
    const { language } = req.body;
    const fileBytes = fs.readFileSync(req.file.path);
    const audioBytes = fileBytes.toString("base64");

    const request = {
      audio: { content: audioBytes },
      config: {
        encoding: "WEBM_OPUS",
        sampleRateHertz: 48000,
        languageCode: language,
      },
    };

    const [response] = await speechClient.recognize(request);
    const transcription = response.results
      .map((r) => r.alternatives[0].transcript)
      .join("\n");

    fs.unlinkSync(req.file.path);
    res.json({ text: transcription });
  } catch (err) {
    console.error("Speech Error:", err);
    res.status(500).json({ error: "Speech recognition failed" });
  }
});
*/

// Placeholder for speech-to-text (install @google-cloud/speech to enable)
app.post("/speech-to-text", upload.single("audio"), async (req, res) => {
  res.json({ 
    transcript: "Speech-to-text not configured. Install @google-cloud/speech package." 
  });
});

// ====== Webcam Video Prediction (Multi-Model Support) ======
app.post("/predict-webcam", upload.single("video"), (req, res) => {
  console.log("📹 Received webcam prediction request");
  
  if (!req.file) {
    console.log("❌ No video file received");
    return res.status(400).json({ error: "No video uploaded" });
  }

  const videoPath = path.resolve(req.file.path);
  const modelName = req.body.model || "asl_cnn_lstm"; // Default to ASL model
  const scriptPath = path.join(__dirname, "predict_webcam_multi_model.py");

  console.log(`📁 Video saved at: ${videoPath}`);
  console.log(`🤖 Using model: ${modelName}`);
  console.log(`🐍 Running Python script: ${scriptPath}`);

  // Call Python prediction script with model selection
  const python = spawn("python", [scriptPath, videoPath, modelName]);

  let result = "";
  let error = "";

  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    error += data.toString();
  });

  python.on("close", (code) => {
    console.log(`🐍 Python script finished with code: ${code}`);
    
    // Clean up uploaded file
    try {
      fs.unlinkSync(videoPath);
      console.log("🗑️ Temp video file deleted");
    } catch (e) {
      console.error("Failed to delete temp file:", e);
    }

    if (code !== 0) {
      console.error("❌ Python error:", error);
      return res.status(500).json({ 
        text: "Error processing video",
        error: error 
      });
    }

    console.log("📤 Python output:", result);

    try {
      const prediction = JSON.parse(result.trim());
      console.log("✅ Prediction successful:", prediction.text);
      res.json(prediction);
    } catch (e) {
      console.error("JSON parse error:", e);
      res.json({ 
        text: result.trim() || "Unable to recognize sign",
        confidence: 0,
        success: false
      });
    }
  });
});

// ====== Get Available Models ======
app.get("/models", (req, res) => {
  const modelsDir = path.join(__dirname, "..");
  const models = [
    {
      id: "asl_cnn_lstm",
      name: "American Sign Language (ASL)",
      file: "asl_best_cnn_lstm.keras",
      language: "american",
      flag: "🇺🇸",
      description: "CNN-LSTM model for ASL A-Z alphabet recognition",
      available: fs.existsSync(path.join(modelsDir, "asl_best_cnn_lstm.keras"))
    },
    {
      id: "eth_mobilenet",
      name: "Ethiopian Sign Language",
      file: "eth_model_mobilenet_best.keras",
      language: "ethiopian",
      flag: "🇪🇹",
      description: "MobileNet model for Ethiopian sign recognition",
      available: fs.existsSync(path.join(modelsDir, "eth_model_mobilenet_best.keras"))
    }
  ];
  
  res.json(models);
});

// ====== Default Route ======
app.get("/", (req, res) => {
  res.send("✅ Backend API is running!");
});

// ====== Start Server ======
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
}).on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`❌ Port ${PORT} is already in use!`);
    console.log('Try: taskkill /F /IM node.exe');
    process.exit(1);
  } else {
    console.error('❌ Server error:', err);
  }
});
