const mysql = require("mysql2");
const bcrypt = require("bcryptjs");

// Admin credentials - CHANGE THESE!
const ADMIN_EMAIL = "admin@admin.com";
const ADMIN_PASSWORD = "admin123";
const ADMIN_NAME = "Admin";

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "",
  database: "sign",
});

db.connect(async (err) => {
  if (err) {
    console.error("❌ Database connection failed:", err.message);
    process.exit(1);
  }

  console.log("✅ Connected to database");

  // Hash password
  const hashedPassword = await bcrypt.hash(ADMIN_PASSWORD, 10);

  // Insert admin user
  const query = `INSERT INTO users (name, email, password, is_admin) VALUES (?, ?, ?, TRUE)`;

  db.query(query, [ADMIN_NAME, ADMIN_EMAIL, hashedPassword], (err, result) => {
    if (err) {
      if (err.code === "ER_DUP_ENTRY") {
        // Update existing user to admin
        db.query(
          "UPDATE users SET is_admin = TRUE WHERE email = ?",
          [ADMIN_EMAIL],
          (err2) => {
            if (err2) {
              console.error("❌ Error:", err2.message);
            } else {
              console.log("✅ User already exists - promoted to admin!");
              console.log(`\n📧 Email: ${ADMIN_EMAIL}`);
              console.log(`🔑 Password: ${ADMIN_PASSWORD}`);
            }
            db.end();
          }
        );
      } else {
        console.error("❌ Error:", err.message);
        db.end();
      }
    } else {
      console.log("✅ Admin user created!");
      console.log(`\n📧 Email: ${ADMIN_EMAIL}`);
      console.log(`🔑 Password: ${ADMIN_PASSWORD}`);
      console.log(`\nLogin at: http://localhost:3000/login`);
      console.log(`Then go to: http://localhost:3000/admin`);
      db.end();
    }
  });
});
