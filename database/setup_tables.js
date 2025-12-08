/**
 * Automatically create missing database tables
 */

const mysql = require("mysql2");

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "",
  database: "sign",
});

console.log("🔧 Setting up database tables...\n");

db.connect((err) => {
  if (err) {
    console.error("❌ Connection failed:", err.message);
    process.exit(1);
  }

  console.log("✅ Connected to MySQL\n");

  // Create user_activity table
  const createActivityTable = `
    CREATE TABLE IF NOT EXISTS user_activity (
      id INT AUTO_INCREMENT PRIMARY KEY,
      user_id INT NOT NULL,
      feature_used VARCHAR(50) NOT NULL,
      used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
  `;

  db.query(createActivityTable, (err) => {
    if (err) {
      console.error("❌ Failed to create user_activity table:", err.message);
      db.end();
      process.exit(1);
    }

    console.log("✅ user_activity table created/verified");

    // Verify tables exist
    db.query("SHOW TABLES", (err, results) => {
      if (err) {
        console.error("❌ Error checking tables:", err.message);
      } else {
        console.log("\n📋 Tables in database:");
        results.forEach(row => {
          console.log(`   - ${Object.values(row)[0]}`);
        });
      }

      db.end();
      console.log("\n✅ Database setup complete!");
      console.log("You can now restart your backend server.");
    });
  });
});
