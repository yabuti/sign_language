import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiUsers, FiActivity, FiTrash2, FiShield, FiShieldOff, FiLogOut, FiVideo, FiImage, FiMic } from "react-icons/fi";
import axios from "axios";
import "./AdminDashboard.css";

function AdminDashboard() {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userActivity, setUserActivity] = useState([]);
  const navigate = useNavigate();

  const token = localStorage.getItem("token");

  // Prevent access without login
  if (!token) {
    window.location.href = "/login";
  }

  // Fetch all users
  const fetchUsers = async () => {
    try {
      const res = await axios.get("http://localhost:5000/admin/users", {
        headers: { Authorization: `Bearer ${token}` },
      });

      setUsers(res.data);
    } catch (err) {
      console.error("Fetch users error:", err);
      alert("Unable to load admin data. You may not be an admin.");
    } finally {
      setLoading(false);
    }
  };

  // Fetch user activity
  const fetchUserActivity = async (userId) => {
    try {
      const res = await axios.get(`http://localhost:5000/admin/users/${userId}/activity`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUserActivity(res.data);
      setSelectedUser(userId);
    } catch (err) {
      console.error("Fetch activity error:", err);
    }
  };

  useEffect(() => {
    fetchUsers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Delete a user
  const deleteUser = async (id) => {
    if (!window.confirm("Are you sure you want to delete this user?")) return;

    try {
      await axios.delete(`http://localhost:5000/admin/users/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      fetchUsers();
      if (selectedUser === id) {
        setSelectedUser(null);
        setUserActivity([]);
      }
    } catch (err) {
      console.error("Delete user error:", err);
      alert("Failed to delete user.");
    }
  };

  // Promote to admin
  const promoteUser = async (id) => {
    try {
      await axios.put(
        `http://localhost:5000/admin/users/${id}/make-admin`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );

      fetchUsers();
    } catch (err) {
      console.error("Promote error:", err);
      alert("Failed to promote user.");
    }
  };

  // Demote admin
  const demoteUser = async (id) => {
    try {
      await axios.put(
        `http://localhost:5000/admin/users/${id}/remove-admin`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );

      fetchUsers();
    } catch (err) {
      console.error("Demote error:", err);
      alert("Failed to demote user.");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    navigate("/login");
  };

  // Loading screen
  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner-large"></div>
        <h2>Loading Admin Dashboard...</h2>
      </div>
    );
  }

  return (
    <div className="admin-page">
      {/* Header */}
      <div className="admin-header">
        <h1>
          <FiShield size={32} />
          Admin Dashboard
        </h1>
        <button onClick={handleLogout} className="logout-btn">
          <FiLogOut size={18} />
          Logout
        </button>
      </div>

      {/* Stats Cards */}
      <div className="stats-container">
        <div className="stat-card">
          <FiUsers size={40} />
          <div>
            <h3>{users.length}</h3>
            <p>Total Users</p>
          </div>
        </div>
        <div className="stat-card">
          <FiActivity size={40} />
          <div>
            <h3>{users.reduce((sum, u) => sum + (u.activity_count || 0), 0)}</h3>
            <p>Total Activities</p>
          </div>
        </div>
        <div className="stat-card">
          <FiShield size={40} />
          <div>
            <h3>{users.filter(u => u.is_admin).length}</h3>
            <p>Admins</p>
          </div>
        </div>
      </div>

      {/* Users Table */}
      <div className="table-container">
        <h2>
          <FiUsers size={24} />
          User Management
        </h2>
        <table className="admin-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Email</th>
              <th>Role</th>
              <th>Activities</th>
              <th>Last Active</th>
              <th>Joined</th>
              <th>Actions</th>
            </tr>
          </thead>

          <tbody>
            {users.length === 0 ? (
              <tr>
                <td colSpan="8" style={{ textAlign: "center" }}>
                  No users found.
                </td>
              </tr>
            ) : (
              users.map((user) => (
                <tr key={user.id} className={selectedUser === user.id ? "selected-row" : ""}>
                  <td>{user.id}</td>
                  <td>{user.name || "—"}</td>
                  <td>{user.email || "—"}</td>
                  <td>
                    <span className={user.is_admin ? "badge admin-badge" : "badge user-badge"}>
                      {user.is_admin ? "Admin" : "User"}
                    </span>
                  </td>
                  <td>
                    <button 
                      className="activity-btn"
                      onClick={() => fetchUserActivity(user.id)}
                    >
                      {user.activity_count || 0} activities
                    </button>
                  </td>
                  <td>
                    {user.last_activity
                      ? new Date(user.last_activity).toLocaleString()
                      : "Never"}
                  </td>
                  <td>
                    {user.created_at
                      ? new Date(user.created_at).toLocaleDateString()
                      : "—"}
                  </td>

                  <td className="action-buttons">
                    {!user.is_admin && (
                      <button
                        className="action-btn promote-btn"
                        onClick={() => promoteUser(user.id)}
                        title="Promote to Admin"
                      >
                        <FiShield size={16} />
                      </button>
                    )}

                    {user.is_admin && (
                      <button
                        className="action-btn demote-btn"
                        onClick={() => demoteUser(user.id)}
                        title="Remove Admin"
                      >
                        <FiShieldOff size={16} />
                      </button>
                    )}

                    <button
                      className="action-btn delete-btn"
                      onClick={() => deleteUser(user.id)}
                      title="Delete User"
                    >
                      <FiTrash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Activity Panel */}
      {selectedUser && (
        <div className="activity-panel">
          <h2>
            <FiActivity size={24} />
            User Activity Log
          </h2>
          {userActivity.length === 0 ? (
            <p className="no-activity">No activity recorded yet.</p>
          ) : (
            <div className="activity-list">
              {userActivity.map((activity, index) => (
                <div key={index} className="activity-item">
                  <span className={`feature-badge feature-${activity.feature_used}`}>
                    {activity.feature_used === "webcam" && <><FiVideo size={14} /> Webcam</>}
                    {activity.feature_used === "image" && <><FiImage size={14} /> Image</>}
                    {activity.feature_used === "voice" && <><FiMic size={14} /> Voice</>}
                    {!["webcam", "image", "voice"].includes(activity.feature_used) && activity.feature_used}
                  </span>
                  <span className="activity-time">
                    {new Date(activity.used_at).toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default AdminDashboard;
