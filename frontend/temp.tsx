.dashboard-bg {  font-family: 'Inter', Arial, sans-serif;  background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);  min-height: 100vh;}.dashboard-title {
  text-align: left;  color: #6366f1;  margin-top: 0px;  font-weight: 800;  letter-spacing: 2px;  font-size: 2.5rem;  text-shadow: 0 2px 8px #e0e7ff;}.dashboard-container { 
   display: flex;  height: 70vh;  margin: 32px auto;  
   box-shadow: 0 4px 24px rgba(99,102,241,0.08);  
   border-radius: 24px;  overflow: hidden;  background: #fff;}.sidebar {  width: 320px;  
    background: linear-gradient(135deg, #f1f5f9 0%, #e0e7ff 100%);  
    padding: 32px 24px;  box-sizing: border-box;  
    border-right: 1px solid #e0e7ff;  display: flex;  
    flex-direction: column;  gap: 24px; overflow-y: auto;}.sidebar-title {  color: #6366f1;  font-weight: 700;  font-size: 1.5rem;  margin-bottom: 16px;}.sidebar label {  color: #64748b;  font-weight: 500;  display: block;  margin-bottom: 8px;}

.sidebar input {
  width: 100%;
  margin-top: 8px;
  margin-bottom: 12px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #c7d2fe;
  font-size: 1rem;
  background: #fff;
  box-shadow: 0 1px 4px rgba(99,102,241,0.04);
}

.main-content {
  flex: 1;
  background: #fff;
}

.card {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(99,102,241,0.08);
  padding: 24px;
  max-width: 400px;
  margin: 32px auto;
  text-align: center;
}

.cute-btn {
  background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 12px 32px;
  font-size: 1.2rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(99,102,241,0.12);
  transition: background 0.2s;
}

.read-the-docs {
  text-align: center;
  color: #6366f1;
  font-weight: 500;
  margin-top: 16px;
  font-size: 1rem;
}

.squares-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 24px;
}

.square-row {
  display: flex;
}

.square {
  background: #fff;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.07);
  font-size: 1.5rem;
  width: 60px;
  height: 60px;
  margin: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.database-box {
  margin-top: 24px;
  padding: 16px;
  background: #eef2ff;
  border-radius: 12px;
}

.database-title {
  margin: 0 0 12px 0;
  color: #6366f1;
  font-size: 1.2rem;
  font-weight: 600;
}

.database-select {
  width: 100%;
  margin-top: 8px;
  margin-bottom: 12px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #c7d2fe;
  font-size: 1rem;
  background: #fff;
  box-shadow: 0 1px 4px rgba(99,102,241,0.04);
  color: #6366f1;
}

.database-time-label {
  display: block;
  margin-bottom: 8px;
  color: #64748b;
}

.database-time-input {
  width: 100%;
  margin-top: 4px;
  margin-bottom: 12px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #c7d2fe;
  font-size: 1rem;
  background: #fff;
  box-shadow: 0 1px 4px rgba(99,102,241,0.04);
}