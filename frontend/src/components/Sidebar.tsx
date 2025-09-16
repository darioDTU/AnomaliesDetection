import React from 'react'
import '../css/Dashboard.css'

interface SidebarProps {
  latitude: number;
  setLatitude: (lat: number) => void;
  longitude: number;
  setLongitude: (lng: number) => void;
  selectedDb: string;
  setSelectedDb: (db: string) => void;
  dbList: string[];
  selectedVariable: string;
  setSelectedVariable: (variable: string) => void;
  variableList: string[];
  startTime: string;
  setStartTime: (time: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  latitude, setLatitude,
  longitude, setLongitude,
  selectedDb, setSelectedDb,
  dbList,
  selectedVariable, setSelectedVariable,
  variableList,
  startTime, setStartTime,
}) => (
  <aside className="sidebar">
    <div className="sidebar-section sidebar-dashboard">
      <div>
        <h2>Dashboard</h2>
        <p>Configure detection parameters and analyze regions</p>
      </div>
    </div>
    <div className="sidebar-section">
      <h3>Coordinates</h3>
      <label>
        Latitude:
        <input type="number" value={latitude} onChange={e => setLatitude(Number(e.target.value))} />
      </label>
      <label>
        Longitude:
        <input type="number" value={longitude} onChange={e => setLongitude(Number(e.target.value))} />
      </label>
    </div>
    <div className="sidebar-section">
      <h3>Database</h3>
      <label>
        Choose Database:
        <select value={selectedDb} onChange={e => setSelectedDb(e.target.value)}>
          {dbList.map(db => <option key={db} value={db}>{db}</option>)}
        </select>
      </label>
      <label>
        Choose Variable:
        <select value={selectedVariable} onChange={e => setSelectedVariable(e.target.value)}>
          {variableList.map(variable => <option key={variable} value={variable}>{variable}</option>)}
        </select>
      </label>
    </div>
    <div className="sidebar-section">
      <h3>Starting Year</h3>
      <input
        type="number"
        value={startTime}
        min="1900"
        max="2100"
        onChange={e => setStartTime(e.target.value)}
        placeholder="YYYY"
      />
    </div>
    {/* Add regions section if needed */}
  </aside>
)

export default Sidebar