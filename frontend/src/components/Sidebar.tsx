import React from 'react'
import '../css/Sidebar.css'

interface SidebarProps {
  selectedDb: string;
  setSelectedDb: (db: string) => void;
  dbList: string[];
  selectedVariable: string;
  setSelectedVariable: (variable: string) => void;
  variableList: string[];
  depth: number;
  setDepth: (depth: number) => void;
  startTime: string;
  setStartTime: (time: string) => void;
  selectedAlgorithm: string;
  setSelectedAlgorithm: (algorithm: string) => void;
  algorithmList: string[];
  availableYears: string[]; // Add availableYears prop
}

const Sidebar: React.FC<SidebarProps> = ({
  selectedDb, setSelectedDb,
  dbList,
  selectedVariable, setSelectedVariable,
  variableList,
  depth, setDepth,
  startTime, setStartTime,
  selectedAlgorithm, setSelectedAlgorithm,
  algorithmList,
  availableYears
}) => {
  const depthOptions: Array<{ value: number; label: string }> = [
    {value: -1, label: 'Surface'},
    { value: 0, label: '0 - 50' },
    { value: 50, label: '50 - 100' },
    { value: 100, label: '100 - 150' },
    { value: 150, label: '150 - 200' },
  ];

  return (
  <aside className="sidebar">
    
      <img
        className="sidebar-logo"
        src="/branding/logo.png"
        alt="OceanICU logo"
      />
    
    <div className="sidebar-section sidebar-dashboard">
      <div>
        <h2>Anomaly Detection</h2>
      </div>
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
          {variableList.map(variable => (
            <option key={variable} value={variable}>
              {variable}
            </option>
          ))}
          {/* Optionally, show disabled Salinity if not available */}
          {/* {!variableList.includes('so-Salinity') && (
            <option value="so-Salinity" disabled> so-Salinity (not available)</option>
          )} */}
        </select>
      </label>
    </div>
    <div className="sidebar-section">
      <h3>Depth</h3>
      <label>
        Choose Depth Range:
        <select
          value={depth}
          onChange={e => setDepth(Number(e.target.value))}
        >
          {depthOptions.map(opt => (
            opt.value >= 0 ? (
              <option key={opt.value} value={opt.value} disabled>
                {opt.label} (work in progress)
              </option>
            ) : 
            (<option key={opt.value} value={opt.value} >
              {opt.label}
            </option>
            )
          ))}
        </select>
      </label>
    </div>
    <div className="sidebar-section">
      <h3>Algorithm</h3>
      <label>
        Choose Algorithm:
        <select value={selectedAlgorithm} onChange={e => setSelectedAlgorithm(e.target.value)}>
          {algorithmList.map(alg => <option key={alg} value={alg}>{alg}</option>)}
        </select>
      </label>
    </div>
    <div className="sidebar-section">
      <h3>Anomaly Year</h3>
      <label>
        <select value={startTime} onChange={e => setStartTime(e.target.value)}>
          {availableYears.map(year => (
            <option key={year} value={year}>{year}</option>
          ))}
        </select>
      </label>
    </div>
    {/* Add regions section if needed */}
  </aside>
  )
}

export default Sidebar