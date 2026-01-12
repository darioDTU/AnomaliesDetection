import './css/Dashboard.css'
import './css/Api.css'
import './css/Map.css'
import './css/Sidebar.css'
import {ENDPOINTS} from './service/endpoints.ts'
import {BASE_URL,fetchData, postData} from './service/api.ts'
import { dbVariables, dbYears } from './service/db_variables.ts'
import Map from './components/Map.tsx'
import Sidebar from './components/Sidebar.tsx';
import { useState, useEffect } from 'react'

export type PipelineRequest = {
  dataset: string;
  latitude: number;
  longitude: number;
  starting_time: string;
  variable: string;
};

function App() {
  const [latitude, setLatitude] = useState(0);
  const [longitude, setLongitude] = useState(0);
  const [selectedDb, setSelectedDb] = useState('cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m');
  const [selectedVariable, setSelectedVariable] = useState('thetao');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('Classic');
  const [startTime, setStartTime] = useState(
    (dbYears[selectedDb] && dbYears[selectedDb][0]) || ""
  );
  const dbList = ['cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m', 'cmems_mod_glo_phy_my_0.083deg_P1D-m', 'cmems_mod_glo_bgc_my_0.25deg_P1D-m'];
  const variableList = dbVariables[selectedDb] || [];
  const algorithmList = ['POT', 'Classic'];

  const [, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  const [hasPressButton, setHasPressButton] = useState(false);
  const paramsReady = latitude !== null && longitude !== null && selectedDb && startTime && selectedVariable
  const requestBody : PipelineRequest = {
    dataset: selectedDb,
    latitude: latitude,
    longitude: longitude,
    starting_time: startTime,
    variable: selectedVariable
  };
  const handleApiCall = async () => {
    if (!paramsReady) return;
    setLoading(true);
    setError(null); 
    setHasPressButton(true);
    try {
      // const result = await fetchData(ENDPOINTS.run);
      // setData(result);
      let pipelineResult;
    if (selectedAlgorithm === 'POT') {
      pipelineResult = await postData(ENDPOINTS.runPOT, requestBody);
    } else {
      pipelineResult = await postData(ENDPOINTS.runClassic, requestBody);
    }
      setData(pipelineResult);
      // Fetch stats after running the pipeline
      const stats = await fetchData(ENDPOINTS.getStats);
      setStats(stats.Statistics);
    } catch (error) {
      console.error('Error fetching data:', error);
      let message = 'Unknown error';
      if (error && typeof error === 'object' && 'message' in error) {
        message = (error as { message: string }).message;
      } else if (typeof error === 'string') {
        message = error;
      }
      setError(`Failed to load: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const RECT_RESOLUTION = 2;
  const halfRes = RECT_RESOLUTION / 2;

  const rectangleBounds: [number, number][] = [
    [longitude - halfRes, latitude - halfRes], // SW
    [longitude + halfRes, latitude - halfRes], // SE
    [longitude + halfRes, latitude + halfRes], // NE
    [longitude - halfRes, latitude + halfRes], // NW
    [longitude - halfRes, latitude - halfRes], // Close polygon
  ];

  const handleMapClick = (lng : number, lat: number) => {
    setLongitude(lng);
    setLatitude(lat);
  };

  useEffect(() => {
    setStartTime((dbYears[selectedDb] && dbYears[selectedDb][0]) || "");
    const availableVars = dbVariables[selectedDb] || [];
    if (availableVars.length > 0 && !availableVars.includes(selectedVariable))
      setSelectedVariable(availableVars[0]);
  }, [selectedDb]);

  return (
    <div className="dashboard-bg">
      {/* <header className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Anomalies Detection</h1>
          <p className="dashboard-subtitle">Advanced geospatial analytics dashboard</p>
        </div>
      </header> */}
      <div className="top-section">
        <aside className="sidebar">
          <Sidebar
            selectedDb={selectedDb}
            startTime={startTime}
            selectedVariable={selectedVariable}
            selectedAlgorithm={selectedAlgorithm}
            setSelectedDb={setSelectedDb}
            setStartTime={setStartTime}
            setSelectedVariable={setSelectedVariable}
            setSelectedAlgorithm={setSelectedAlgorithm}
            dbList={dbList}
            variableList={variableList}
            algorithmList={algorithmList}
            availableYears={dbYears[selectedDb] || []}
            
          />
        </aside>
        <div className="map-container">
          <button
            className="cute-btn run-btn-top"
            onClick={handleApiCall}
            disabled={!paramsReady || loading}
          >
            {loading ? 'Running...' : 'Run'}
          </button>
          <Map
              latitude={latitude}
              longitude={longitude}
              rectangleBounds={rectangleBounds}
              onMapClick={handleMapClick}
            />
          <div className="floating-sidebar">
            <div className="coord-adjust-section">
              <h3>Coordinates</h3>
              <div className="coord-input-group">
                <label>Latitude</label>
                <input type="number" value={latitude} onChange={e => setLatitude(Number(e.target.value))} />
              </div>
              <div className="coord-input-group">
                <label>Longitude</label>
                <input type="number" value={longitude} onChange={e => setLongitude(Number(e.target.value))} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {!loading && hasPressButton && error==null && stats!==null && (
        <div className="modal-overlay" onClick={() => setHasPressButton(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setHasPressButton(false)}>×</button>
            <div className="modal-header">
              <h2>Analysis Results</h2>
            </div>
            <div className="modal-body">
              <img
                src={`${BASE_URL}/${ENDPOINTS.showImage}?t=${Date.now()}`}
                alt="API result"
                className="modal-image"
              />
              <div className="modal-stats">
                <h4>Statistics</h4>
                <ul>
                  {stats 
                  ? Object.entries(stats).map(([key, value]) => (
                      <li key={key}>
                        <strong>{key}:</strong> {value as string}
                      </li>))
                    : <li>No statistics available for area with no anomalies</li>}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && hasPressButton && (
        <div className="modal-overlay" onClick={() => setHasPressButton(false)}>
          <div className="modal-content error-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setHasPressButton(false)}>×</button>
            <div className="modal-header error-header">
              <h2>Error</h2>
            </div>
            <div className="modal-body">
              <p>{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
