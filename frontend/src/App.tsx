import './css/Dashboard.css'
import './css/Api.css'
import {ENDPOINTS} from './service/endpoints.ts'
import {BASE_URL,fetchData, postData} from './service/api.ts'
import Map from './components/Map.tsx'
import Sidebar from './components/Sidebar.tsx';
import { useState } from 'react'

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
  const [startTime, setStartTime] = useState('');
  const dbList = ['cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m', 'db2', 'db3'];
  const variableList = ['thetao', 'Salinity'];
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

  return (
    <div className="dashboard-bg">
      <header className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Anomalies Detection</h1>
          <p className="dashboard-subtitle">Advanced geospatial analytics dashboard</p>
        </div>
      </header>
      <div className="dashboard-main">
        <Sidebar
          latitude={latitude}
          longitude={longitude}
          selectedDb={selectedDb}
          startTime={startTime}
          selectedVariable={selectedVariable}
          selectedAlgorithm={selectedAlgorithm}
          setLatitude={setLatitude}
          setLongitude={setLongitude}
          setSelectedDb={setSelectedDb}
          setStartTime={setStartTime}
          setSelectedVariable={setSelectedVariable}
          setSelectedAlgorithm={setSelectedAlgorithm}
          dbList={dbList}
          variableList={variableList}
          algorithmList={algorithmList}
          
        />
        <main className="dashboard-content">
          <Map
            latitude={latitude}
            longitude={longitude}
            rectangleBounds={rectangleBounds}
            onMapClick={handleMapClick}
          />
          <button
            className="cute-btn"
            onClick={handleApiCall}
            disabled={!paramsReady || loading}
            style={{ margin: '16px 0' }}
          >
            Run 
          </button>
          {hasPressButton && (
          <>
          {error && <div className='error'>{error}</div>}
          {loading ? <div className='loading'>Loading...</div> : 
            <div className='api-result'>
            <div className="api-extra">
              <img
                src={`${BASE_URL}/${ENDPOINTS.showImage}?t=${Date.now()}`}
                alt="API result"
                className="api-image"
              />
              <div className="api-stats">
                <h4>Statistics</h4>
                <ul>
                  {stats 
                  ? Object.entries(stats).map(([key, value]) => (
                      <li key={key}>
                        <strong>{key}:</strong> {value as string}
                      </li>))
                      : <li>No statistics available.</li>}
                </ul>
              </div>
            </div>
            </div>}
          </>
          )}

        </main>
      </div>
    </div>
  )
}

export default App
